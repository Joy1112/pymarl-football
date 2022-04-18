import numpy as np
from copy import deepcopy

from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from .episode_runner import EpisodeRunner
from wurl.apwd import assign_reward
from wurl.buffer import Cache


class URLRunner(EpisodeRunner):

    def __init__(self, args, logger):
        super(URLRunner, self).__init__(args, logger)
        self.num_modes = args.num_modes

        self.caches_empty = [Cache(args.cache_size) for _ in range(self.num_modes)]
        self.caches_dict = {}
    
    def setup(self, scheme, groups, preprocess, macs):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.macs = macs
    
    def reset(self, mode_id=None):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0
        # self.mode_id = np.random.randint(0, high=self.num_modes)
        self.mode_id = mode_id
        self.pseudo = False

    def run(self, test_mode=False, mode_id=None):
        self.reset(mode_id)

        terminated = False
        episode_return = 0
        episode_pseudo_return = 0
        self.macs[self.mode_id].init_hidden(batch_size=self.batch_size)

        control_traj = []
        control_traj_reward = []
        observations = self.env.get_obs()
        url_feature = self.build_url_feature(observations)
        active_agents = tuple(url_feature[-3:])
        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [observations]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.macs[self.mode_id].select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            # Fix memory leak
            cpu_actions = actions.to("cpu").numpy()
            
            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": cpu_actions,
                "ori_reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            control_traj.append(url_feature)
            control_traj_reward.append(reward)

            # assert the controlling agents are the same
            new_observations = self.env.get_obs()
            new_url_feature = self.build_url_feature(new_observations)

            if tuple(new_url_feature[-3:]) == active_agents and len(control_traj) < self.args.max_control_len:
                controller_updated = False
            else:
                controller_updated = True
            
            # when the control traj ended, calculate the pseudo rewards.
            if terminated or controller_updated:
                if self.t_env >= self.args.start_steps:
                    pseudo_rewards = self.calc_pseudo_rewards(active_agents, control_traj)
                    if pseudo_rewards is not None:
                        self.pseudo = True
                        pseudo_rewards_data = {
                            "reward": pseudo_rewards,
                        }
                        self.batch.update(pseudo_rewards_data, ts=slice((self.t - len(pseudo_rewards)) + 1, self.t + 1))
                        episode_pseudo_return += np.sum(pseudo_rewards)
                control_traj = []
            
            # insert the url_feature into the cache.
            if active_agents not in self.caches_dict.keys():
                self.caches_dict[active_agents] = deepcopy(self.caches_empty)
            self.caches_dict[active_agents][self.mode_id].push((url_feature, ))

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1
            observations = new_observations
            url_feature = new_url_feature
            active_agents = tuple(url_feature[-3:])

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.macs[self.mode_id].select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        # Fix memory leak
        cpu_actions = actions.to("cpu").numpy()
        self.batch.update({"actions": cpu_actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        # cur_returns.append(episode_return)
        cur_returns.append(episode_pseudo_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.macs[self.mode_id].action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.macs[self.mode_id].action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def build_url_feature(self, observations):
        assert self.env.n_agents == 2, "only support 2 agents now."
        url_feature = []
        for obs in observations:
            url_feature.append(obs[0:6])        # [ego_positions, relative_positions_1, relative_positions_2]
            url_feature.append(obs[12:16])      # [relative_opponent_positions_1, relative_opponent_positions_2]
            url_feature.append(obs[20:22])      # [relative_ball_x, relative_ball_y]

        obs = observations[0]
        url_feature.append(obs[22:23])             # [ball_z]

        url_feature.append(obs[6:12])           # [left_team_movements]
        url_feature.append(obs[16:20])          # [right_team_movements]
        url_feature.append(obs[23:])            # other infos
        
        return np.concatenate(url_feature, axis=0)

    def calc_pseudo_rewards(self, active_agents, control_traj, control_traj_reward=None):
        try:
            target_data_batches = [list(self.caches_dict[active_agents][i].dump(self.args.max_control_len))[0] for i in range(self.num_modes)]
            pseudo_rewards = assign_reward(np.array(control_traj), target_data_batches, traj_reward=control_traj_reward)
        except:
            return None

        return pseudo_rewards.reshape(-1, 1)
