import numpy as np
import torch
import time
from copy import deepcopy

from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from .episode_runner import EpisodeRunner
from url_algo import REGISTRY as url_assigner_REGISTRY
from url_algo.buffer import Cache


class URLRunner(EpisodeRunner):

    def __init__(self, args, logger):
        super(URLRunner, self).__init__(args, logger)
        self.num_modes = args.num_modes

        if self.args.url_algo == "diayn":
            self.cache = Cache(args.cache_size)
        else:
            self.caches_empty = [Cache(args.cache_size) for _ in range(self.num_modes)]
            self.caches_dict = {}

        self.url_assigner_fn = url_assigner_REGISTRY[self.args.url_algo]
    
    def setup(self, scheme, groups, preprocess, macs, disc_trainer=None):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.macs = macs
        self.disc_trainer = disc_trainer
    
    def create_env(self, env_args):
        del self.env
        self.env = env_REGISTRY[self.args.env](**env_args)
    
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

        if self.args.url_algo == "gwd":
            url_feature, active_agents = self.build_graph(observations)
        else:
            url_feature, active_agents = self.build_url_feature(observations)

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
            if self.args.url_algo == "gwd":
                new_url_feature, new_active_agents = self.build_graph(new_observations)
            else:
                new_url_feature, new_active_agents = self.build_url_feature(new_observations)
                
            if new_active_agents == active_agents and len(control_traj) < self.args.max_control_len:
                controller_updated = False
            else:
                controller_updated = True
            
            # when the control traj ended, calculate the pseudo rewards.
            if terminated or controller_updated:
                if self.t_env >= self.args.start_steps:
                    pseudo_rewards = self.calc_pseudo_rewards(active_agents, control_traj, control_traj_reward)
                    if pseudo_rewards is not None:
                        self.pseudo = True
                        pseudo_rewards_data = {
                            "reward": pseudo_rewards,
                        }
                        self.batch.update(pseudo_rewards_data, ts=slice((self.t - len(pseudo_rewards)) + 1, self.t + 1))
                        episode_pseudo_return += np.sum(pseudo_rewards)
                control_traj = []
            
            # insert the url_feature into the cache.
            if self.args.url_algo == "diayn":
                self.cache.push((np.array([self.mode_id]), url_feature))
            else:
                if active_agents not in self.caches_dict.keys():
                    self.caches_dict[active_agents] = deepcopy(self.caches_empty)
                self.caches_dict[active_agents][self.mode_id].push((url_feature, ))

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1
            observations = new_observations
            url_feature = new_url_feature
            active_agents = new_active_agents

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
        if self.args.env == "gfootball":
            assert self.env.n_agents == 2, "only support 2 agents now."
            url_feature_list = []
            for obs in observations:
                url_feature_list.append(obs[0:6])        # [ego_positions, relative_positions_1, relative_positions_2]
                url_feature_list.append(obs[12:16])      # [relative_opponent_positions_1, relative_opponent_positions_2]
                url_feature_list.append(obs[20:22])      # [relative_ball_x, relative_ball_y]

            obs = observations[0]
            url_feature_list.append(obs[22:23])             # [ball_z]

            url_feature_list.append(obs[6:12])           # [left_team_movements]
            url_feature_list.append(obs[16:20])          # [right_team_movements]
            url_feature_list.append(obs[23:])            # other infos
            
            url_feature = np.concatenate(url_feature_list, axis=0)
            active_agents = tuple(url_feature[-3:])
        elif self.args.env == "mpe":
            url_feature_list = []
            for obs in observations:
                url_feature_list.append(obs[2:4])       # [ego_positions]
                if self.args.url_velocity:
                    url_feature_list.append(obs[0:2])   # [ego_velocity]

            url_feature = np.concatenate(url_feature_list, axis=0)
            active_agents = 1
        else:
            raise NotImplementedError

        return url_feature, active_agents
    
    def build_graph(self, observations):
        if self.args.env == "gfootball":
            # assert self.env.n_agents == 3, "only support 3 agents now."
            agents_pos_x, agents_pos_y = [], []
            for obs in observations:
                agents_pos_x.append(obs[0])
                agents_pos_y.append(obs[1])
            
            obs = observations[0]
            if self.args.opponent_graph:
                agents_pos_x.append(obs[12])
                agents_pos_y.append(obs[13])
            
            if self.args.ball_graph:
                agents_pos_x.append(obs[20])
                agents_pos_y.append(obs[21])

            active_agents = tuple(obs[-3:])
        elif self.args.env == "mpe":
            agents_pos_x, agents_pos_y = [], []
            for obs in observations:
                agents_pos_x.append(obs[2])
                agents_pos_y.append(obs[3])
            active_agents = 1
        elif self.args.env == 'sc2':
            if self.args.env_args['map_name'] == 'corridor':
                agents_pos_x, agents_pos_y, health = [], [], []
                
                active_agents=1
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        agents_pos_x = torch.as_tensor(agents_pos_x).reshape(-1, 1)
        agents_pos_y = torch.as_tensor(agents_pos_y).reshape(-1, 1)

        relative_pos_x = agents_pos_x - agents_pos_x.T
        relative_pos_y = agents_pos_y - agents_pos_y.T

        url_graph = torch.sqrt(relative_pos_x ** 2 + relative_pos_y ** 2)

        return url_graph, active_agents

    def calc_pseudo_rewards(self, active_agents, control_traj, control_traj_reward=None):
        try:
            target_data_batches = None
            if self.args.url_algo != "diayn":
                target_data_batches = [list(self.caches_dict[active_agents][i].dump(self.args.max_control_len))[0] for i in range(self.num_modes)]
            pseudo_rewards = self.url_assigner_fn(
                traj_data=control_traj,
                target_data_batches=target_data_batches,
                ot_hyperparams=self.args.ot_hyperparams,
                mode_id=self.mode_id,
                disc_trainer=self.disc_trainer,
                num_modes=self.args.num_modes,
                pseudo_reward_scale=self.args.pseudo_reward_scale,
                reward_scale=self.args.reward_scale,
                norm_reward=self.args.norm_reward,
                traj_reward=control_traj_reward,
                device="cuda",
                use_batch_apwd=self.args.batch_apwd
            )
        except:
            return None

        return pseudo_rewards.reshape(-1, 1)
