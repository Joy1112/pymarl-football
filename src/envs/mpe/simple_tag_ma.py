# import os
# import sys
# import torch
# sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from .. import MultiAgentEnv
# from multiagentenv import MultiAgentEnv
from .scenarios import simple_tag_new
import gym
import numpy as np


class SimpleTag(MultiAgentEnv):
    """Only the adversaries are controlled."""
    def __init__(
        self,
        n_agents=4,
        time_limit=100,
        time_step=0,
        map_name='simple_tag',
        seed=0,
        **kwargs,
    ):
        self.n_agents = n_agents
        self.episode_limit = time_limit
        self.time_step = time_step
        self.env_name = map_name
        self.seed_i = seed
        self.env = simple_tag_new.parallel_env(
            num_good=1,
            num_adversaries=self.n_agents - 1,
            num_obstacles=0,
            max_cycles=self.episode_limit,
            continuous_actions=False
        )
        self.env.seed(self.seed_i)

        # self.action_space = [self.env.action_space('adversary_' + str(i)) for i in range(self.n_agents)]
        # self.observation_space = [self.env.observation_space('adversary_' + str(i)) for i in range(self.n_agents)]
        self.agents = self.env.possible_agents
        self.action_space = [self.env.action_space(agent) for agent in self.agents]
        self.observation_space = [self.env.observation_space(agent) for agent in self.agents]

        self.n_actions = self.action_space[0].n

        self.unit_dim = self.obs_dim = self.observation_space[0].shape[0]
        self.obs_dict = None

    def get_global_state(self):
        return self.env.state()

    def step(self, actions):
        """Returns reward, terminated, info."""
        self.time_step += 1
        actions_list = actions.to('cpu').numpy().tolist()
        self.obs_dict, original_rewards, dones, infos = self.env.step({agent: actions_list[i] for i, agent in enumerate(self.agents)})
        # rewards = list(original_rewards)

        # only done when reach the episode_limit
        if self.time_step >= self.episode_limit:
            done = True
        else:
            done = False

        # if sum(rewards) <= 0:
        #     return -int(done), done, infos

        # return 100, done, infos
        return 0, done, {}

    def get_obs(self):
        """Returns all agent observations in a list."""
        return list(self.obs_dict.values())

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        return self.obs_dict[self.agents[agent_id]]

    def get_obs_size(self):
        """Returns the size of the observation."""
        return self.obs_dim

    def get_state(self):
        """Returns the global state."""
        return self.get_global_state()

    def get_state_size(self):
        """Returns the size of the global state."""
        return self.env.state_space.shape[0]

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        return [[1 for _ in range(self.n_actions)] for agent_id in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        return self.get_avail_actions()[agent_id]
    
    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.action_space[0].n

    def reset(self):
        """Returns initial observations and states."""
        self.time_step = 0
        self.obs_dict = self.env.reset()

        return self.get_obs(), self.get_global_state()

    def render(self):
        pass

    def close(self):
        self.env.close()

    def seed(self, seed):
        self.seed_i = seed
        self.env.seed(self.seed_i)

    def save_replay(self):
        """Save a replay."""
        pass

    def get_stats(self):
        return  {}


if __name__ == "__main__":
    env = SimpleTag()
    env.seed(1234342)
    obs, state = env.reset()
    print('state_size: ', env.get_state_size())
    print('obs_size: ', env.get_obs_size())
    print('avail_actions: ', env.get_avail_actions())
    print('avail_actions for good_agent: ', env.get_avail_agent_actions(3))
    print('obs: ', env.get_obs())
    print('obs for adversary_1: ', env.get_obs_agent(1))
    print('obs for good_agent: ', env.get_obs_agent(3))
    print('step returns: ', env.step(torch.as_tensor([0, 1, 1, 0])))

