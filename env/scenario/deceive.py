import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


def distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))


class Scenario(BaseScenario):
    def __init__(self, n_targets, goal):
        self.n_targets = n_targets
        self.goal = goal
        self.multiplier = 1

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 2
        world.num_agents = num_agents
        num_adversaries = 1
        num_landmarks = self.n_targets
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.2
            agent.accel = 10.0
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.1
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        world.agents[0].color = np.array([0.85, 0.35, 0.35])
        for i in range(1, world.num_agents):
            world.agents[i].color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.15, 0.15, 0.15])
        # set goal landmark
        goal = world.landmarks[self.goal]
        goal.color = np.array([0.15, 0.65, 0.15])
        for agent in world.agents:
            agent.goal_a = goal
        # set random initial states
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.live = True
            agent.last_distance = distance(agent.state.p_pos, goal.state.p_pos)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            return np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        else:
            dists = []
            for l in world.landmarks:
                dists.append(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
            dists.append(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
            return tuple(dists)

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    # def agent_reward(self, agent, world):
    #     # Rewarded based on how close any good agent is to the goal landmark, and how far the adversary is from it
    #     shaped_reward = True
    #     shaped_adv_reward = True
    #
    #     # Calculate negative reward for adversary
    #     adversary_agents = self.adversaries(world)
    #     if shaped_adv_reward:  # distance-based adversary reward
    #         adv_rew = sum([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in adversary_agents])
    #     else:  # proximity-based adversary reward (binary)
    #         adv_rew = 0
    #         for a in adversary_agents:
    #             if np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) < 2 * a.goal_a.size:
    #                 adv_rew -= 5
    #
    #     # Calculate positive reward for agents
    #     good_agents = self.good_agents(world)
    #     if shaped_reward:  # distance-based agent reward
    #         pos_rew = -min(
    #             [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents])
    #     else:  # proximity-based agent reward (binary)
    #         pos_rew = 0
    #         if min([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents]) \
    #                 < 2 * agent.goal_a.size:
    #             pos_rew += 5
    #         pos_rew -= min(
    #             [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents])
    #     # return pos_rew + adv_rew
    #     return pos_rew
    #
    # def adversary_reward(self, agent, world):
    #     # Rewarded based on proximity to the goal landmark
    #     shaped_reward = True
    #     if shaped_reward:  # distance-based reward
    #         return -np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
    #     else:  # proximity-based reward (binary)
    #         adv_rew = 0
    #         if np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))) < 2 * agent.goal_a.size:
    #             adv_rew += 5
    #         return adv_rew

    def agent_reward(self, agent, world):
        # distance = sum([np.sqrt(np.sum(np.square(agent.state.p_pos - landmark.state.p_pos))) for landmark in world.landmarks])
        if not agent.live:
            return 0.
        dist = distance(agent.state.p_pos, agent.goal_a.state.p_pos)
        shaping = agent.last_distance - dist
        agent.last_distance = dist

        rew = 0.
        if distance(agent.state.p_pos, agent.goal_a.state.p_pos) < agent.size + agent.goal_a.size:
            rew += 50.

        for landmark in world.landmarks:
            if distance(agent.state.p_pos, landmark.state.p_pos) < agent.size + landmark.size:
                agent.live = False

        return shaping + rew

    def adversary_reward(self, agent, world):
        if not agent.live:
            return 0.
        dist = distance(agent.state.p_pos, agent.goal_a.state.p_pos)
        shaping = agent.last_distance - dist
        agent.last_distance = dist

        rew = 0.
        if distance(agent.state.p_pos, agent.goal_a.state.p_pos) < agent.size + agent.goal_a.size:
            rew += 50.

        for landmark in world.landmarks:
            if distance(agent.state.p_pos, landmark.state.p_pos) < agent.size + landmark.size:
                agent.live = False

        return shaping + rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:
            entity_color.append(entity.color)
        # communication of all other agents
        all_pos = []
        for other in world.agents:
            all_pos.append(other.state.p_pos - agent.state.p_pos)

        if not agent.adversary:
            # return np.concatenate([agent.goal_a.state.p_pos - agent.state.p_pos] + entity_pos + other_pos)
            return np.concatenate(entity_pos + all_pos)
        else:
            return np.concatenate(entity_pos + all_pos)

    def is_touching(self, agent, landmark):
        return np.sqrt(np.sum(np.square(agent.state.p_pos - landmark.state.p_pos))) <= agent.size + landmark.size