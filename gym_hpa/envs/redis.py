import csv
import datetime
import logging
import time
from statistics import mean

import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from datetime import datetime

from gym_hpa.envs.deployment import get_max_cpu, get_max_mem, get_max_traffic, get_redis_deployment_list
from gym_hpa.envs.util import save_to_csv, get_cost_reward, get_latency_reward_redis, get_num_pods

# MIN and MAX Replication
MIN_REPLICATION = 1
MAX_REPLICATION = 8

MAX_STEPS = 25  # MAX Number of steps per episode

# Possible Actions (Discrete)
ACTION_DO_NOTHING = 0
ACTION_ADD_1_REPLICA = 1
ACTION_ADD_2_REPLICA = 2
ACTION_ADD_3_REPLICA = 3
ACTION_ADD_4_REPLICA = 4
ACTION_ADD_5_REPLICA = 5
ACTION_ADD_6_REPLICA = 6
ACTION_ADD_7_REPLICA = 7
ACTION_TERMINATE_1_REPLICA = 8
ACTION_TERMINATE_2_REPLICA = 9
ACTION_TERMINATE_3_REPLICA = 10
ACTION_TERMINATE_4_REPLICA = 11
ACTION_TERMINATE_5_REPLICA = 12
ACTION_TERMINATE_6_REPLICA = 13
ACTION_TERMINATE_7_REPLICA = 14

# Deployments
DEPLOYMENTS = ["redis-leader", "redis-follower"]

# Action Moves
MOVES = ["None", "Add-1", "Add-2", "Add-3", "Add-4", "Add-5", "Add-6", "Add-7",
         "Stop-1", "Stop-2", "Stop-3", "Stop-4", "Stop-5", "Stop-6", "Stop-7"]

# IDs
ID_DEPLOYMENTS = 0
ID_MOVES = 1

ID_MASTER = 0
ID_SLAVE = 1

# Reward objectives
LATENCY = 'latency'
COST = 'cost'


class Redis(gym.Env):
    """Horizontal Scaling for Redis in Kubernetes - an OpenAI gym environment"""
    metadata = {'render.modes': ['human', 'ansi', 'array']}

    def __init__(self, k8s=False, goal_reward=COST, waiting_period=0.3):
        # Define action and observation space
        # They must be gym.spaces objects

        super(Redis, self).__init__()

        self.k8s = k8s
        self.name = "redis_gym"
        self.__version__ = "0.0.1"
        self.seed()
        self.goal_reward = goal_reward
        self.waiting_period = waiting_period  # seconds to wait after action

        logging.info("[Init] Env: {} | K8s: {} | Version {} |".format(self.name, self.k8s, self.__version__))

        # Current Step
        self.current_step = 0

        # Actions identified by integers 0-n -> 15 actions!
        self.num_actions = 15

        # Multi-Discrete version
        # Deployment: Discrete 2 - Master[0], Slave[1]
        # Action: Discrete 9 - None[0], Add-1[1], Add-2[2], Add-3[3], Add-4[4],
        #                      Stop-1[5], Stop-2[6], Stop-3[7], Stop-4[8]

        self.action_space = spaces.MultiDiscrete([2, self.num_actions])

        # Observations: 22 Metrics! -> 2 * 11 = 22
        # "number_pods"                     -> Number of deployed Pods
        # "cpu_usage_aggregated"            -> via metrics-server
        # "mem_usage_aggregated"            -> via metrics-server
        # "cpu_requests"                    -> via metrics-server/pod
        # "mem_requests"                    -> via metrics-server/pod
        # "cpu_limits"                      -> via metrics-server
        # "mem_limits"                      -> via metrics-server
        # "lstm_cpu_prediction_1_step"      -> via pod annotation
        # "lstm_cpu_prediction_5_step"      -> via pod annotation
        # "average_number of requests"      -> Prometheus metric: sum(rate(http_server_requests_seconds_count[5m]))

        self.min_pods = MIN_REPLICATION
        self.max_pods = MAX_REPLICATION
        self.num_apps = 2

        # Deployment Data
        self.deploymentList = get_redis_deployment_list(self.k8s, self.min_pods, self.max_pods)

        self.observation_space = self.get_observation_space()

        # Action and Observation Space
        # logging.info("[Init] Action Spaces: " + str(self.action_space))
        # logging.info("[Init] Observation Spaces: " + str(self.observation_space))

        # Info
        self.total_reward = None
        self.avg_pods = []
        self.avg_latency = []

        # episode over
        self.episode_over = False
        self.info = {}

        # Keywords for Reward calculation
        self.constraint_max_pod_replicas = False
        self.constraint_min_pod_replicas = False
        self.cost_weight = 0  # add here a value to consider cost in the reward function

        self.time_start = 0
        self.execution_time = 0
        self.episode_count = 0
        self.file_results = "results.csv"
        self.obs_csv = self.name + "_observation.csv"
        self.df = pd.read_csv("../../datasets/real/" + self.deploymentList[0].namespace + "/v1/"
                              + self.name + '_' + 'observation.csv')

    def step(self, action):
        if self.current_step == 1:
            if not self.k8s:
                self.simulation_update()

            self.time_start = time.time()

        # Get first action: deployment
        if action[ID_DEPLOYMENTS] == 0:  # master
            n = ID_MASTER  # master
        else:
            n = ID_SLAVE  # slave

        # Execute one time step within the environment
        self.take_action(action[ID_MOVES], n)

        # Wait a few seconds if on real k8s cluster
        if self.k8s:
            if action[ID_MOVES] != ACTION_DO_NOTHING \
                    and self.constraint_min_pod_replicas is False \
                    and self.constraint_max_pod_replicas is False:
                # logging.info('[Step {}] | Waiting {} seconds for enabling action ...'
                # .format(self.current_step, self.waiting_period))
                time.sleep(self.waiting_period)  # Wait a few seconds...

        # Update observation before reward calculation:
        if self.k8s:  # k8s cluster
            for d in self.deploymentList:
                d.update_obs_k8s()
        else:  # simulation
            self.simulation_update()

        # Get reward
        reward = self.get_reward

        # Update Infos
        self.total_reward += reward
        self.avg_pods.append(get_num_pods(self.deploymentList))
        self.avg_latency.append(self.deploymentList[0].latency)

        # Print Step and Total Reward
        # if self.current_step == MAX_STEPS:
        logging.info('[Step {}] | Action (Deployment): {} | Action (Move): {} | Reward: {} | Total Reward: {}'.format(
            self.current_step, DEPLOYMENTS[action[0]], MOVES[action[1]], reward, self.total_reward))

        ob = self.get_state()
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.save_obs_to_csv(self.obs_csv, np.array(ob), date, self.deploymentList[0].latency)

        self.info = dict(
            total_reward=self.total_reward,
        )

        # Update Reward Keywords
        self.constraint_max_pod_replicas = False
        self.constraint_min_pod_replicas = False

        if self.current_step == MAX_STEPS:
            self.episode_count += 1
            self.execution_time = time.time() - self.time_start
            save_to_csv(self.file_results, self.episode_count, mean(self.avg_pods), mean(self.avg_latency),
                        self.total_reward, self.execution_time)

        # return ob, reward, self.episode_over, self.info
        return np.array(ob), reward, self.episode_over, self.info

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.current_step = 0
        self.episode_over = False
        self.total_reward = 0
        self.avg_pods = []
        self.avg_latency = []

        self.constraint_max_pod_replicas = False
        self.constraint_min_pod_replicas = False

        # Deployment Data
        self.deploymentList = get_redis_deployment_list(self.k8s, self.min_pods, self.max_pods)

        return np.array(self.get_state())

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        return

    def take_action(self, action, id):
        self.current_step += 1

        # Stop if MAX_STEPS
        if self.current_step == MAX_STEPS:
            # logging.info('[Take Action] MAX STEPS achieved, ending ...')
            self.episode_over = True

        # ACTIONS
        if action == ACTION_DO_NOTHING:
            # logging.info("[Take Action] SELECTED ACTION: DO NOTHING ...")
            pass

        elif action == ACTION_ADD_1_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: ADD 1 Replica ...")
            self.deploymentList[id].deploy_pod_replicas(1, self)

        elif action == ACTION_ADD_2_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: ADD 2 Replicas ...")
            self.deploymentList[id].deploy_pod_replicas(2, self)

        elif action == ACTION_ADD_3_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: ADD 3 Replicas ...")
            self.deploymentList[id].deploy_pod_replicas(3, self)

        elif action == ACTION_ADD_4_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: ADD 4 Replicas ...")
            self.deploymentList[id].deploy_pod_replicas(4, self)

        elif action == ACTION_ADD_5_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: ADD 5 Replicas ...")
            self.deploymentList[id].deploy_pod_replicas(5, self)

        elif action == ACTION_ADD_6_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: ADD 6 Replicas ...")
            self.deploymentList[id].deploy_pod_replicas(6, self)

        elif action == ACTION_ADD_7_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: ADD 7 Replicas ...")
            self.deploymentList[id].deploy_pod_replicas(7, self)

        elif action == ACTION_TERMINATE_1_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: TERMINATE 1 Replica ...")
            self.deploymentList[id].terminate_pod_replicas(1, self)

        elif action == ACTION_TERMINATE_2_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: TERMINATE 2 Replicas ...")
            self.deploymentList[id].terminate_pod_replicas(2, self)

        elif action == ACTION_TERMINATE_3_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: TERMINATE 3 Replicas ...")
            self.deploymentList[id].terminate_pod_replicas(3, self)

        elif action == ACTION_TERMINATE_4_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: TERMINATE 4 Replicas ...")
            self.deploymentList[id].terminate_pod_replicas(4, self)

        elif action == ACTION_TERMINATE_5_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: TERMINATE 5 Replicas ...")
            self.deploymentList[id].terminate_pod_replicas(5, self)

        elif action == ACTION_TERMINATE_6_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: TERMINATE 6 Replicas ...")
            self.deploymentList[id].terminate_pod_replicas(6, self)

        elif action == ACTION_TERMINATE_7_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: TERMINATE 7 Replicas ...")
            self.deploymentList[id].terminate_pod_replicas(7, self)

        else:
            logging.info('[Take Action] Unrecognized Action: ' + str(action))

    @property
    def get_reward(self):
        """ Calculate Rewards """
        '''
        ob = self.get_state()
        logging.info('[Reward] | Master Pods: {} | CPU Usage: {} | MEM Usage: {} | Requests: {} | Response Time: {} | '
                     'Slave Pods: {} | CPU Usage: {} | MEM Usage: {} | Requests: {} | Response Time: {} |'.format(
            ob.__getitem__(0), ob.__getitem__(1), ob.__getitem__(2), ob.__getitem__(9), ob.__getitem__(10),
            ob.__getitem__(11), ob.__getitem__(12), ob.__getitem__(13), ob.__getitem__(20), ob.__getitem__(21), ))
        '''
        # Reward based on Keyword!
        if self.constraint_max_pod_replicas:
            if self.goal_reward == COST:
                return -1  # penalty
            elif self.goal_reward == LATENCY:
                return -250  # penalty

        if self.constraint_min_pod_replicas:
            if self.goal_reward == COST:
                return -1  # penalty
            elif self.goal_reward == LATENCY:
                return -250  # penalty

        # Reward Calculation
        reward = self.calculate_reward()
        # logging.info('[Get Reward] Reward: {} | Ob: {} |'.format(reward, ob))
        # logging.info('[Get Reward] Acc. Reward: {} |'.format(self.total_reward))

        return reward

    def get_state(self):
        # Observations: metrics - 3 Metrics!!
        # "number_pods"
        # "cpu"
        # "mem"
        # "requests"

        # Return ob
        ob = (
            self.deploymentList[0].num_pods, self.deploymentList[0].desired_replicas,
            self.deploymentList[0].cpu_usage, self.deploymentList[0].mem_usage,
            self.deploymentList[0].received_traffic, self.deploymentList[0].transmit_traffic,
            self.deploymentList[1].num_pods, self.deploymentList[1].desired_replicas,
            self.deploymentList[1].cpu_usage, self.deploymentList[1].mem_usage,
            self.deploymentList[1].received_traffic, self.deploymentList[1].transmit_traffic,
        )

        return ob

    def get_observation_space(self):
        return spaces.Box(
                low=np.array([
                    self.min_pods,  # Number of Pods  -- master metrics
                    self.min_pods,  # Desired Replicas
                    0,  # CPU Usage (in m)
                    0,  # MEM Usage (in MiB)
                    0,  # Average Number of received traffic
                    0,  # Average Number of transmit traffic
                    self.min_pods,  # Number of Pods -- slave metrics
                    self.min_pods,  # Number of Pods -- slave metrics
                    0,  # CPU Usage (in m)
                    0,  # MEM Usage (in MiB)
                    0,  # Average Number of received traffic
                    0,  # Average Number of transmit traffic
                ]), high=np.array([
                    self.max_pods,  # Number of Pods -- master metrics
                    self.max_pods,  # Desired Replicas
                    get_max_cpu(),  # CPU Usage (in m)
                    get_max_mem(),  # MEM Usage (in MiB)
                    get_max_traffic(),  # Average Number of received traffic
                    get_max_traffic(),  # Average Number of transmit traffic
                    self.max_pods,  # Number of Pods -- slave metrics
                    self.max_pods,  # Desired Replicas
                    get_max_cpu(),  # CPU Usage (in m)
                    get_max_mem(),  # MEM Usage (in MiB)
                    get_max_traffic(),  # Average Number of received traffic
                    get_max_traffic(),  # Average Number of transmit traffic
                ]),
                dtype=np.float32
            )

    # calculates the reward based on the objective
    def calculate_reward(self):
        reward = 0
        if self.goal_reward == COST:
            reward = get_cost_reward(self.deploymentList)
        elif self.goal_reward == LATENCY:
            reward = get_latency_reward_redis(ID_MASTER, self.deploymentList)

        return reward

    def simulation_update(self):
        if self.current_step == 1:
            # Get a random sample!
            sample = self.df.sample()
            # print(sample)

            self.deploymentList[0].num_pods = int(sample['redis-leader_num_pods'].values[0])
            self.deploymentList[0].num_previous_pods = int(sample['redis-leader_num_pods'].values[0])
            self.deploymentList[1].num_pods = int(sample['redis-follower_num_pods'].values[0])
            self.deploymentList[1].num_previous_pods = int(sample['redis-follower_num_pods'].values[0])

        else:
            leader_pods = self.deploymentList[0].num_pods
            leader_previous_pods = self.deploymentList[0].num_previous_pods
            follower_pods = self.deploymentList[1].num_pods
            follower_previous_pods = self.deploymentList[1].num_previous_pods

            diff_leader = leader_pods - leader_previous_pods
            diff_follower = follower_pods - follower_previous_pods

            self.df['diff-leader'] = self.df['redis-leader_num_pods'].diff()
            self.df['diff-follower'] = self.df['redis-follower_num_pods'].diff()

            data = self.df.loc[self.df['redis-leader_num_pods'] == leader_pods]

            data = data.loc[data['diff-leader'] == diff_leader]

            if data.size == 0:
                data = self.df.loc[self.df['redis-leader_num_pods'] == leader_pods]

            data = self.df.loc[self.df['redis-follower_num_pods'] == follower_pods]

            data = data.loc[data['diff-follower'] == diff_follower]

            if data.size == 0:
                data = self.df.loc[self.df['redis-follower_num_pods'] == follower_pods]

            sample = data.sample()
            # print(sample)

        self.deploymentList[0].cpu_usage = int(sample['redis-leader_cpu_usage'].values[0])
        self.deploymentList[0].mem_usage = int(sample['redis-leader_mem_usage'].values[0])
        self.deploymentList[0].received_traffic = int(sample['redis-leader_traffic_in'].values[0])
        self.deploymentList[0].transmit_traffic = int(sample['redis-leader_traffic_out'].values[0])
        self.deploymentList[0].latency = float(sample['redis-leader_latency'].values[0])

        self.deploymentList[1].cpu_usage = int(sample['redis-follower_cpu_usage'].values[0])
        self.deploymentList[1].mem_usage = int(sample['redis-follower_mem_usage'].values[0])
        self.deploymentList[1].received_traffic = int(sample['redis-follower_traffic_in'].values[0])
        self.deploymentList[1].transmit_traffic = int(sample['redis-follower_traffic_out'].values[0])
        self.deploymentList[1].latency = float(sample['redis-follower_latency'].values[0])

        for d in self.deploymentList:
            # Update Desired replicas
            d.update_replicas()
        return

    def save_obs_to_csv(self, obs_file, obs, date, latency):
        file = open(obs_file, 'a+', newline='')  # append
        # file = open(file_name, 'w', newline='') # new
        fields = []
        with file:
            fields.append('date')
            for d in self.deploymentList:
                fields.append(d.name + '_num_pods')
                fields.append(d.name + '_desired_replicas')
                fields.append(d.name + '_cpu_usage')
                fields.append(d.name + '_mem_usage')
                fields.append(d.name + '_traffic_in')
                fields.append(d.name + '_traffic_out')
                fields.append(d.name + '_latency')

            '''
            fields = ['date', 'redis-leader_num_pods', 'redis-leader_desired_replicas', 'redis-leader_cpu_usage', 'redis-leader_mem_usage',
                      'redis-leader_cpu_request', 'redis-leader_mem_request', 'redis-leader_cpu_limit', 'redis-leader_mem_limit',
                      'redis-leader_traffic_in', 'redis-leader_traffic_out',
                      'redis-follower_num_pods', 'redis-follower_desired_replicas', 'redis-follower_cpu_usage',
                      'redis-follower_mem_usage', 'redis-follower_cpu_request', 'redis-follower_mem_request', 'redis-follower_cpu_limit',
                      'redis-follower_mem_limit', 'redis-follower_traffic_in', 'redis-follower_traffic_out']
            '''
            writer = csv.DictWriter(file, fieldnames=fields)
            # writer.writeheader() # write header

            writer.writerow(
                {'date': date,
                 'redis-leader_num_pods': int("{}".format(obs[0])),
                 'redis-leader_desired_replicas': int("{}".format(obs[1])),
                 'redis-leader_cpu_usage': int("{}".format(obs[2])),
                 'redis-leader_mem_usage': int("{}".format(obs[3])),
                 'redis-leader_traffic_in': int("{}".format(obs[4])),
                 'redis-leader_traffic_out': int("{}".format(obs[5])),
                 'redis-leader_latency': float("{:.3f}".format(latency)),
                 'redis-follower_num_pods': int("{}".format(obs[6])),
                 'redis-follower_desired_replicas': int("{}".format(obs[7])),
                 'redis-follower_cpu_usage': int("{}".format(obs[8])),
                 'redis-follower_mem_usage': int("{}".format(obs[9])),
                 'redis-follower_traffic_in': int("{}".format(obs[10])),
                 'redis-follower_traffic_out': int("{}".format(obs[11])),
                 'redis-follower_latency': float("{:.3f}".format(latency))
                 }
            )
        return
