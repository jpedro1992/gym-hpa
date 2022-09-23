import csv
import datetime
from datetime import datetime
import logging
import time
from statistics import mean

import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding

# Number of Requests - Discrete Event
from gym_hpa.envs.deployment import get_max_cpu, get_max_mem, get_max_traffic, get_online_boutique_deployment_list
from gym_hpa.envs.util import save_to_csv, get_num_pods, get_cost_reward, \
    get_latency_reward_online_boutique

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
DEPLOYMENTS = ["recommendationservice", "productcatalogservice", "cartservice", "adservice",
               "paymentservice", "shippingservice", "currencyservice", "redis-cart",
               "checkoutservice", "frontend", "emailservice"]

# Action Moves
MOVES = ["None", "Add-1", "Add-2", "Add-3", "Add-4", "Add-5", "Add-6", "Add-7",
         "Stop-1", "Stop-2", "Stop-3", "Stop-4", "Stop-5", "Stop-6", "Stop-7"]

# IDs
ID_DEPLOYMENTS = 0
ID_MOVES = 1

ID_recommendation = 0
ID_product_catalog = 1
ID_cart_service = 2
ID_ad_service = 3
ID_payment_service = 4
ID_shipping_service = 5
ID_currency_service = 6
ID_redis_cart = 7
ID_checkout_service = 8
ID_frontend = 9
ID_email = 10

# Reward objectives
LATENCY = 'latency'
COST = 'cost'


class OnlineBoutique(gym.Env):
    """Horizontal Scaling for Online Boutique in Kubernetes - an OpenAI gym environment"""

    metadata = {'render.modes': ['human', 'ansi', 'array']}

    def __init__(self, k8s=False, goal_reward="cost", waiting_period=0.3):
        # Define action and observation space
        # They must be gym.spaces objects

        super(OnlineBoutique, self).__init__()

        self.k8s = k8s
        self.name = "online_boutique_gym"
        self.__version__ = "0.0.1"
        self.seed()
        self.goal_reward = goal_reward
        self.waiting_period = waiting_period  # seconds to wait after action

        logging.info("[Init] Env: {} | K8s: {} | Version {} |".format(self.name, self.k8s, self.__version__))

        # Current Step
        self.current_step = 0

        # Actions identified by integers 0-n -> 15 actions!
        self.num_actions = 15

        # Multi-Discrete
        # Deployment: Discrete 11
        # Action: Discrete 9 - None[0], Add-1[1], Add-2[2], Add-3[3], Add-4[4],
        #                      Stop-1[5], Stop-2[6], Stop-3[7], Stop-4[8]

        self.action_space = spaces.MultiDiscrete([11, self.num_actions])

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
        self.deploymentList = get_online_boutique_deployment_list(self.k8s, self.min_pods, self.max_pods)

        # Logging Deployment
        for d in self.deploymentList:
            d.print_deployment()

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

    # revision here!
    def step(self, action):
        if self.current_step == 1:
            if not self.k8s:
                self.simulation_update()

            self.time_start = time.time()

        # Get first action: deployment
        if action[ID_DEPLOYMENTS] == 0:  # recommendation
            n = ID_recommendation
        elif action[ID_DEPLOYMENTS] == 1:  # product catalog
            n = ID_product_catalog
        elif action[ID_DEPLOYMENTS] == 2:  # cart_service
            n = ID_cart_service
        elif action[ID_DEPLOYMENTS] == 3:  # ad_service
            n = ID_ad_service
        elif action[ID_DEPLOYMENTS] == 4:  # payment_service
            n = ID_payment_service
        elif action[ID_DEPLOYMENTS] == 5:  # shipping_service
            n = ID_shipping_service
        elif action[ID_DEPLOYMENTS] == 6:  # currency_service
            n = ID_currency_service
        elif action[ID_DEPLOYMENTS] == 7:  # redis_cart
            n = ID_redis_cart
        elif action[ID_DEPLOYMENTS] == 8:  # checkout_service
            n = ID_checkout_service
        elif action[ID_DEPLOYMENTS] == 9:  # frontend
            n = ID_frontend
        else:  # ==10 email
            n = ID_email

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
        else:
            self.simulation_update()

        # Get reward
        reward = self.get_reward
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

            # logging.info('Avg. latency : {} ', float("{:.3f}".format(mean(self.avg_latency))))
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
        self.deploymentList = get_online_boutique_deployment_list(self.k8s, self.min_pods, self.max_pods)

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
        # Reward based on Keyword!
        if self.constraint_max_pod_replicas:
            if self.goal_reward == COST:
                return -1  # penalty
            elif self.goal_reward == LATENCY:
                return -3000  # penalty

        if self.constraint_min_pod_replicas:
            if self.goal_reward == COST:
                return -1  # penalty
            elif self.goal_reward == LATENCY:
                return -3000  # penalty

        # Reward Calculation
        reward = self.calculate_reward()
        return reward

    def get_state(self):
        # Observations: metrics - 3 Metrics!!
        # "number_pods"
        # "cpu"
        # "mem"
        # "requests"

        # Return ob
        ob = (
                self.deploymentList[ID_recommendation].num_pods,
                self.deploymentList[ID_recommendation].desired_replicas,
                self.deploymentList[ID_recommendation].cpu_usage, self.deploymentList[ID_recommendation].mem_usage,
                self.deploymentList[ID_recommendation].received_traffic,
                self.deploymentList[ID_recommendation].transmit_traffic,
                self.deploymentList[ID_product_catalog].num_pods,
                self.deploymentList[ID_product_catalog].desired_replicas,
                self.deploymentList[ID_product_catalog].cpu_usage, self.deploymentList[ID_product_catalog].mem_usage,
                self.deploymentList[ID_product_catalog].received_traffic,
                self.deploymentList[ID_product_catalog].transmit_traffic,
                self.deploymentList[ID_cart_service].num_pods, self.deploymentList[ID_cart_service].desired_replicas,
                self.deploymentList[ID_cart_service].cpu_usage, self.deploymentList[ID_cart_service].mem_usage,
                self.deploymentList[ID_cart_service].received_traffic,
                self.deploymentList[ID_cart_service].transmit_traffic,
                self.deploymentList[ID_ad_service].num_pods, self.deploymentList[ID_ad_service].desired_replicas,
                self.deploymentList[ID_ad_service].cpu_usage, self.deploymentList[ID_ad_service].mem_usage,
                self.deploymentList[ID_ad_service].received_traffic,
                self.deploymentList[ID_ad_service].transmit_traffic,
                self.deploymentList[ID_payment_service].num_pods,
                self.deploymentList[ID_payment_service].desired_replicas,
                self.deploymentList[ID_payment_service].cpu_usage, self.deploymentList[ID_payment_service].mem_usage,
                self.deploymentList[ID_payment_service].received_traffic,
                self.deploymentList[ID_payment_service].transmit_traffic,
                self.deploymentList[ID_shipping_service].num_pods,
                self.deploymentList[ID_shipping_service].desired_replicas,
                self.deploymentList[ID_shipping_service].cpu_usage, self.deploymentList[ID_shipping_service].mem_usage,
                self.deploymentList[ID_shipping_service].received_traffic,
                self.deploymentList[ID_shipping_service].transmit_traffic,
                self.deploymentList[ID_currency_service].num_pods,
                self.deploymentList[ID_currency_service].desired_replicas,
                self.deploymentList[ID_currency_service].cpu_usage, self.deploymentList[ID_currency_service].mem_usage,
                self.deploymentList[ID_currency_service].received_traffic,
                self.deploymentList[ID_currency_service].transmit_traffic,
                self.deploymentList[ID_redis_cart].num_pods, self.deploymentList[ID_redis_cart].desired_replicas,
                self.deploymentList[ID_redis_cart].cpu_usage, self.deploymentList[ID_redis_cart].mem_usage,
                self.deploymentList[ID_redis_cart].received_traffic,
                self.deploymentList[ID_redis_cart].transmit_traffic,
                self.deploymentList[ID_checkout_service].num_pods,
                self.deploymentList[ID_checkout_service].desired_replicas,
                self.deploymentList[ID_checkout_service].cpu_usage, self.deploymentList[ID_checkout_service].mem_usage,
                self.deploymentList[ID_checkout_service].received_traffic,
                self.deploymentList[ID_checkout_service].transmit_traffic,
                self.deploymentList[ID_frontend].num_pods, self.deploymentList[ID_frontend].desired_replicas,
                self.deploymentList[ID_frontend].cpu_usage, self.deploymentList[ID_frontend].mem_usage,
                self.deploymentList[ID_frontend].received_traffic, self.deploymentList[ID_frontend].transmit_traffic,
                self.deploymentList[ID_email].num_pods, self.deploymentList[ID_email].desired_replicas,
                self.deploymentList[ID_email].cpu_usage, self.deploymentList[ID_email].mem_usage,
                self.deploymentList[ID_email].received_traffic, self.deploymentList[ID_email].transmit_traffic,
            )

        return ob

    def get_observation_space(self):
            return spaces.Box(
                low=np.array([
                    self.min_pods,  # Number of Pods  -- 1) recommendationservice
                    self.min_pods,  # Desired Replicas
                    0,  # CPU Usage (in m)
                    0,  # MEM Usage (in MiB)
                    0,  # Average Number of received traffic
                    0,  # Average Number of transmit traffic
                    self.min_pods,  # Number of Pods -- 2) productcatalogservice
                    self.min_pods,  # Desired Replicas
                    0,  # CPU Usage (in m)
                    0,  # MEM Usage (in MiB)
                    0,  # Average Number of received traffic
                    0,  # Average Number of transmit traffic
                    self.min_pods,  # Number of Pods -- 3) cartservice
                    self.min_pods,  # Desired Replicas
                    0,  # CPU Usage (in m)
                    0,  # MEM Usage (in MiB)
                    0,  # Average Number of received traffic
                    0,  # Average Number of transmit traffic
                    self.min_pods,  # Number of Pods -- 4) adservice
                    self.min_pods,  # Desired Replicas
                    0,  # CPU Usage (in m)
                    0,  # MEM Usage (in MiB)
                    0,  # Average Number of received traffic
                    0,  # Average Number of transmit traffic
                    self.min_pods,  # Number of Pods -- 5) paymentservice
                    self.min_pods,  # Desired Replicas
                    0,  # CPU Usage (in m)
                    0,  # MEM Usage (in MiB)
                    0,  # Average Number of received traffic
                    0,  # Average Number of transmit traffic
                    self.min_pods,  # Number of Pods -- 6) shippingservice
                    self.min_pods,  # Desired Replicas
                    0,  # CPU Usage (in m)
                    0,  # MEM Usage (in MiB)
                    0,  # Average Number of received traffic
                    0,  # Average Number of transmit traffic
                    self.min_pods,  # Number of Pods -- 7) currencyservice
                    self.min_pods,  # Desired Replicas
                    0,  # CPU Usage (in m)
                    0,  # MEM Usage (in MiB)
                    0,  # Average Number of received traffic
                    0,  # Average Number of transmit traffic
                    self.min_pods,  # Number of Pods -- 8) redis-cart
                    self.min_pods,  # Desired Replicas
                    0,  # CPU Usage (in m)
                    0,  # MEM Usage (in MiB)
                    0,  # Average Number of received traffic
                    0,  # Average Number of transmit traffic
                    self.min_pods,  # Number of Pods -- 9) checkoutservice
                    self.min_pods,  # Desired Replicas
                    0,  # CPU Usage (in m)
                    0,  # MEM Usage (in MiB)
                    0,  # Average Number of received traffic
                    0,  # Average Number of transmit traffic
                    self.min_pods,  # Number of Pods -- 10) frontend
                    self.min_pods,  # Desired Replicas
                    0,  # CPU Usage (in m)
                    0,  # MEM Usage (in MiB)
                    0,  # Average Number of received traffic
                    0,  # Average Number of transmit traffic
                    self.min_pods,  # Number of Pods -- 11) emailservice
                    self.min_pods,  # Desired Replicas
                    0,  # CPU Usage (in m)
                    0,  # MEM Usage (in MiB)
                    0,  # Average Number of received traffic
                    0,  # Average Number of transmit traffic
                ]), high=np.array([
                    self.max_pods,  # Number of Pods -- 1)
                    self.max_pods,  # Desired Replicas
                    get_max_cpu(),  # CPU Usage (in m)
                    get_max_mem(),  # MEM Usage (in MiB)
                    get_max_traffic(),  # Average Number of received traffic
                    get_max_traffic(),  # Average Number of transmit traffic
                    self.max_pods,  # Number of Pods -- 2)
                    self.max_pods,  # Desired Replicas
                    get_max_cpu(),  # CPU Usage (in m)
                    get_max_mem(),  # MEM Usage (in MiB)
                    get_max_traffic(),  # Average Number of received traffic
                    get_max_traffic(),  # Average Number of transmit traffic
                    self.max_pods,  # Number of Pods -- 3)
                    self.max_pods,  # Desired Replicas
                    get_max_cpu(),  # CPU Usage (in m)
                    get_max_mem(),  # MEM Usage (in MiB)
                    get_max_traffic(),  # Average Number of received traffic
                    get_max_traffic(),  # Average Number of transmit traffic
                    self.max_pods,  # Number of Pods -- 4)
                    self.max_pods,  # Desired Replicas
                    get_max_cpu(),  # CPU Usage (in m)
                    get_max_mem(),  # MEM Usage (in MiB)
                    get_max_traffic(),  # Average Number of received traffic
                    get_max_traffic(),  # Average Number of transmit traffic
                    self.max_pods,  # Number of Pods -- 5)
                    self.max_pods,  # Desired Replicas
                    get_max_cpu(),  # CPU Usage (in m)
                    get_max_mem(),  # MEM Usage (in MiB)
                    get_max_traffic(),  # Average Number of received traffic
                    get_max_traffic(),  # Average Number of transmit traffic
                    self.max_pods,  # Number of Pods -- 6)
                    self.max_pods,  # Desired Replicas
                    get_max_cpu(),  # CPU Usage (in m)
                    get_max_mem(),  # MEM Usage (in MiB)
                    get_max_traffic(),  # Average Number of received traffic
                    get_max_traffic(),  # Average Number of transmit traffic
                    self.max_pods,  # Number of Pods -- 7)
                    self.max_pods,  # Desired Replicas
                    get_max_cpu(),  # CPU Usage (in m)
                    get_max_mem(),  # MEM Usage (in MiB)
                    get_max_traffic(),  # Average Number of received traffic
                    get_max_traffic(),  # Average Number of transmit traffic
                    self.max_pods,  # Number of Pods -- 8)
                    self.max_pods,  # Desired Replicas
                    get_max_cpu(),  # CPU Usage (in m)
                    get_max_mem(),  # MEM Usage (in MiB)
                    get_max_traffic(),  # Average Number of received traffic
                    get_max_traffic(),  # Average Number of transmit traffic
                    self.max_pods,  # Number of Pods -- 9)
                    self.max_pods,  # Desired Replicas
                    get_max_cpu(),  # CPU Usage (in m)
                    get_max_mem(),  # MEM Usage (in MiB)
                    get_max_traffic(),  # Average Number of received traffic
                    get_max_traffic(),  # Average Number of transmit traffic
                    self.max_pods,  # Number of Pods -- 10)
                    self.max_pods,  # Desired Replicas
                    get_max_cpu(),  # CPU Usage (in m)
                    get_max_mem(),  # MEM Usage (in MiB)
                    get_max_traffic(),  # Average Number of received traffic
                    get_max_traffic(),  # Average Number of transmit traffic
                    self.max_pods,  # Number of Pods -- 11)
                    self.max_pods,  # Desired Replicas
                    get_max_cpu(),  # CPU Usage (in m)
                    get_max_mem(),  # MEM Usage (in MiB)
                    get_max_traffic(),  # Average Number of received traffic
                    get_max_traffic(),  # Average Number of transmit traffic
                ]),
                dtype=np.float32
            )

    # calculates the desired replica count based on a target metric utilization
    def calculate_reward(self):
        # Calculate Number of desired Replicas
        reward = 0
        if self.goal_reward == COST:
            reward = get_cost_reward(self.deploymentList)
        elif self.goal_reward == LATENCY:
            reward = get_latency_reward_online_boutique(ID_recommendation, self.deploymentList)

        return reward

    def simulation_update(self):
        if self.current_step == 1:
            # Get a random sample!
            sample = self.df.sample()
            # print(sample)

            for i in range(len(DEPLOYMENTS)):
                self.deploymentList[i].num_pods = int(sample[DEPLOYMENTS[i] + '_num_pods'].values[0])
                self.deploymentList[i].num_previous_pods = int(sample[DEPLOYMENTS[i] + '_num_pods'].values[0])

        else:
            pods = []
            previous_pods = []
            diff = []
            for i in range(len(DEPLOYMENTS)):
                pods.append(self.deploymentList[i].num_pods)
                previous_pods.append(self.deploymentList[i].num_previous_pods)
                aux = pods[i] - previous_pods[i]
                diff.append(aux)
                self.df['diff-' + DEPLOYMENTS[i]] = self.df[DEPLOYMENTS[i] + '_num_pods'].diff()

            # print(pods)
            # print(previous_pods)
            # print(diff)
            # print(self.df_aggr)

            data = 0
            for i in range(len(DEPLOYMENTS)):
                data = self.df.loc[self.df[DEPLOYMENTS[i] + '_num_pods'] == pods[i]]
                data = data.loc[data['diff-' + DEPLOYMENTS[i]] == diff[i]]
                if data.size == 0:
                    data = self.df.loc[self.df[DEPLOYMENTS[i] + '_num_pods'] == pods[i]]

            sample = data.sample()
            # print(sample)

        for i in range(len(DEPLOYMENTS)):
            self.deploymentList[i].cpu_usage = int(sample[DEPLOYMENTS[i] + '_cpu_usage'].values[0])
            self.deploymentList[i].mem_usage = int(sample[DEPLOYMENTS[i] + '_mem_usage'].values[0])
            self.deploymentList[i].received_traffic = int(sample[DEPLOYMENTS[i] + '_traffic_in'].values[0])
            self.deploymentList[i].transmit_traffic = int(sample[DEPLOYMENTS[i] + '_traffic_out'].values[0])
            self.deploymentList[i].latency = float("{:.3f}".format(sample[DEPLOYMENTS[i] + '_latency'].values[0]))

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

            # TO ALTER!
            # DEPLOYMENTS = ["recommendationservice", "productcatalogservice", "cartservice",
            # "adservice", "paymentservice", "shippingservice", "currencyservice",
            # "redis-cart", "checkoutservice", "frontend", "emailservice"]

            writer.writerow(
                {'date': date,
                 'recommendationservice_num_pods': int("{}".format(obs[0])),
                 'recommendationservice_desired_replicas': int("{}".format(obs[1])),
                 'recommendationservice_cpu_usage': int("{}".format(obs[2])),
                 'recommendationservice_mem_usage': int("{}".format(obs[3])),
                 'recommendationservice_traffic_in': int("{}".format(obs[4])),
                 'recommendationservice_traffic_out': int("{}".format(obs[5])),
                 'recommendationservice_latency': float("{:.3f}".format(latency)),

                 'productcatalogservice_num_pods': int("{}".format(obs[6])),
                 'productcatalogservice_desired_replicas': int("{}".format(obs[7])),
                 'productcatalogservice_cpu_usage': int("{}".format(obs[8])),
                 'productcatalogservice_mem_usage': int("{}".format(obs[9])),
                 'productcatalogservice_traffic_in': int("{}".format(obs[10])),
                 'productcatalogservice_traffic_out': int("{}".format(obs[11])),
                 'productcatalogservice_latency': float("{:.3f}".format(latency)),

                 'cartservice_num_pods': int("{}".format(obs[12])),
                 'cartservice_desired_replicas': int("{}".format(obs[13])),
                 'cartservice_cpu_usage': int("{}".format(obs[14])),
                 'cartservice_mem_usage': int("{}".format(obs[15])),
                 'cartservice_traffic_in': int("{}".format(obs[16])),
                 'cartservice_traffic_out': int("{}".format(obs[17])),
                 'cartservice_latency': float("{:.3f}".format(latency)),

                 'adservice_num_pods': int("{}".format(obs[18])),
                 'adservice_desired_replicas': int("{}".format(obs[19])),
                 'adservice_cpu_usage': int("{}".format(obs[20])),
                 'adservice_mem_usage': int("{}".format(obs[21])),
                 'adservice_traffic_in': int("{}".format(obs[22])),
                 'adservice_traffic_out': int("{}".format(obs[23])),
                 'adservice_latency': float("{:.3f}".format(latency)),

                 'paymentservice_num_pods': int("{}".format(obs[24])),
                 'paymentservice_desired_replicas': int("{}".format(obs[25])),
                 'paymentservice_cpu_usage': int("{}".format(obs[26])),
                 'paymentservice_mem_usage': int("{}".format(obs[27])),
                 'paymentservice_traffic_in': int("{}".format(obs[28])),
                 'paymentservice_traffic_out': int("{}".format(obs[29])),
                 'paymentservice_latency': float("{:.3f}".format(latency)),

                 'shippingservice_num_pods': int("{}".format(obs[30])),
                 'shippingservice_desired_replicas': int("{}".format(obs[31])),
                 'shippingservice_cpu_usage': int("{}".format(obs[32])),
                 'shippingservice_mem_usage': int("{}".format(obs[33])),
                 'shippingservice_traffic_in': int("{}".format(obs[34])),
                 'shippingservice_traffic_out': int("{}".format(obs[35])),
                 'shippingservice_latency': float("{:.3f}".format(latency)),

                 'currencyservice_num_pods': int("{}".format(obs[36])),
                 'currencyservice_desired_replicas': int("{}".format(obs[37])),
                 'currencyservice_cpu_usage': int("{}".format(obs[38])),
                 'currencyservice_mem_usage': int("{}".format(obs[39])),
                 'currencyservice_traffic_in': int("{}".format(obs[40])),
                 'currencyservice_traffic_out': int("{}".format(obs[41])),
                 'currencyservice_latency': float("{:.3f}".format(latency)),

                 'redis-cart_num_pods': int("{}".format(obs[42])),
                 'redis-cart_desired_replicas': int("{}".format(obs[43])),
                 'redis-cart_cpu_usage': int("{}".format(obs[44])),
                 'redis-cart_mem_usage': int("{}".format(obs[45])),
                 'redis-cart_traffic_in': int("{}".format(obs[46])),
                 'redis-cart_traffic_out': int("{}".format(obs[47])),
                 'redis-cart_latency': float("{:.3f}".format(latency)),

                 'checkoutservice_num_pods': int("{}".format(obs[48])),
                 'checkoutservice_desired_replicas': int("{}".format(obs[49])),
                 'checkoutservice_cpu_usage': int("{}".format(obs[50])),
                 'checkoutservice_mem_usage': int("{}".format(obs[51])),
                 'checkoutservice_traffic_in': int("{}".format(obs[52])),
                 'checkoutservice_traffic_out': int("{}".format(obs[53])),
                 'checkoutservice_latency': float("{:.3f}".format(latency)),

                 'frontend_num_pods': int("{}".format(obs[54])),
                 'frontend_desired_replicas': int("{}".format(obs[55])),
                 'frontend_cpu_usage': int("{}".format(obs[56])),
                 'frontend_mem_usage': int("{}".format(obs[57])),
                 'frontend_traffic_in': int("{}".format(obs[58])),
                 'frontend_traffic_out': int("{}".format(obs[59])),
                 'frontend_latency': float("{:.3f}".format(latency)),

                 'emailservice_num_pods': int("{}".format(obs[60])),
                 'emailservice_desired_replicas': int("{}".format(obs[61])),
                 'emailservice_cpu_usage': int("{}".format(obs[62])),
                 'emailservice_mem_usage': int("{}".format(obs[63])),
                 'emailservice_traffic_in': int("{}".format(obs[64])),
                 'emailservice_traffic_out': int("{}".format(obs[65])),
                 'emailservice_latency': float("{:.3f}".format(latency))
                 }
            )
        return
