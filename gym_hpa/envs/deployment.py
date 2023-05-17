import logging
import math
import random
import time
import requests
from kubernetes import client

# Constants
MAX_CPU = 10000  # cpu in m
MAX_MEM = 10000  # memory in MiB
MAX_TRAFFIC = 20000  # MAX Number of requests (in Kbit/s)

CPU_WEIGHT = 0.7
MEM_WEIGHT = 0.3

# port-forward in k8s cluster
PROMETHEUS_URL = 'http://localhost:9090/'

# Endpoint of your Kube cluster: kube proxy enabled
HOST = "http://localhost:8080"

# TODO: Add the TOKEN from your cluster!
TOKEN = ""


def get_redis_deployment_list(k8s, min, max):
    deployment_list = [
        DeploymentStatus(k8s, "redis-leader", "redis", "leader", "docker.io/redis:6.0.5",
                         max, min, 250, 500, 250, 500),
        DeploymentStatus(k8s, "redis-follower", "redis", "follower",
                         "gcr.io/google_samples/gb-redis-follower:v2",
                         max, min, 250, 500, 250, 500)]
    return deployment_list


def get_online_boutique_deployment_list(k8s, min, max):
    deployment_list = [
        # 1
        DeploymentStatus(k8s, "recommendationservice", "onlineboutique", "recommendationservice",
                         "quay.io/signalfuse/microservices-demo-recommendationservice:433c23881a",
                         max, min, 100, 200, 220, 450),
        # 2
        DeploymentStatus(k8s, "productcatalogservice", "onlineboutique", "productcatalogservice",
                         "quay.io/signalfuse/microservices-demo-productcatalogservice:433c23881a",
                         max, min, 100, 200, 64, 128),
        # 3
        DeploymentStatus(k8s, "cartservice", "onlineboutique", "cartservice",
                         "quay.io/signalfuse/microservices-demo-cartservice:433c23881a",
                         max, min, 200, 300, 64, 128),
        # 4
        DeploymentStatus(k8s, "adservice", "onlineboutique", "adservice",
                         "quay.io/signalfuse/microservices-demo-adservice:433c23881a",
                         max, min, 200, 300, 180, 300),
        # 5
        DeploymentStatus(k8s, "paymentservice", "onlineboutique", "paymentservice",
                         "quay.io/signalfuse/microservices-demo-paymentservice:433c23881a",
                         max, min, 100, 200, 64, 128),
        # 6
        DeploymentStatus(k8s, "shippingservice", "onlineboutique", "shippingservice",
                         "quay.io/signalfuse/microservices-demo-shippingservice:433c23881a",
                         max, min, 100, 200, 64, 128),
        # 7
        DeploymentStatus(k8s, "currencyservice", "onlineboutique", "currencyservice",
                         "quay.io/signalfuse/microservices-demo-currencyservice:433c23881a",
                         max, min, 100, 200, 64, 128),
        # 8
        DeploymentStatus(k8s, "redis-cart", "onlineboutique", "redis-cart",
                         "redis:alpine",
                         max, min, 70, 125, 200, 256),
        # 9
        DeploymentStatus(k8s, "checkoutservice", "onlineboutique", "checkoutservice",
                         "quay.io/signalfuse/microservices-demo-checkoutservice:433c23881a",
                         max, min, 100, 200, 64, 128),
        # 10
        DeploymentStatus(k8s, "frontend", "onlineboutique", "frontend",
                         "quay.io/signalfuse/microservices-demo-frontend:433c23881a",
                         max, min, 100, 200, 64, 128),
        # 11
        DeploymentStatus(k8s, "emailservice", "onlineboutique", "emailservice",
                         "quay.io/signalfuse/microservices-demo-frontend:433c23881a",
                         max, min, 100, 200, 64, 128),
    ]
    return deployment_list


def get_max_cpu():
    return MAX_CPU


def get_max_mem():
    return MAX_MEM


def get_max_traffic():
    return MAX_TRAFFIC

def convert_to_milli_cpu(value):
    new_value = int(value[:-1])
    if value[-1] == "n":
        new_value = int(value[:-1])
        new_value = int(new_value / 1000000)

    return new_value


def change_usage(min, max, max_threshold):
    if max > max_threshold:
        max = max_threshold

    if min < 0:
        min = 0

    return random.randint(min, max)


def convert_to_mega_memory(value):
    last_two = value[-2:]
    new_value = 0

    if last_two == "Ki":
        size = len(value)
        # Slice string to remove last 2 characters
        new_value = int(value[:size - 2])
        new_value = int(new_value / 1000)

    return new_value


class DeploymentStatus:  # Deployment Status (Workload)
    def __init__(self, k8s, name, namespace, container_name, container_image, max_pods, min_pods,
                 cpu_request, cpu_limit, mem_request, mem_limit, threshold=0.75):
        self.name = name
        # namespace
        self.namespace = namespace
        # container_name
        self.container_name = container_name
        # container image
        self.container_image = container_image

        # CPU & MEM threshold
        self.threshold = threshold
        # CPU weight for replica calculation
        self.cpu_weight = CPU_WEIGHT
        # MEM weight for replica calculation
        self.mem_weight = MEM_WEIGHT

        # Pod Names
        self.pod_names = ["pod-1"]
        # MAX Number of Pods
        self.max_pods = max_pods
        # MIN Number of Pods
        self.min_pods = min_pods
        # Number of Pods
        self.num_pods = 1  # Initialize as 1
        # Number of Pods in previous step
        self.num_previous_pods = 1  # Initialize as 1
        # Number of desired replicas
        self.desired_replicas = 1

        # CPU request (in m)
        self.cpu_request = cpu_request
        # CPU limit (in m)
        self.cpu_limit = cpu_limit

        # MEM request (in MiB)
        self.mem_request = mem_request
        # MEM limit (in MiB)
        self.mem_limit = mem_limit

        # CPU Target (in m)
        self.cpu_target = int(self.threshold * self.cpu_request)

        # MEM Target (in MiB)
        self.mem_target = int(self.threshold * self.mem_request)

        self.MAX_CPU = MAX_CPU  # cpu in m
        self.MAX_MEM = MAX_MEM  # memory in MiB
        self.MAX_TRAFFIC = MAX_TRAFFIC  # MAX Number of requests

        # Get dataset
        # self.version = 'v1'
        # self.df = pd.read_csv(
        #     "../../datasets/real/" + self.namespace + "/" + self.version +
        #     "/" + self.namespace + '_' + self.name + '.csv')

        # CPU Usage Aggregated (in m)
        self.cpu_usage = random.randint(1, get_max_cpu())  # sample['cpu'].values[0]

        # MEM Usage Aggregated (in MiB)
        self.mem_usage = random.randint(1, get_max_mem())  # sample['mem'].values[0]

        # Current Requests
        self.received_traffic = random.randint(1, get_max_traffic())  # sample['traffic_in'].values[0]
        self.transmit_traffic = random.randint(1, get_max_traffic())  # sample['traffic_out'].values[0]

        # Throughput PING INLINE
        # self.ping = 0

        # K8s enabled?
        self.k8s = k8s

        # csv file
        self.csv = self.namespace + '_' + self.name + '.csv'

        # time between API calls if failure happens
        self.sleep = 0.2

        # App. Latency
        self.latency = 0

        if self.k8s:  # Real env: consider a k8s cluster
            logging.info("[Deployment] Consider a real k8s cluster ... ")
            # out of cluster!
            # config.load_kube_config()

            # In cluster config!
            # config.load_incluster_config()

            # token for VWall cluster
            self.token = TOKEN

            # Create a configuration object
            self.config = client.Configuration()
            self.config.verify_ssl = False
            self.config.api_key = {"authorization": "Bearer " + self.token}

            # Specify the endpoint of your Kube cluster: kube proxy enabled
            self.config.host = HOST

            # Create a ApiClient with our config
            self.client = client.ApiClient(self.config)

            # v1 api
            self.v1 = client.CoreV1Api(self.client)
            # apps v1 api
            self.apps_v1 = client.AppsV1Api(self.client)

            # metrics api
            # self.metrics_api = client.CustomObjectsApi(self.client)
            # Get deployment object
            self.deployment_object = self.apps_v1.read_namespaced_deployment(name=self.name, namespace=self.namespace)

            # Update number of Pods
            self.num_pods = self.deployment_object.spec.replicas
            self.num_previous_pods = self.deployment_object.spec.replicas

            # update obs
            self.update_obs_k8s()

        # else: # Simulation Environment
        # Update Desired replicas
        # self.update_replicas()

    def update_obs_k8s(self):
        self.pod_names = []
        pods = self.v1.list_namespaced_pod(namespace=self.namespace)
        for p in pods.items:
            if p.metadata.labels['app'] == self.name:
                self.pod_names.append(p.metadata.name)

        self.cpu_usage = 0
        self.mem_usage = 0
        self.received_traffic = 0
        self.transmit_traffic = 0

        # Previous number of Pods
        self.num_previous_pods = self.deployment_object.spec.replicas

        # Get deployment object
        self.deployment_object = self.apps_v1.read_namespaced_deployment(name=self.name, namespace=self.namespace)

        # Update number of Pods
        self.num_pods = self.deployment_object.spec.replicas

        # logging.info("[Update obs] Current Pods: " + str(self.num_pods))

        # Get received / transmit traffic
        for p in self.pod_names:
            query_cpu = 'sum(irate(container_cpu_usage_seconds_total{namespace=' \
                        '"' + self.namespace + '", pod="' + p + '"}[5m])) by (pod)'

            query_mem = 'sum(irate(container_memory_working_set_bytes{namespace=' \
                        '"' + self.namespace + '", pod="' + p + '"}[5m])) by (pod)'

            query_received = 'sum(irate(container_network_receive_bytes_total{namespace=' \
                             '"' + self.namespace + '", pod="' + p + '"}[5m])) by (pod)'
            query_transmit = 'sum(irate(container_network_transmit_bytes_total{namespace="' \
                             + self.namespace + '", pod="' + p + '"}[5m])) by (pod)'

            # -------------- CPU ----------------
            results_cpu = self.fetch_prom(query_cpu)
            if results_cpu:
                cpu = int(float(results_cpu[0]['value'][1]) * 1000)  # saved as m
                self.cpu_usage += cpu

            # -------------- MEM ----------------
            results_mem = self.fetch_prom(query_mem)
            if results_mem:
                mem = int(float(results_mem[0]['value'][1]) / 1000000)  # saved as Mi
                self.mem_usage += mem

            # -------------- Received Traffic  ----------------
            results_received = self.fetch_prom(query_received)
            if results_received:
                rec = int(float(results_received[0]['value'][1]))
                rec = int(rec / 1000)  # saved as KBit/s
                self.received_traffic += rec

            # -------------- Transmit Traffic  ----------------
            results_transmit = self.fetch_prom(query_transmit)
            if results_transmit:
                trans = int(float(results_transmit[0]['value'][1]))
                trans = int(trans / 1000)  # saved as KBit/s
                self.transmit_traffic += trans

            if self.name == 'redis-leader':
                query_duration = 'sum(irate(redis_commands_duration_seconds_total[5m]))'
                query_processed = 'sum(irate(redis_commands_processed_total[5m]))'
                redis_duration = 0
                redis_processed = 0

                results_duration = self.fetch_prom(query_duration)
                if results_duration:
                    dur = float(results_duration[0]['value'][1])
                    dur = dur * 1000  # saved as ms
                    redis_duration = float("{:.3f}".format(dur))
                # logging.info("[Deployment] redis duration (in ms): " + str(self.redis_duration))

                results_processed = self.fetch_prom(query_processed)
                if results_processed:
                    proc = float(results_processed[0]['value'][1])
                    redis_processed = float("{:.3f}".format(proc))
                # logging.info("[Deployment] redis processed: " + str(self.redis_processed))

                if redis_processed != 0:
                    redis_latency = redis_duration / redis_processed
                else:
                    redis_latency = redis_duration

                self.latency = float("{:.3f}".format(redis_latency))
                # logging.info("[Deployment] redis latency (in ms): " + str(self.redis_latency))

            if self.name == 'recommendationservice':
                query_get_cart = 'locust_requests_avg_response_time{method="GET", name="/cart"}'
                get_cart = 0

                results_get_cart = self.fetch_prom(query_get_cart)
                if results_get_cart:
                    dur = float(results_get_cart[0]['value'][1])
                    get_cart = float("{:.3f}".format(dur))
                    # logging.info("[Deployment] get cart (in ms): " + str(get_cart))

                # self.latency = float("{:.3f}".format((get_cart + post_cart + post_cart_checkout) / 3))
                self.latency = float("{:.3f}".format(get_cart))
                # logging.info("[Deployment] Online Bout. Latency (in ms): " + str(self.latency))

        # Update Desired replicas
        self.update_replicas()

        return

    def update_replicas(self):
        cpu_target_usage = self.num_pods * self.cpu_target
        mem_target_usage = self.num_pods * self.mem_target

        desired_replicas_cpu = math.ceil(self.num_pods * (self.cpu_usage / cpu_target_usage))
        desired_replicas_mem = math.ceil(self.num_pods * (self.mem_usage / mem_target_usage))

        # CPU and Memory
        # CPU = 0.7
        # MEM = 0.3
        self.desired_replicas = math.ceil((self.cpu_weight * desired_replicas_cpu)
                                          + (self.mem_weight * desired_replicas_mem))

        # min = 1
        if self.desired_replicas == 0:
            self.desired_replicas = 1

        # max = should be equal to the maximum
        if self.desired_replicas > self.max_pods:
            self.desired_replicas = self.max_pods

        return

    def fetch_prom(self, query):
        try:
            response = requests.get(PROMETHEUS_URL + '/api/v1/query',
                                    params={'query': query})

        except requests.exceptions.RequestException as e:
            print(e)
            print("Retrying in {}...".format(self.sleep))
            time.sleep(self.sleep)
            return self.fetch_prom(query)

        if response.json()['status'] != "success":
            print("Error processing the request: " + response.json()['status'])
            print("The Error is: " + response.json()['error'])
            print("Retrying in {}s...".format(self.sleep))
            time.sleep(self.sleep)
            return self.fetch_prom(query)

        result = response.json()['data']['result']
        return result

    def print_deployment(self):
        logging.info("[Deployment] Name: " + str(self.name))
        logging.info("[Deployment] Namespace: " + str(self.namespace))
        logging.info("[Deployment] Number of pods: " + str(self.num_pods))
        logging.info("[Deployment] Desired Replicas: " + str(self.desired_replicas))
        logging.info("[Deployment] Pod Names: " + str(self.pod_names))
        logging.info("[Deployment] MAX Pods: " + str(self.max_pods))
        logging.info("[Deployment] MIN Pods: " + str(self.min_pods))
        logging.info("[Deployment] CPU Usage (in m): " + str(self.cpu_usage))
        logging.info("[Deployment] MEM Usage (in Mi): " + str(self.mem_usage))
        logging.info("[Deployment] Received traffic (in Kbit/s): " + str(self.received_traffic))
        logging.info("[Deployment] Transmit traffic (in Kbit/s): " + str(self.transmit_traffic))
        logging.info("[Deployment] latency (in ms): " + str(self.latency))

    def update_deployment(self, new_replicas):
        # Get deployment object
        self.deployment_object = self.apps_v1.read_namespaced_deployment(name=self.name, namespace=self.namespace)
        # logging.info(self.deployment_object)

        # Update previous number of pods
        self.num_previous_pods = self.deployment_object.spec.replicas

        # Update replicas
        self.deployment_object.spec.replicas = new_replicas

        # try to patch the deployment
        self.patch_deployment(new_replicas)

    def patch_deployment(self, new_replicas):
        try:
            self.apps_v1.patch_namespaced_deployment(
                name=self.name, namespace=self.namespace, body=self.deployment_object
            )
        except Exception as e:
            print(e)
            print("Retrying in {}s...".format(self.sleep))
            time.sleep(self.sleep)
            return self.update_deployment(new_replicas)

    def deploy_pod_replicas(self, n, env):
        # Deploy pods if possible
        replicas = self.num_pods + n

        # logging.info("Deployment name: " + str(self.name))
        # logging.info("Current replicas: " + str(self.num_pods))
        # logging.info("New replicas: " + str(replicas))

        if replicas <= self.max_pods:
            # logging.info("[Take Action] Add {} Replicas".format(str(n)))
            if self.k8s:  # patch deployment on k8s cluster
                self.update_deployment(replicas)
            else:
                self.num_previous_pods = self.num_pods
                self.num_pods = replicas
            return
        else:
            # logging.info("Constraint: MAX Pod Replicas! Desired replicas: " + str(replicas))
            env.constraint_max_pod_replicas = True

    def terminate_pod_replicas(self, n, env):
        # Terminate pods if possible
        replicas = self.num_pods - n

        # logging.info("Deployment name: " + str(self.name))
        # logging.info("Current replicas: " + str(self.num_pods))
        # logging.info("New replicas: " + str(replicas))

        if replicas >= self.min_pods:
            # logging.info("[Take Action] Terminate {} Replicas".format(str(n)))
            if self.k8s:  # patch deployment on k8s cluster
                self.update_deployment(replicas)
            else:
                self.num_previous_pods = self.num_pods
                self.num_pods = replicas
            return
        else:
            # logging.info("Constraint: MIN Pod Replicas! Desired replicas: " + str(replicas))
            env.constraint_min_pod_replicas = True