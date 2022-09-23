import csv


def save_obs_to_csv(file_name, timestamp, num_pods, desired_replicas, cpu_usage, mem_usage,
                    traffic_in, traffic_out, latency, lstm_1_step, lstm_5_step):
    file = open(file_name, 'a+', newline='')  # append
    # file = open(file_name, 'w', newline='') # new
    with file:
        fields = ['date', 'num_pods', 'cpu', 'mem', 'desired_replicas',
                  'traffic_in', 'traffic_out', 'latency', 'lstm_1_step', 'lstm_5_step']
        writer = csv.DictWriter(file, fieldnames=fields)
        # writer.writeheader() # write header
        writer.writerow(
            {'date': timestamp,
             'num_pods': int("{}".format(num_pods)),
             'cpu': int("{}".format(cpu_usage)),
             'mem': int("{}".format(mem_usage)),
             'desired_replicas': int("{}".format(desired_replicas)),
             'traffic_in': int("{}".format(traffic_in)),
             'traffic_out': int("{}".format(traffic_out)),
             'latency': float("{:.3f}".format(latency)),
             'lstm_1_step': int("{}".format(lstm_1_step)),
             'lstm_5_step': int("{}".format(lstm_5_step))}
        )


def save_to_csv(file_name, episode, avg_pods, avg_latency, reward, execution_time):
    file = open(file_name, 'a+', newline='')  # append
    # file = open(file_name, 'w', newline='')
    with file:
        fields = ['episode', 'avg_pods', 'avg_latency', 'reward', 'execution_time']
        writer = csv.DictWriter(file, fieldnames=fields)
        # writer.writeheader()
        writer.writerow(
            {'episode': episode,
             'avg_pods': float("{:.2f}".format(avg_pods)),
             'avg_latency': float("{:.4f}".format(avg_latency)),
             'reward': float("{:.2f}".format(reward)),
             'execution_time': float("{:.2f}".format(execution_time))}
        )


def get_cost_reward(deployment_list):
    reward = 0

    for d in deployment_list:
        num_pods = d.num_pods
        desired_replicas = d.desired_replicas
        if num_pods == desired_replicas:
            reward += 1

    return reward


def get_latency_reward_redis(ID_MASTER, deployment_list):
    # Calculate the redis latency based on the redis exporter
    reward = float(deployment_list[ID_MASTER].latency)
    if reward > 250.0:
        reward = -250  # highest penalty over 250 ms
    else:
        reward = -float(deployment_list[ID_MASTER].latency)  # negative reward

    return reward


def get_latency_reward_online_boutique(ID_recommendation, deployment_list):
    # Calculate the latency based on the GET / POST requests
    reward = float(deployment_list[ID_recommendation].latency)
    if reward > 3000.0:
        reward = -3000  # highest penalty over 3 s
    else:
        reward = -float(deployment_list[ID_recommendation].latency)  # negative reward

    return reward


def get_num_pods(deployment_list):
    n = 0
    for d in deployment_list:
        n += d.num_pods

    return n
