import time
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
sys.path.append(r"D:\\BaiduNetdiskDownload\\2025_C4智能导航赛道_赛事资源\\usvlib4ros_origin\\usvlib4ros_origin")
from usvlib4ros.dqn.new_env import Env
from usvlib4ros.dqn.dqn_ros_service import DQN_ROS_Service
from usvlib4ros.usvRosUtil import USVRosbridgeClient

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
N_ACTIONS = 5
MEMORY_CAPACITY = 2000
BATCH_SIZE = 128
LR = 0.001
GAMMA = 0.9
TARGET_REPLACE_ITER = 10
MAX_EPOCH = 4000
EVAL_INTERVAL = 100

# Visualization
reward_list = []
eval_rewards = []
eval_epochs = []

def plot_rewards():
    plt.figure(figsize=(10, 5))
    plt.plot(reward_list, label="Train Reward")
    plt.plot(eval_epochs, eval_rewards, 'ro-', label="Eval Reward")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.title("DQN Training Performance")
    plt.legend()
    plt.grid()
    plt.savefig("reward_plot.png")
    plt.close()

class Net(nn.Module):
    def __init__(self, n_states):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, N_ACTIONS)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class DQN:
    def __init__(self, n_states):
        self.n_states = n_states
        self.eval_net = Net(n_states).to(device)
        self.target_net = Net(n_states).to(device)
        self.memory = np.zeros((MEMORY_CAPACITY, n_states * 2 + 2))
        self.memory_counter = 0
        self.learn_step_counter = 0
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.double_dqn = True

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(device)
        if np.random.uniform() > self.epsilon:
            action = torch.max(self.eval_net(x), 1)[1].cpu().item()
        else:
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        indices = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[indices, :]
        b_s = torch.FloatTensor(b_memory[:, :self.n_states]).to(device)
        b_a = torch.LongTensor(b_memory[:, self.n_states:self.n_states + 1].astype(int)).to(device)
        b_r = torch.FloatTensor(b_memory[:, self.n_states + 1:self.n_states + 2]).to(device)
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_states:]).to(device)

        q_eval = self.eval_net(b_s).gather(1, b_a)
        if self.double_dqn:
            next_actions = self.eval_net(b_s_).max(1)[1].view(-1, 1)
            q_next = self.target_net(b_s_).gather(1, next_actions).detach()
        else:
            q_next = self.target_net(b_s_).detach().max(1)[0].view(BATCH_SIZE, 1)

        q_target = b_r + GAMMA * q_next

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == "__main__":
    try:
        USVRosbridgeClient.Host = "47.94.163.11"
        USVRosbridgeClient.Port = 9090
        env = Env(action_size=N_ACTIONS, rosHost=USVRosbridgeClient.Host, deviceId="2ec0bf09846e84bf7c5680953464185a930384b5",
                  include_angle_and_distance_features=True)

        dqn = None
        N_STATES = None

        print("[INFO] DQN training script started")
        print("Waiting for train button trigger ...")

        while env.isTrainActionTrigger() == 0:
            time.sleep(1)

        for e in range(MAX_EPOCH):
            if env.isTrainActionTrigger() == 0:
                break

            s = env.reset()
            if dqn is None:
                N_STATES = len(s)
                dqn = DQN(N_STATES)

            max_distance = env.goal_distance
            episode_reward = 0

            for t in range(3000):
                if env.isTrainActionTrigger() == 0:
                    break

                a = dqn.choose_action(s)
                s_, r, done, max_distance = env.step(s, a, max_distance)
                dqn.store_transition(s, a, r, s_)
                episode_reward += r
                s = s_

                if dqn.memory_counter > MEMORY_CAPACITY:
                    dqn.learn()

                if done or t >= 2500:
                    break

            reward_list.append(episode_reward)
            DQN_ROS_Service.updateTrainStatus(e, t, int(episode_reward), 0, MAX_EPOCH, 2)

            if dqn.epsilon > dqn.epsilon_min:
                dqn.epsilon *= 0.995
                dqn.epsilon = max(dqn.epsilon, dqn.epsilon_min)

            if e % EVAL_INTERVAL == 0:
                eval_reward = 0
                for _ in range(5):
                    s = env.reset()
                    max_distance = env.goal_distance
                    for _ in range(1000):
                        a = dqn.choose_action(s)
                        s_, r, done, max_distance = env.step(s, a, max_distance)
                        eval_reward += r
                        s = s_
                        if done:
                            break
                avg_reward = eval_reward / 5
                eval_epochs.append(e)
                eval_rewards.append(avg_reward)

        plot_rewards()

    except Exception as e:
        from usvlib4ros.usvRosUtil import LogUtil
        LogUtil.error(e)
