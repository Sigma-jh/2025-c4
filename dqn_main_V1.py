import time
import sys
sys.path.append(r"D:\\BaiduNetdiskDownload\\2025_C4智能导航赛道_赛事资源\\usvlib4ros_origin\\usvlib4ros_origin")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#import matplotlib.pyplot as plt

from usvlib4ros.dqn.dqn_env import Env
from usvlib4ros.dqn.dqn_ros_service import DQN_ROS_Service
from usvlib4ros.usvRosUtil import LogUtil,USVRosbridgeClient

#device 如果有cuda更好，看能否加快计算速度
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

"""
已知，batch_size、gamma、episilon 还有更新频率都对模型的训练有很大的影响 如果效果有问题需要考虑修改
测试用模型与本模型的区别：
船的物理模型，控制方面可能会有误差
数据获取方面，所用的点云数据是否相似，数据处理上是否类似
state和actions要酌情减少
"""

N_ACTIONS = 5   # 油门, 推杆 同时也是网络输出的通道数,这个要改一下    5 and 28  25 and 128
N_STATES = 184  # 状态个数 (4个) 365       想办法给state降维184
MEMORY_CAPACITY = 2000                          # 记忆库容量
BATCH_SIZE = 128                               # 样本数量 原来是128
LR = 0.001                                       # 学习率
GAMMA = 0.9                                     # reward discount，折扣因子，防止学习过快，关键参数
TARGET_REPLACE_ITER = 10
MAX_Epoch = 4000

reward_list = []                                #评估用 数据存储和绘图

# 定义Net类 (定义网络)
class Net(nn.Module):

    def __init__(self):                                                         # 定义Net的一系列属性 网络的架构是两个全连接层带激活函数 输出通道是action
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()                                             # 等价与nn.Module.__init__()

        self.fc1 = nn.Linear(N_STATES, 128)                                      # 设置第一个全连接层(输入层到隐藏层): 状态数个神经元到50个神经元
        self.fc1.weight.data.normal_(0, 0.1)                                    # 权重初始化 (均值为0，方差为0.1的正态分布)
        self.fc2 =nn.Linear(128,128)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(128, N_ACTIONS)                                     # 设置第二个全连接层(隐藏层到输出层): 50个神经元到动作数个神经元
        self.out.weight.data.normal_(0, 0.1)                                    # 权重初始化 (均值为0，方差为0.1的正态分布)

    def forward(self, x):                                                       # 定义forward函数 (x为状态)
        x = F.relu(self.fc1(x))                                                 # 连接输入层到隐藏层，且使用激励函数ReLU来处理经过隐藏层后的值
        x = F.relu(self.fc2(x))
        x = F.dropout(self.fc2(x))
        actions_value = self.out(x)                                             # 连接隐藏层到输出层，获得最终的输出值 (即动作值)
        return actions_value
    pass

# class Net(nn.Module):     # 使用2dconv要就行数据整形，现在的情况state并不是真实数据而是距离信息
#     def __init__(self, N_STATES, N_ACTIONS):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=N_STATES, out_channels=32, kernel_size=8, stride=4)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
#
#         self.fc1 = nn.Linear(in_features=7*7*64, out_features=512)
#         self.fc2 = nn.Linear(in_features=512, out_features=N_ACTIONS)
#
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.relu(self.conv3(x))
#         x = x.view(x.size(0), -1)
#         x = self.relu(self.fc1(x))
#         actions_value = self.fc2(x)
#         return actions_value
#     pass

class DQN:

    def __init__(self,loadModel = False):
        self.eval_net = Net()       #评估网络
        self.target_net = Net()     # 目标网络
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, 370))  # 初始化记忆库，一行代表一个transition(62)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)  # 使用Adam优化器 (输入为评估网络的参数和学习率)
        self.loss_func = nn.MSELoss()  # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)
        self.start_epoch = 0
        self.epsilon = 1              #贪心值，有1-ep的概率探索路径 这个影响很大 原来数据为1.0，贪心策略会自动更新
        self.epsilon_min = 0.01       #若加载模型，贪心策略设置为0，不在探索路径 之前是0.01，一直学习到0.99
        self.load_ep = 780            #可以理解为超参数
        self.loss = 0
        self.q_eval = 0
        self.q_target = 0
        self.learn_step_counter = 0
        if loadModel:
            self.epsilon = 0        #贪心为0，都给出最优策略
            self.start_epoch = self.load_ep
            checkpoint = torch.load("a.pt")
            print(checkpoint.keys())
            print(checkpoint['epoch'])
            print(checkpoint)

            self.target_net.load_state_dict(checkpoint['target_net'])
            self.eval_net.load_state_dict(checkpoint['eval_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = checkpoint['epoch'] + 1
            print("loadmodel")

            pass

    def choose_action(self, x):  # 定义动作选择函数 (x为状态)
        print("[DEBUG] choose_action() called")
        x = torch.unsqueeze(torch.FloatTensor(x), 0)  # 将x转换成32-bit floating point形式，并在dim=0增加维数为1的维度
        if np.random.uniform() > self.epsilon:  # 生成一个在[0, 1)内的随机数，如果小于EPSILON，选择最优动作
            actions_value = self.eval_net.forward(x)  # 通过对评估网络输入状态x，前向传播获得动作值
            action = torch.max(actions_value, 1)[1].data.numpy()  # 输出每一行最大值的索引，并转化为numpy ndarray形式

            action = action[0]  # 输出action的第一个数
        else:  # 随机选择动作
            action = np.random.randint(0, N_ACTIONS)  # 这里action随机等于0或1 (N_ACTIONS = 2)
        return action  # 返回选择的动作 (0或1)

    def store_transition(self, s, a, r, s_):  # 定义记忆存储函数 (这里输入为一个transition)
        transition = np.hstack((s, [a, r], s_))  # 在水平方向上拼接数组
        # 如果记忆库满了，便覆盖旧的数据
        index = self.memory_counter % MEMORY_CAPACITY  # 获取transition要置入的行数
        self.memory[index, :] = transition  # 置入transition
        self.memory_counter += 1  # memory_counter自加1

    def learn(self):  # 定义学习函数(记忆库已满后便开始学习)
        # 目标网络参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:  # 一开始触发，然后每10步触发
            self.target_net.load_state_dict(self.eval_net.state_dict())  # 将评估网络的参数赋给目标网络
        self.learn_step_counter += 1  # 学习步数自加1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)  # 在[0, 2000)内随机抽取32个数，可能会重复
        b_memory = self.memory[sample_index, :]  # 抽取32个索引对应的32个transition，存入b_memory
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        # 将32个s抽出，转为32-bit floating point形式，并存储到b_s中，b_s为32行4列
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        # 将32个a抽出，转为64-bit integer (signed)形式，并存储到b_a中 (之所以为LongTensor类型，是为了方便后面torch.gather的使用)，b_a为32行1列
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        # 将32个r抽出，转为32-bit floating point形式，并存储到b_s中，b_r为32行1列
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
        # 将32个s_抽出，转为32-bit floating point形式，并存储到b_s中，b_s_为32行4列

        # 获取32个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
        q_eval = self.eval_net(b_s).gather(1, b_a)
        # eval_net(b_s)通过评估网络输出32行每个b_s对应的一系列动作值，然后.gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
        q_next = self.target_net(b_s_).detach()
        # q_next不进行反向传递误差，所以detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(BATCH_SIZE, 1)的形状；最终通过公式得到目标值
        loss = self.loss_func(q_eval, q_target)     # 目标和训练网络的均方差
        self.loss = torch.max(loss)
        self.q_eval = torch.max(q_eval)
        self.q_target = torch.max(q_target)
        # 输入32个评估值和32个目标值，使用均方损失函数
        self.optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward()  # 误差反向传播, 计算参数更新值
        self.optimizer.step()  # 更新评估网络的所有参数

    def save_model(self,e):
        state = {'target_net': self.target_net.state_dict(), 'eval_net': self.eval_net.state_dict(),
                 'optimizer': self.optimizer.state_dict(), 'epoch': e}
        torch.save(state, "a_%s.pt" % (str(e)))

    pass

if __name__ == "__main__":
    try:
        # USVRosbridgeClient.Host = "192.168.3.119"
        # USVRosbridgeClient.Host = "192.168.50.106"
        USVRosbridgeClient.Host = "47.94.163.11"
        USVRosbridgeClient.Port = 9090
        env = Env(action_size=N_ACTIONS,rosHost=USVRosbridgeClient.Host,deviceId="2ec0bf09846e84bf7c5680953464185a930384b5")
        dqn = DQN()
        e = dqn.start_epoch
        print("[INFO] DQN training script started")
        print("wait train button trigger ...")

        while env.isTrainActionTrigger() == 0:
            time.sleep(1)
            pass

        episode_reward_sum = 0 #总奖励值
        goal_count = 0

        for e in range(MAX_Epoch):
            if env.isTrainActionTrigger() == 0:
                print("Stop train ...")
                break

            print("train %d ..."%(e))
            s = env.reset()
            max_distance = env.goal_distance

            # episode_reward_sum = 0  # 初始化该循环对应的episode的总奖励
            done = False
            episode_step = 3000     #原来是6k steps的数量要改
            start = time.time()
            for t in range(episode_step):  # 开始一个episode (每一个循环代表一步)
                if env.isTrainActionTrigger() == 0:
                    print("Stop train step ...")
                    break

                print(f"[DEBUG] Step {t}: choosing action ...")
                a = dqn.choose_action(s)  # 输入该步对应的状态s，选择动作
                print(f"[DEBUG] Step {t}: action = {a}")
                s_, r, done, max_distance = env.step(s, a, max_distance)  # 执行动作，获得反馈
                print(f"[DEBUG] Step {t}: reward = {r}, done = {done}, goal_distance = {env.goal_distance}")

                # # 修改奖励 (不修改也可以，修改奖励只是为了更快地得到训练好的摆杆)
                # x, x_dot, theta, theta_dot = s_
                # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
                # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
                # new_r = r1 + r2

                dqn.store_transition(s, a, r, s_)  # 存储样本
                episode_reward_sum += r  # 逐步加上一个episode内每个step的reward   观测值
                s = s_  # 更新状态
                #loss = 0
                if type(dqn.loss) is not int:
                    loss = float(dqn.loss.detach().numpy())
                else:
                    loss = dqn.loss
                DQN_ROS_Service.updateTrainStatus(e,t,int(episode_reward_sum),goal_count,MAX_Epoch,2)  #最后是个标志位

                if t >= 2500:
                    print("time out!")
                    done = True

                if dqn.memory_counter > MEMORY_CAPACITY:  # 如果累计的transition数量超过了记忆库的固定容量2000
                    # 开始学习 (抽取记忆，即32个transition，并对评估网络参数进行更新，并在开始学习后每隔10次将评估网络的参数赋给目标网络)
                    dqn.learn()
                    #print("learning...")

                now = time.time()
                temp = now - start
                start = now
                #print(max_distance)

                if done or env.get_goalbox:         #要记录goalbox的次数
                    print(f"[DEBUG] Episode ends at step {t}")
                    print("episode en use time %s s , e = %s , t = %s , r = %s , d = %s m , " % (temp, e, t, r, env.goal_distance))
                    print('Ep: %d score: %.2f memory: %d epsilon: %.2f '% (e, episode_reward_sum, dqn.memory_counter, dqn.epsilon))
                    #print(episode_reward_sum,float(dqn.loss),float(dqn.q_eval),float(dqn.q_target))
                    reward_list.append(episode_reward_sum)

                    if r > 500:
                        goal_count += 1

                    break

                if dqn.epsilon > dqn.epsilon_min:               #贪心策略更新
                    dqn.epsilon = dqn.epsilon - 0.0001          #大概要8000步

            if e % 100 == 0:
                dqn.save_model(str(e)) #存模型
                # if t >= 2500:
                #     print("time out!")
                #     done = True       #超时暂时用不到

    # episodes_list = list(range(len(reward_list)))
    # plt.plot(episodes_list, reward_list)
    # plt.xlabel('Episodes')
    # plt.ylabel('Returns')
    # plt.title('DQN Returns')
    # plt.show()


    except Exception as e:
        LogUtil.error(e)
    pass
