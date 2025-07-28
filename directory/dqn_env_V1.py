import time
import sys
sys.path.append(r"D:\\BaiduNetdiskDownload\\2025_C4智能导航赛道_赛事资源\\usvlib4ros_origin\\usvlib4ros_origin")

import torch
import math
import numpy as np
# from geopy.distance import great_circle
# from geopy import Point
from usvlib4ros.dqn.dqn_ros_service import DictToObject,DQN_ROS_Service
from usvlib4ros.usvRosUtil import USVMathUtil,LogUtil
from usvlib4ros.dqn.twist import Twist
pi = math.pi


class Env:
# 192.168.3.119
    def __init__(self, action_size=25, rosHost="121.41.106.238", deviceId="2ec0bf09846e84bf7c5680953464185a930384b5",
                 include_angle_and_distance_features=False):  # 初始化值
        DQN_ROS_Service.startService(rosHost=rosHost, deviceId=deviceId)
        DQN_ROS_Service.registerTrainActionProxy()
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.action_size = action_size
        self.include_angle_and_distance_features = include_angle_and_distance_features
        self.initGoal = True
        self.get_goalbox = False
        self.goal_distance = 0
        self.score = 0
        self.position = DictToObject(
            **{"position": {"x": 0.0, "y": 0.0, "z": 0.0}, "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 0.0}})  # Pose
        pass


    def isTrainActionTrigger(self):
            return DQN_ROS_Service.isTrainning()

    # 该函数计算当前位置到目标点的距离并返回--这个非常重要
    # 设计一个基于gps的计算
    def getGoalDistace(self):
        trainPos = DQN_ROS_Service.trainPosSubscriber.getMsgData()
        # print(f"[DEBUG] trainPos msg: {trainPos}")
        self.position.x = trainPos.x
        self.position.y = trainPos.z

        self.goal_x = trainPos.targetX
        self.goal_y = trainPos.targetZ

        distance = math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y)
        goal_distance = round(distance, 2)

        return goal_distance

    # 计算经纬度的距离 lat lng index应该是航线给出的
    # def getTrueDistance(self, index):
    #     currentPos = DQN_ROS_Service.poseSubscriber.getMsgData()
    #     routeList = DQN_ROS_Service.routeSubscriber.getMsgData()
    #
    #     start = routeList.startIndex
    #     if not hasattr(self, 'start'):
    #         self.start = routeList.startIndex  # 第一次调用函数时初始化 start
    #
    #
    #     shipPos = Point(currentPos.lat, currentPos.lng)
    #     targetPos = Point(routeList.points[index].lat, routeList.points[index].lng)
    #     trueDistance = great_circle(shipPos, targetPos).meters

    #     return trueDistance
    #
    # def getMaxRoute(self):
    #     routeList = DQN_ROS_Service.routeSubscriber.getMsgData()
    #     maxEpoch = len(routeList.points)
    #
    #     return maxEpoch


    # 接收订阅的odom消息，并解析出机器人当前的位置和朝向


    def getOdometry(self):
        trainPos = DQN_ROS_Service.trainPosSubscriber.getMsgData()
        imu_data = DQN_ROS_Service.imuSubscriber.getMsgData()

        if not trainPos or not imu_data:
            print("[WARN] getOdometry: data not ready")
            self.heading = 0.0
            return

    # 更新当前 position
        self.position.x = trainPos.x
        self.position.y = trainPos.z

    # 更新 orientation
        orientation = imu_data.orientation
        self.position.orientation = orientation

        roll_deg, pitch_deg, yaw_deg = USVMathUtil.quaternionToEulerAngles(orientation.x, orientation.y, orientation.z,
                                                                       orientation.w)  # 将四元数转换成欧拉角，计算出当前机器人的位置和朝向
        yaw_deg_corr = yaw_deg + 90
        yaw = math.radians(yaw_deg_corr)  # 转为弧度

    # orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        goal_angle = math.atan2(self.goal_y - self.position.position.y, self.goal_x - self.position.position.x)

        heading = goal_angle - yaw
    # print(f"heading:{heading}")
        if heading > math.pi:
            heading -= 2 * math.pi

        elif heading < -math.pi:
            heading += 2 * math.pi

        self.heading = round(heading, 2)
        print(
            f"[getOdometry] yaw = {yaw_deg_corr:.2f}°, goal_angle = {math.degrees(goal_angle):.2f}°, heading = {math.degrees(self.heading):.2f}°")

    # getstate函数，获取机器人当前状态
    # 在功能测试钟，主要是获取到目标点之间的距离，只需要获取距离即可（使用gps）


    def getState(self):
        scan_range = self.getLaserData()  # 180维
        current_distance = self.getDistanceToGoal()  # 当前到目标的距离（1维）
        heading = self.getHeadingToGoal()  # 当前朝向与目标的角度差（1维）

        usv_x, usv_y = self.usv_pose[0], self.usv_pose[1]
        goal_x, goal_y = self.goal[0], self.goal[1]
        dx = goal_x - usv_x
        dy = goal_y - usv_y
        dx /= 10.0
        dy /= 10.0

        if hasattr(self, 'prev_heading'):
            delta_heading = heading - self.prev_heading
            delta_distance = current_distance - self.prev_distance
        else:
            delta_heading = 0.0
            delta_distance = 0.0

        self.prev_heading = heading
        self.prev_distance = current_distance

        goal_info = [heading, current_distance] * 5  # 10维
        additional_info = [dx, dy, delta_heading, delta_distance]  # 4维
        obstacle_min_range, obstacle_angle = self.getObstacleInfo()
        obstacle_info = [obstacle_min_range, obstacle_angle]

        state = scan_range + goal_info + additional_info + obstacle_info  # 共：180 + 10 + 4 + 2 = **196维**

        if self.include_angle_and_distance_features:
            heading_sin = math.sin(heading)
            heading_cos = math.cos(heading)
            state += [heading_sin, heading_cos, current_distance]  # 新增3维

        return state, self.isDone()


    def setReward(self, state, done, action, max_distance):       # 少一个方向reward    scanreward和obreward只能有一个好像
        yaw_reward = []
        obstacle_min_range = state[-2]
        current_distance = state[-3]
        heading = state[-4]
        scan_reward = 0
        ob_reward = 0

        if current_distance > 1:
            scan_reward = 1 - (current_distance / max_distance)
            if scan_reward < 0:
                scan_reward = scan_reward * 2
            else:
                scan_reward = scan_reward * 5
        # if obstacle_min_range <= 2:
        #     scan_reward = min(scan_reward,obstacle_min_range)

        if current_distance < max_distance:                        #设置新的max返回
            max_distance = current_distance

        # 从五个方向上给出的奖励值，也就是对应了五个action 这里随着action要变动

        for i in range(5):
            angle = -pi / 4 + heading + (pi / 8 * i) + pi / 2
            tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
            yaw_reward.append(tr)

        distance_rate = 2 ** (current_distance / self.goal_distance)    # self.dis是全局的距离吗 这里的reward值都是正数
        forward_reward = ((round(yaw_reward[action] * 5, 2)) * distance_rate)

        if obstacle_min_range < 3:
            ob_reward = -5
        if current_distance < 3:
            ob_reward = 1
        # else:
        #     ob_reward = 1                                       #考虑这里就不要这个了

        # reward = scan_reward * 0.5  + forward_reward * 0.3 + ob_reward * 0.2        #尝试使用加权奖励
        reward = scan_reward * 0.6 + ob_reward * 0.2 + forward_reward * 0.2

        #sprint(self.goal_distance, max_distance)

        if self.get_goalbox:
            LogUtil.info("Goal!!")
            reward += 1000

            twist = Twist()
            DQN_ROS_Service.updateVehicleAction(twist,0, 0)
            print("reset by Goal")
            self.goal_x, self.goal_y = DQN_ROS_Service.wait_dest_position_refresh()
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False
        elif done:
            LogUtil.info("Collision!!")
            reward += -500
            twist = Twist()
            DQN_ROS_Service.updateVehicleAction(twist,0,0)
            print("reset by Collision")
            self.goal_x, self.goal_y = DQN_ROS_Service.wait_dest_position_refresh()
            # time.sleep(5)
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False

            # self.goal_x, self.goal_y = DQN_ROS_Service.wait_dest_position_refresh()
            # self.goal_distance = self.getGoalDistace()
            # self.get_goalbox = False

        return reward, max_distance

    def reset(self):
        # print("[DEBUG] reset() start")
        try:
            # print("[DEBUG] calling wait_for_service_reset()")
            DQN_ROS_Service.wait_for_service_reset(timeout=5)
            # print("[DEBUG] reset service call done")
        except Exception as e:
            print("[ERROR] reset service exception:", e)
            pass
        data = None
        while data is None:
            try:
                # print("[DEBUG] waiting for laser scan data ...")
                data = DQN_ROS_Service.wait_for_message_laserScan(timeout=5)
                # if data is not None:
                #     print("[DEBUG] received laser scan data")
            except Exception as e:
                print("[ERROR] laser scan wait exception:", e)
                pass

        # print("[DEBUG] preparing goal position ...")
        self.initGoal = True
        if self.initGoal:
            try:

                self.goal_x, self.goal_y = DQN_ROS_Service.wait_dest_position_refresh()
                # print(f"[DEBUG] goal position set to ({self.goal_x}, {self.goal_y})")
                self.goal_distance = self.getGoalDistace()
                # print(f"[DEBUG] goal distance: {self.goal_distance}")
            except Exception as e:
                print("[ERROR] goal refresh exception:", e)
                pass
            # self.goal_x, self.goal_y = self.respawn_goal.getPosition(
            self.initGoal = False

        self.goal_distance = self.getGoalDistace()
        # print("[DEBUG] calling getState() ...")
        state, done = self.getState(data)
        # print("[DEBUG] getState() completed")

        return np.asarray(state)

        # step函数是用于执行一个动作并观察环境反馈的函数。它接收一个动作作为输入，并返回执行该动作后的新状态、奖励和完成标志。
    def step(self, state, action, max_distance):
        obstacle_min_range = state[-2]  #最近障碍物距离
        current_distance = state[-3]    #与目标距离
        heading = state[-4]
        max_angular_vel = 100 #
        ang_vel = ((self.action_size - 1) / 2 - action) * 100.0 / ((self.action_size - 1) / 2)      #计算角度值

        vel_cmd = Twist()
        vel_cmd.angular.z = round(ang_vel,0) * 2 # [0，100]转向速度百分比

        vel_cmd.linear.x = 0.5 #0.5                   #设置速度默认值

        # 实现靠近障碍物的时候速度慢，离障碍物远的时候速度快，查看是否可以减速
        if obstacle_min_range < 4:                #0.5 1.3是碰撞
            vel_cmd.linear.x = 0.1 #obstacle_min_range * 0.1  # m/s range/2，让速度再小一点，不然可能影响碰撞  /2*0.5
        if current_distance < 3:                  #与目标点之间的距离
            vel_cmd.linear.x = 0.1 #current_distance * 0.1  油门给到20%

        vel_cmd.linear.x = min(round(vel_cmd.linear.x * 100 / 0.5 ,0),100) # [0,100]油门百分比
        DQN_ROS_Service.updateVehicleAction(twist=vel_cmd,heading=heading,distance=current_distance)
        # print(vel_cmd.angular.z, vel_cmd.linear.x)          #测试角度和速度

        data = None
        while data is None:
            try:
                data = DQN_ROS_Service.wait_for_message_laserScan(timeout=5)
            except:
                pass

        state, done = self.getState(data)
        reward, max_distance = self.setReward(state, done, action, max_distance)

        return np.asarray(state), reward, done, max_distance

    pass
