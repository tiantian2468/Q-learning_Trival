import numpy as np
import random
import matplotlib.pyplot as plt

class TouristPlanner:
    def __init__(self, num_spots, spot_info, max_time, commute_time):
        self.num_spots = num_spots
        self.spot_info = spot_info
        self.max_time = max_time
        self.commute_time = commute_time  # 通勤时间矩阵
        self.q_table = np.zeros((num_spots, num_spots))  # Q值表
        self.time_left = max_time  # 剩余时间
        self.visited = [False] * num_spots  # 记录景点是否已经被访问过
        self.start_point = 0  # 假设起始点是酒店（景点0）
        self.rewards = []  # 用于存储每个episode的总奖励

    def reset(self):
        self.time_left = self.max_time
        self.visited = [False] * self.num_spots

    def get_state(self):
        return self.visited, self.time_left

    def get_available_actions(self):
        # 返回未被访问过且可以在剩余时间内游览的景点
        available = []
        for i in range(self.num_spots):
            if not self.visited[i] and self.spot_info[i][1] <= self.time_left:
                available.append(i)
        return available

    def take_action(self, action):
        # 更新状态：选择游览一个景点
        self.visited[action] = True
        self.time_left -= self.spot_info[action][1]  # 扣除游览时长

    def get_reward(self, current_spot, action):
        # 奖励：可以根据推荐度、花费、通勤时间等综合设计
        recommendation, duration, _ = self.spot_info[action]

        # 确保索引不超出范围
        if current_spot < self.num_spots and action < self.num_spots:
            commute_time = self.commute_time[current_spot][action]
        else:
            commute_time = 0  # 默认如果通勤时间超出索引则为0

        reward = recommendation - duration - commute_time
        return reward

    def train(self, episodes=1000):
        for episode in range(episodes):
            self.reset()
            total_reward = 0
            current_spot = self.start_point
            while self.time_left > 0:
                available_actions = self.get_available_actions()

                if not available_actions:
                    break

                # 贪婪策略：选择当前最优的动作
                action = max(available_actions, key=lambda a: self.get_reward(current_spot, a))

                # 执行动作并获得奖励
                reward = self.get_reward(current_spot, action)
                self.take_action(action)

                total_reward += reward
                current_spot = action  # 更新当前位置

            self.rewards.append(total_reward)

class AntColonyTouristPlanner(TouristPlanner):
    def train(self, episodes=1000, alpha=1, beta=2):
        # alpha, beta 控制信息素和启发函数的重要性
        pheromone = np.ones((self.num_spots, self.num_spots))  # 初始化信息素矩阵
        for episode in range(episodes):
            self.reset()
            total_reward = 0
            current_spot = self.start_point
            while self.time_left > 0:
                available_actions = self.get_available_actions()

                if not available_actions:
                    break

                # 计算每个可行动作的概率
                total_pheromone = sum(
                    [pheromone[current_spot][action] ** alpha * (1.0 / self.spot_info[action][1]) ** beta for action in
                     available_actions])
                probabilities = [
                    (pheromone[current_spot][action] ** alpha * (
                                1.0 / self.spot_info[action][1]) ** beta) / total_pheromone
                    for action in available_actions
                ]

                # 选择动作（概率选择）
                action = random.choices(available_actions, probabilities)[0]

                # 执行动作并获得奖励
                reward = self.get_reward(current_spot, action)
                self.take_action(action)

                # 更新信息素
                pheromone[current_spot][action] += reward  # 信息素增加的方式可根据需求调整

                total_reward += reward
                current_spot = action  # 更新当前位置

            self.rewards.append(total_reward)

def calculate_recommendation(average_rating, positive_review_rate, alpha=0.5, beta=0.5):
    return alpha * average_rating + beta * positive_review_rate

# 景点信息：[(推荐度, 游览时长, 花费)]
spot_info = [
    (calculate_recommendation(9, 95.6), 2, 15),  # 景点1: 推荐度 9, 游览时长 2小时, 花费 15元
    (calculate_recommendation(7, 86.8), 3, 10),  # 景点2: 推荐度 7, 游览时长 3小时, 花费 10元
    (calculate_recommendation(8, 80.0), 1, 5),  # 景点3: 推荐度 8, 游览时长 1小时, 花费 5元
    (calculate_recommendation(6, 90.0), 4, 20),  # 景点4: 推荐度 6, 游览时长 4小时, 花费 20元
    (calculate_recommendation(10, 93.1), 5, 25),  # 景点5: 推荐度 10, 游览时长 5小时, 花费 25元
    (calculate_recommendation(5, 82.5), 2, 8),  # 景点6: 推荐度 5, 游览时长 2小时, 花费 8元
    (calculate_recommendation(8, 88.5), 3, 18),  # 景点7: 推荐度 8, 游览时长 3小时, 花费 18元
    (calculate_recommendation(7, 80.0), 1, 6),  # 景点8: 推荐度 7, 游览时长 1小时, 花费 6元
    (calculate_recommendation(9, 90.0), 3, 22),  # 景点9: 推荐度 9, 游览时长 3小时, 花费 22元
    (calculate_recommendation(6, 84.0), 2, 12),  # 景点10: 推荐度 6, 游览时长 2小时, 花费 12元
    (calculate_recommendation(8, 90.0), 2, 16),  # 景点11: 推荐度 8, 游览时长 2小时, 花费 16元
    (calculate_recommendation(9, 91.0), 4, 19),  # 景点12: 推荐度 9, 游览时长 4小时, 花费 19元
    (calculate_recommendation(7, 85.0), 1, 7),  # 景点13: 推荐度 7, 游览时长 1小时, 花费 7元
    (calculate_recommendation(6, 80.0), 3, 17),  # 景点14: 推荐度 6, 游览时长 3小时, 花费 17元
    (calculate_recommendation(8, 92.5), 5, 30),  # 景点15: 推荐度 8, 游览时长 5小时, 花费 30元
    (calculate_recommendation(5, 75.0), 2, 11),  # 景点16: 推荐度 5, 游览时长 2小时, 花费 11元
    (calculate_recommendation(7, 82.5), 3, 20),  # 景点17: 推荐度 7, 游览时长 3小时, 花费 20元
    (calculate_recommendation(8, 89.5), 2, 15),  # 景点18: 推荐度 8, 游览时长 2小时, 花费 15元
    (calculate_recommendation(6, 88.0), 4, 23),  # 景点19: 推荐度 6, 游览时长 4小时, 花费 23元
    (calculate_recommendation(9, 92.0), 3, 21)  # 景点20: 推荐度 9, 游览时长 3小时, 花费 21元
]

# 通勤时间矩阵（单位：小时）
commute_time = [
    [0.        , 0.96376963, 1.01744732, 0.8847529 , 0.94981489,
     0.77875   , 0.82429832, 0.82360576, 0.94867449, 0.95597639,
     0.8048188 , 0.95117427, 0.87503211, 0.92725993, 0.98016978,
     0.98427544, 0.82060618, 0.96409232, 0.96028659, 0.93088606],
    [0.96376963, 0.        , 0.06478986, 0.23103771, 0.02068886,
     0.97899446, 0.14234444, 0.14381445, 0.03427308, 0.08851889,
     0.16549556, 0.01297499, 0.19378513, 0.03672938, 0.02129498,
     0.04865662, 0.14433646, 0.03764777, 0.12770755, 0.06472159],
    [1.01744732, 0.06478986, 0.        , 0.28972756, 0.08513487,
     1.04242563, 0.20309647, 0.20496753, 0.06891236, 0.14080627,
     0.22735087, 0.07702158, 0.25489305, 0.09551965, 0.04361138,
     0.08844273, 0.20368176, 0.05340861, 0.17677848, 0.08922142],
    [0.8847529 , 0.23103771, 0.28972756, 0.        , 0.21125646,
     0.7597818 , 0.18390342, 0.18069112, 0.25286065, 0.15007741,
     0.17393861, 0.2228463 , 0.04315018, 0.2208981 , 0.25072953,
     0.20337705, 0.19377446, 0.26507266, 0.12118734, 0.27028453],
    [0.94981489, 0.02068886, 0.08513487, 0.21125646, 0.        ,
     0.95830565, 0.12630869, 0.12746449, 0.04571141, 0.07259761,
     0.1484404 , 0.01215713, 0.17347394, 0.02946777, 0.04190345,
     0.04463675, 0.12927491, 0.05454696, 0.11200627, 0.07284559],
    [0.77875   , 0.97899446, 1.04242563, 0.7597818 , 0.95830565,
     0.        , 0.86887644, 0.86564049, 0.99135344, 0.90796903,
     0.84414794, 0.96819541, 0.78904874, 0.95755174, 1.0001242 ,
     0.96052321, 0.87430344, 1.00669256, 0.88093536, 0.99775679],
    [0.82429832, 0.14234444, 0.20309647, 0.18390342, 0.12630869,
     0.86887644, 0.        , 0.00354106, 0.13668257, 0.14202847,
     0.02560393, 0.129391  , 0.14170136, 0.10770757, 0.1612389 ,
     0.16043054, 0.00998198, 0.15344368, 0.16291108, 0.1329714 ],
    [0.82360576, 0.14381445, 0.20496753, 0.18069112, 0.12746449,
     0.86564049, 0.00354106, 0.        , 0.13888109, 0.10019881,
     0.13846606, 0.02528813, 0.135379  , 0.16064574, 0.12861837,
     0.1137293 , 0.16968704, 0.07430834, 0.17177565, 0.12861837],
    [0.92725993, 0.03672938, 0.09551965, 0.2208981 , 0.02946777,
     0.95755174, 0.10770757, 0.10949666, 0.03383989, 0.09492717,
     0.13183121, 0.02492931, 0.18050534, 0.        , 0.05371727,
     0.07410266, 0.10884173, 0.04939678, 0.13336755, 0.05086825],
    [0.98016978, 0.02129498, 0.04361138, 0.25072953, 0.04190345,
     1.0001242 , 0.1612389 , 0.16290949, 0.03605722, 0.1052525 ,
     0.18497201, 0.03342918, 0.21419432, 0.05371727, 0.        ,
     0.05836561, 0.16255711, 0.02920496, 0.14365022, 0.06532172],
    [0.98427544, 0.04865662, 0.08844273, 0.20337705, 0.04463675,
     0.96052321, 0.16043054, 0.16088153, 0.08289368, 0.05331609,
     0.17956537, 0.05245144, 0.17149116, 0.07410266, 0.05836561,
     0.        , 0.1651626 , 0.08460133, 0.08834923, 0.11306645],
    [0.61545464, 0.10825235, 0.15276132, 0.14533084, 0.09695618,
     0.65572758, 0.00748649, 0.0098126 , 0.10221444, 0.11201203,
     0.0226484 , 0.09860212, 0.1137293 , 0.0816313 , 0.12191783,
     0.12387195, 0.        , 0.11474723, 0.12862232, 0.09775532],
    [0.72306924, 0.02823583, 0.04005646, 0.19880449, 0.04091022,
     0.75501942, 0.11508276, 0.11673163, 0.01257139, 0.09441698,
     0.13383552, 0.03189355, 0.16968704, 0.03704758, 0.02920496,
     0.08460133, 0.15299631, 0.        , 0.16519641, 0.03813991],
    [0.72021494, 0.09578066, 0.13258386, 0.0908905 , 0.08400471,
     0.66070152, 0.12218331, 0.12108434, 0.11828589, 0.02956745,
     0.12809864, 0.09304118, 0.09907778, 0.13336755, 0.14365022,
     0.08834923, 0.17149643, 0.16519641, 0.        , 0.1833042 ],
    [0.69816455, 0.04854119, 0.06691606, 0.2027134 , 0.05463419,
     0.74831759, 0.09972855, 0.10180041, 0.02305653, 0.10820981,
     0.11893092, 0.04639659, 0.17177565, 0.03815119, 0.04899129,
     0.08479984, 0.09775532, 0.02860493, 0.13747815, 0.        ]
]

# 最大游览时间
max_time = 10

# 初始化蚁群优化算法
ant_planner = AntColonyTouristPlanner(num_spots=15, spot_info=spot_info, max_time=max_time, commute_time=commute_time)

# 训练模型
ant_planner.train(episodes=1000)

# 绘制奖励图
plt.plot(ant_planner.rewards, label="Ant Colony Optimization")
plt.title('Total Reward per Episode (Ant Colony Optimization)')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()
plt.show()
