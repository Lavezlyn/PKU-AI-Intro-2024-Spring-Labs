import numpy as np
from typing import List
from utils import TreeNode
from simuScene import PlanningMap
### 定义一些你需要的变量和函数 ###
STEP_DISTANCE = 0.5
TARGET_THREHOLD = 0.25
MAX_ITER = 20000
### 定义一些你需要的变量和函数 ###

class RRT:
    def __init__(self, walls) -> None:
        """
        输入包括地图信息，你需要按顺序吃掉的一列事物位置 
        注意：只有按顺序吃掉上一个食物之后才能吃下一个食物，在吃掉上一个食物之前Pacman经过之后的食物也不会被吃掉
        """
        self.map = PlanningMap(walls)
        self.walls = walls
        
        # 其他需要的变量
        ### 你的代码 ###      
        self.current_idx = 0
        self.iter_cnt = 0
        ### 你的代码 ###
        
        # 如有必要，此行可删除
        self.path = None
        
        
    def find_path(self, current_position, next_food):
        """
        在程序初始化时，以及每当 pacman 吃到一个食物时，主程序会调用此函数
        current_position: pacman 当前的仿真位置
        next_food: 下一个食物的位置
        
        本函数的默认实现是调用 build_tree，并记录生成的 path 信息。你可以在此函数增加其他需要的功能
        """
        
        ### 你的代码 ###      
        self.current_idx = 0
        self.iter_cnt = 0
        ### 你的代码 ###
        # 如有必要，此行可删除
        self.path = self.build_tree(current_position, next_food)
        # 优化已得到的路径
        self.optimize_path()
        if len(self.path) == 0:
            print(f"current_position: {current_position}, next_food: {next_food}")
            print("No path found.")
        
        
    def get_target(self, current_position, current_velocity):
        """
        主程序将在每个仿真步内调用此函数，并用返回的位置计算 PD 控制力来驱动 pacman 移动
        current_position: pacman 当前的仿真位置
        current_velocity: pacman 当前的仿真速度
        一种可能的实现策略是，仅作为参考：
        （1）记录该函数的调用次数
        （2）假设当前 path 中每个节点需要作为目标 n 次
        （3）记录目前已经执行到当前 path 的第几个节点，以及它的执行次数，如果超过 n，则将目标改为下一节点
        
        你也可以考虑根据当前位置与 path 中节点位置的距离来决定如何选择 target
        
        同时需要注意，仿真中的 pacman 并不能准确到达 path 中的节点。你可能需要考虑在什么情况下重新规划 path
        """
        target_pose = np.zeros_like(current_position)
        ### 你的代码 ###
        print(f"current_idx: {self.current_idx}, iter_cnt: {self.iter_cnt}")
        print(f"path: {self.path}")
        n = 30
        if self.iter_cnt < n and np.linalg.norm(current_position - self.path[self.current_idx]) > TARGET_THREHOLD:
            target_pose = self.path[self.current_idx]
            self.iter_cnt += 1
        else:
            self.iter_cnt = 0
            self.current_idx += 1  
            if self.current_idx >= len(self.path):
                self.current_idx = len(self.path) - 1
                self.find_path(current_position, self.path[-1]) 
            if self.current_idx < len(self.path):
                target_pose = self.path[self.current_idx]
            else:
                print("current_idx: ", self.current_idx, "path: ", self.path)
                target_pose = self.path[-1]
        ### 你的代码 ###
        return target_pose
        
    ### 以下是RRT中一些可能用到的函数框架，全部可以修改，当然你也可以自己实现 ###
    def build_tree(self, start, goal):
        """
        实现你的快速探索搜索树，输入为当前目标食物的编号，规划从 start 位置食物到 goal 位置的路径
        返回一个包含坐标的列表，为这条路径上的pd targets
        你可以调用find_nearest_point和connect_a_to_b两个函数
        另外self.map的checkoccupy和checkline也可能会需要，可以参考simuScene.py中的PlanningMap类查看它们的用法
        """
        path = []
        s_graph: List[TreeNode] = []
        s_graph.append(TreeNode(-1, start[0], start[1]))
        g_graph: List[TreeNode] = []
        g_graph.append(TreeNode(-1, float(goal[0]), float(goal[1])))
        closest_point = None
        min_distance = float('inf')
        ### 你的代码 ###
        for _ in range(MAX_ITER):
            rand_point = self.random_point(goal)
            nearest_idx, _ = self.find_nearest_point(rand_point, s_graph)
            nearest_node = s_graph[nearest_idx]
            is_empty, newpoint = self.connect_a_to_b(nearest_node.pos, rand_point)
            can_rush = not self.map.checkline(newpoint.tolist(), goal.tolist())[0]
            if is_empty or can_rush:
                new_node = TreeNode(nearest_idx, newpoint[0], newpoint[1])
                s_graph.append(new_node)
                # 检查是否与目标树相连
                meet_idx, _ = self.find_nearest_point(newpoint, g_graph)
                meet_node = g_graph[meet_idx]
                if np.linalg.norm(newpoint - meet_node.pos) < TARGET_THREHOLD:
                    # 构建从起点到相遇点的路径
                    while new_node.parent_idx != -1:
                        path.append(new_node.pos)
                        new_node = s_graph[new_node.parent_idx]
                    path.reverse()
                    # 构建从相遇点到终点的路径
                    while meet_node.parent_idx != -1:
                        path.append(meet_node.pos)
                        meet_node = g_graph[meet_node.parent_idx]
                    break
            rand_point = self.random_point(start)
            nearest_idx, _ = self.find_nearest_point(rand_point, g_graph)
            nearest_node = g_graph[nearest_idx]
            is_empty, newpoint = self.connect_a_to_b(nearest_node.pos, rand_point)
            can_rush = not self.map.checkline(newpoint.tolist(), start.tolist())[0]
            if is_empty or can_rush:
                new_node = TreeNode(nearest_idx, newpoint[0], newpoint[1])
                g_graph.append(new_node)
                # 检查是否与起点树相连
                meet_idx, _ = self.find_nearest_point(newpoint, s_graph)
                meet_node = s_graph[meet_idx]
                if np.linalg.norm(newpoint - meet_node.pos) < TARGET_THREHOLD:
                    break  
        ### 你的代码 ###
        return path
    
    def random_point(self, goal):
        rand_point = np.zeros(2)
        if np.random.rand() < 0.1:
            rand_point = goal
        else:
            rand_point[0] = np.random.uniform(0, self.map.height)
            rand_point[1] = np.random.uniform(0, self.map.width)
        return rand_point
    
    def optimize_path(self):
        if not self.path or len(self.path) < 3:
            return
        op_path =  [self.path[0]]
        cur = self.path[0]
        for i in range(2, len(self.path)):
            if self.map.checkline(cur.tolist(), self.path[i].tolist())[0]:
                op_path.append(self.path[i-1])
                cur = self.path[i-1]
        op_path.append(self.path[-1])
        self.path = op_path

    @staticmethod
    def find_nearest_point(point, graph):
        """
        找到图中离目标位置最近的节点，返回该节点的编号和到目标位置距离、
        输入：
        point：维度为(2,)的np.array, 目标位置
        graph: List[TreeNode]节点列表
        输出：
        nearest_idx, nearest_distance 离目标位置距离最近节点的编号和距离
        """
        nearest_idx = -1
        nearest_distance = 10000000.
        ### 你的代码 ###
        for idx, node in enumerate(graph):
            distance = np.linalg.norm(point - node.pos)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_idx = idx
        ### 你的代码 ###
        return nearest_idx, nearest_distance
    
    def connect_a_to_b(self, point_a, point_b):
        """
        以A点为起点，沿着A到B的方向前进STEP_DISTANCE的距离，并用self.map.checkline函数检查这段路程是否可以通过
        输入：
        point_a, point_b: 维度为(2,)的np.array，A点和B点位置，注意是从A向B方向前进
        输出：
        is_empty: bool，True表示从A出发前进STEP_DISTANCE这段距离上没有障碍物
        newpoint: 从A点出发向B点方向前进STEP_DISTANCE距离后的新位置，如果is_empty为真，之后的代码需要把这个新位置添加到图中
        """
        is_empty = False
        newpoint = np.zeros(2)
        ### 你的代码 ###
        direction = point_b - point_a
        distance = np.linalg.norm(direction)
        direction /= distance
        newpoint = point_a + direction * STEP_DISTANCE
        pointA = point_a.tolist()
        pointB = point_b.tolist()
        obstacle, _ = self.map.checkline(pointA, pointB)
        is_empty = not obstacle
        ### 你的代码 ###
        return is_empty, newpoint


