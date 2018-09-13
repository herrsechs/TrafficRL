import numpy as np
from pandas import read_csv


class TrafficEnv(object):
    viewer = None
    action_bound = [-5, 5]  # 运动的取值范围
    action_dim = 1          # 运动向量的维度
    state_dim = 4           # 状态向量的维度 [v, a, rv, dist]
    reward_dim = 4          # 反馈向量的维度

    DATA_V_IDX = 0          # 本车车速的下标
    DATA_A_IDX = 1          # 本车加速度的下标
    DATA_RV_IDX = 2         # 与前车相对车速的下标
    DATA_DIST_IDX = 3       # 与前车车距的下标
    DATA_ID_IDX = 4         # 事件ID的下标
    # data_names = ['IMU.Accel_X', 'IMU.Accel_Y', 'SMS.X_Velocity_T1',
    #               'SMS.X_Range_T1', 'SMS.Y_Range_T1',
    #               'FOT_Control.Speed', 'EventId']
    data_names = ['FOT_Control.Speed', 'IMU.Accel_X', 'SMS.X_Velocity_T0',
                  'SMS.X_RANGE_T0',  'EventId']
    data_cols = [2, 3, 6, 7, 16]
    data_path = r'D:\journal_paper\master_final_paper\data\video.carFollowing\2_car_following.csv'

    def __init__(self):
        self.data = self.load_data(self.data_path, self.data_cols)
        self.data_pointer = 1
        self.state = None
        self.time_step = 0.1
        self.true_state = []
        pass

    def step(self, action):
        # action = np.clip(action, *self.action_bound)[0]
        action = action[0]
        s = self.state
        done = False
        info = dict()
        info['is_end'] = False
        info['is_crash'] = False
        d = self.data[self.data_pointer, :]

        v = s[0] + self.time_step * action                              # 本车速度
        rv = d[self.DATA_RV_IDX] + d[self.DATA_V_IDX] - v               # 相对速度
        dist = s[3] + rv * self.time_step                               # 相对距离
        s_ = [v, action, rv, dist]

        r = [np.abs(s_[0] - d[self.DATA_RV_IDX]),
             np.abs(s_[1] - d[self.DATA_DIST_IDX]),
             np.abs(s_[2] - d[self.DATA_V_IDX]),
             np.abs(action - d[self.DATA_A_IDX])]

        true_s = [d[self.DATA_V_IDX], d[self.DATA_A_IDX], d[self.DATA_RV_IDX], d[self.DATA_DIST_IDX]]
        self.true_state.append(true_s)

        self.state = s_
        info['EventId'] = self.data[self.data_pointer, self.DATA_ID_IDX]
        self.data_pointer = (self.data_pointer + 1) % self.data.shape[0]

        if dist < 0 or v < 0:
            done = True
            info['is_crash'] = True
            r = [100, 100, 100, 100]
        elif self.data_pointer >= self.data.shape[0]:
            done = True
            info['is_end'] = True
        elif d[self.DATA_ID_IDX] != self.data[self.data_pointer, self.DATA_ID_IDX]:
            done = True

        return s_, r, done, info

    def reset(self):
        self.true_state = []
        self.state = self.data[self.data_pointer, :self.state_dim]
        self.data_pointer = (self.data_pointer + 1) % self.data.shape[0]
        return self.state

    def load_data(self, path, cols):
        csv_data = read_csv(path, header=1, usecols=cols)
        return csv_data.values

    def get_true_states(self):
        return self.true_state

    def render(self):
        pass


class Viewer(object):
    def __init__(self):
        pass

    def render(self):
        pass

    def on_draw(self):
        pass