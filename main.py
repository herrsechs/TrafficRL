from env import TrafficEnv
from rl import DDPG
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

MAX_EPOCH = 500
MAX_EVENT = 100

env = TrafficEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound
r_dim = env.reward_dim

rl = DDPG(a_dim, s_dim, r_dim, a_bound)


def save_result(d, id):
    data_dict = {'FOT_Control.Speed': d[:, 0],
                 'IMU.Accel_X': d[:, 1],
                 'SMS.X_Velocity_T0': d[:, 2],
                 'SMS.X_RANGE_T0': d[:, 3]}
    d_frame = pd.DataFrame(data_dict)
    d_frame.to_csv('./data/result_' + str(id) + '.csv', sep=',')


def plot(data1, data2):
    plt.figure(1)
    plt.subplot(221)
    p11 = plt.plot(data1[:, 0], color='blue')
    p12 = plt.plot(data2[:, 0], color='red')
    plt.gca().add_artist(plt.legend([p11, p12], ['Pred', 'True']))
    plt.title('Speed')

    plt.subplot(222)
    p21 = plt.plot(data1[:, 1], color='blue')
    p22 = plt.plot(data2[:, 1], color='red')
    plt.gca().add_artist(plt.legend([p21, p22], ['Pred', 'True']))
    plt.title('Accel')

    plt.subplot(223)
    p31 = plt.plot(data1[:, 2], color='blue')
    p32 = plt.plot(data2[:, 2], color='red')
    plt.gca().add_artist(plt.legend([p31, p32], ['Pred', 'True']))
    plt.title('R-Speed')

    plt.subplot(224)
    p41 = plt.plot(data1[:, 3], color='blue')
    p42 = plt.plot(data2[:, 3], color='red')
    plt.gca().add_artist(plt.legend([p41, p42], ['Pred', 'True']))
    plt.title('Dist')
    plt.show()


for i in range(MAX_EPOCH):
    s = env.reset()
    done = False
    is_end = False
    while not is_end:
        a = rl.choose_action(s)
        s_, r, done, info = env.step(a)
        print('a: %f, r: %s ' % (a, r))
        print('v: %f, a: %s, rv: %s dist: %s' % (s_[0], s_[1], s_[2], s_[3]))
        if info['is_crash'] is False:
            rl.store_transition(s, a, r, s_)

        if done:
            data = rl.get_memory()
            true_data = np.array(env.get_true_states())
            save_result(data, info['EventId'])
            plot(data, true_data)

            rl.learn()
            s = env.reset()
            done = False
            print('===================Reset Now=====================')

        else:
            s = s_


