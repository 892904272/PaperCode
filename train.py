from fog_env import Offload
from RL_brain import DeepQNetwork
import numpy as np
import random
# import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)#打印完整的numpy数组a而不截断


def random_pick(some_list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item


def reward_fun(delay, max_delay, unfinish_indi):

    # still use reward, but use the negative value
    penalty = - max_delay * 2 #-20

    if unfinish_indi:
        reward = penalty
    else:
        reward = - delay

    return reward


def train(iot_RL_list, NUM_EPISODE):

    RL_step = 0

    for episode in range(NUM_EPISODE):

        print(episode)
        print(iot_RL_list[0].epsilon)
        # BITRATE ARRIVAL 比特率到达
        bitarrive = np.random.uniform(env.min_bit_arrive, env.max_bit_arrive, size=[env.n_time, env.n_iot])#上界，下界，n_time*n_iot大小的矩阵,产生任务
        task_prob = env.task_arrive_prob
        #print("*************")
        #print(bitarrive)
        bitarrive = bitarrive * (np.random.uniform(0, 1, size=[env.n_time, env.n_iot]) < task_prob)#产生以任务到达率为标准的任务，ex:0.3的数据为有数据，0.7为0
        # print((np.random.uniform(0, 1, size=[env.n_time, env.n_iot]) < task_prob))
        # print("---------------")
        # print(bitarrive)
        bitarrive[-env.max_delay:, :] = np.zeros([env.max_delay, env.n_iot])#最大时延下的任务全部设置为0
        # print("maxdelay",env.max_delay)
        # print(bitarrive[-env.max_delay:, :])

        # =================================================================================================
        # ========================================= DRL ===================================================
        # =================================================================================================

        # OBSERVATION MATRIX SETTING观测矩阵设置
        history = list()#存储着所有时间的所有设备的观察
        for time_index in range(env.n_time):
            history.append(list())
            for iot_index in range(env.n_iot):
                tmp_dict = {'observation': np.zeros(env.n_features),
                            'lstm': np.zeros(env.n_lstm_state),
                            'action': np.nan,
                            'observation_': np.zeros(env.n_features),
                            'lstm_': np.zeros(env.n_lstm_state)}
                history[time_index].append(tmp_dict)

        reward_indicator = np.zeros([env.n_time, env.n_iot])#每个时隙每个iot设备的奖赏

        #print(bitarrive)
        # INITIALIZE OBSERVATION初始化观察
        observation_all, lstm_state_all = env.reset(bitarrive) #行为iot，列为state

        # TRAIN DRL
        while True:
            # PERFORM ACTION行动
            action_all = np.zeros([env.n_iot])
            for iot_index in range(env.n_iot):

                observation = np.squeeze(observation_all[iot_index, :])#squeeze 函数：从数组的形状中删除单维度条目，即把shape中为1的维度去掉
                # print("observation")
                # print(observation)

                if np.sum(observation) == 0:  #np.sum,对所有元素求和
                    # if there is no task, action = 0 (also need to be stored)如果没有任务，则action=0（也需要存储）
                    action_all[iot_index] = 0
                else:
                    action_all[iot_index] = iot_RL_list[iot_index].choose_action(observation)

                if observation[0] != 0:#任务不为空
                    iot_RL_list[iot_index].do_store_action(episode, env.time_count, action_all[iot_index])

            # OBSERVE THE NEXT STATE AND PROCESS DELAY (REWARD)#观察下一个状态和进程延迟（奖励）
            observation_all_, lstm_state_all_, done = env.step(action_all)
            # print("action_all")
            # print(action_all)

            # should store this information in EACH time slot#应在每个时间段中存储此信息
            for iot_index in range(env.n_iot):
                iot_RL_list[iot_index].update_lstm(lstm_state_all_[iot_index,:])

            process_delay = env.process_delay #行为时间，列为设备数
            # print("process_delay")
            # print(process_delay)
            unfinish_indi = env.process_delay_unfinish_ind
            # print("unfinish_indi")
            # print(unfinish_indi)

            # STORE MEMORY; STORE TRANSITION IF THE TASK PROCESS DELAY IS JUST UPDATED
            # #存储存储器；如果任务进程延迟刚刚更新，则存储转换
            for iot_index in range(env.n_iot):

                history[env.time_count - 1][iot_index]['observation'] = observation_all[iot_index, :]
                history[env.time_count - 1][iot_index]['lstm'] = np.squeeze(lstm_state_all[iot_index, :])
                history[env.time_count - 1][iot_index]['action'] = action_all[iot_index]
                history[env.time_count - 1][iot_index]['observation_'] = observation_all_[iot_index]
                history[env.time_count - 1][iot_index]['lstm_'] = np.squeeze(lstm_state_all_[iot_index,:])

                update_index = np.where((1 - reward_indicator[:,iot_index]) * process_delay[:,iot_index] > 0)[0]

                if len(update_index) != 0:
                    for update_ii in range(len(update_index)):
                        time_index = update_index[update_ii]
                        iot_RL_list[iot_index].store_transition(history[time_index][iot_index]['observation'],
                                                                history[time_index][iot_index]['lstm'],
                                                                history[time_index][iot_index]['action'],
                                                                reward_fun(process_delay[time_index, iot_index],
                                                                           env.max_delay,
                                                                           unfinish_indi[time_index, iot_index]),
                                                                history[time_index][iot_index]['observation_'],
                                                                history[time_index][iot_index]['lstm_'])
                        iot_RL_list[iot_index].do_store_reward(episode, time_index,
                                                               reward_fun(process_delay[time_index, iot_index],
                                                                          env.max_delay,
                                                                          unfinish_indi[time_index, iot_index]))
                        iot_RL_list[iot_index].do_store_delay(episode, time_index,
                                                              process_delay[time_index, iot_index])
                        reward_indicator[time_index, iot_index] = 1

            # ADD STEP (one step does not mean one store)#添加步骤（一个步骤并不意味着一个存储）
            RL_step += 1

            # UPDATE OBSERVATION#更新观测
            observation_all = observation_all_
            lstm_state_all = lstm_state_all_

            # CONTROL LEARNING START TIME AND FREQUENCY#控制学习开始时间和频率
            if (RL_step > 200) and (RL_step % 10 == 0):
                for iot in range(env.n_iot):
                    iot_RL_list[iot].learn()

            # GAME ENDS
            if done:
                break
        #  =================================================================================================
        #  ======================================== DRL END=================================================
        #  =================================================================================================

    print("diedaicishu ")
    print(episode)
    print("process_delay")
    print(process_delay)
    print("总时延：")
    print(process_delay.sum())
    print("process_delay.shape[0]")
    print(process_delay.shape[0])
    print(process_delay.shape[1])

    print("总个数")
    x=0
    for i in range(process_delay.shape[0]):
        for j in range(process_delay.shape[1]):
            if process_delay[i][j]!=0:
                x = x+1
    print(x)
    obadfjk=0
    obadfjk=x
    print("unfinish_indi")
    print(unfinish_indi)
    x=0
    for i in range(unfinish_indi.shape[0]):
        for j in range(unfinish_indi.shape[1]):
            if unfinish_indi[i][j]!=0:
                x = x+1
    print("未完成任务数")
    print(x)
    data = open("za/dqnEnd.txt", 'a')
    print("设备数"+str(process_delay.shape[1])+'总任务数'+str(obadfjk)+"总时延"+str(process_delay.sum())+"未完成任务数"+str(x), file=data)
    data.close()

if __name__ == "__main__":
    i=1
    #while i!=11:
    NUM_IOT = 10 #移动设备
    NUM_FOG = 2#边缘节点
    NUM_EPISODE = 10#迭代次数
    NUM_TIME_BASE = 100#
    MAX_DELAY = 30#时隙    #10时隙代表1s
    NUM_TIME = NUM_TIME_BASE + MAX_DELAY

    # GENERATE ENVIRONMENT生成环境
    env = Offload(NUM_IOT, NUM_FOG, NUM_TIME, MAX_DELAY)

    # GENERATE MULTIPLE CLASSES FOR RL   为RL生成多个类
    iot_RL_list = list()
    for iot in range(NUM_IOT):
        # print(env.n_actions)
        # print(env.n_features)
        # print(env.n_lstm_state)
        # print(env.n_time)
        iot_RL_list.append(DeepQNetwork(env.n_actions, env.n_features, env.n_lstm_state, env.n_time,
                                        learning_rate=0.01,
                                        reward_decay=0.9,
                                        e_greedy=0.99,
                                        replace_target_iter=200,  # each 200 steps, update target net
                                        memory_size=500,  # maximum of memory
                                        ))

    # TRAIN THE SYSTEM
    train(iot_RL_list, NUM_EPISODE)
    print('Training Finished')
    i=i+1
