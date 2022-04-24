import numpy as np
import random
import math
import queue

class Offload:

    def __init__(self, num_iot, num_fog, num_time, max_delay):

        # INPUT DATA
        self.n_iot = num_iot #移动设备
        self.n_fog = num_fog#边缘节点
        self.n_time = num_time #总时隙
        self.duration = 0.1 #时隙
        self.duration = 1 #

        # test
        self.drop_trans_count = 0
        self.drop_fog_count = 0
        self.drop_iot_count = 0

        # CONSIDER A SCENARIO RANDOM IS NOT GOOD
        # LOCAL CAP SHOULD NOT BE TOO SMALL, OTHERWISE, THE STATE MATRIX IS TOO LARGE (EXCEED THE MAXIMUM)
        # SHOULD NOT BE LESS THAN ONE
        #考虑随机情况不好
        # 局部上限不能太小，否则状态矩阵太大（超过最大值）
        # 不应少于一个
        self.comp_cap_iot = 0.4 * np.ones(self.n_iot) * self.duration  # 2.5 Gigacycles per second  * duration 每秒2.5千兆周*持续时间，iot计算队列能力矩阵
        self.comp_cap_fog = 3.2 * np.ones([self.n_fog]) * self.duration  # Gigacycles per second * duration 千兆周/秒*持续时间，fog计算队列能力矩阵
        self.tran_cap_iot = 14 * np.ones([self.n_iot, self.n_fog]) * self.duration  # Mbps * duration Mbps*持续时间，iot传输队列
        self.comp_density = 0.31 * np.ones([self.n_iot])  # 0.297 Gigacycles per Mbits 每兆比特0.297千兆周
        self.max_delay = max_delay # time slots  时隙？

        # BITARRIVE_SET (MARKOVIAN)
        self.task_arrive_prob = 0.2#任务到达率
        self.max_bit_arrive = 8 # Mbits 任务最大
        self.min_bit_arrive = 8# Mbits 任务最小
        self.bitArrive_set = np.arange(self.min_bit_arrive, self.max_bit_arrive, 0.1)#返回一个2-5，步长为0.1的列表
        self.bitArrive = np.zeros([self.n_time, self.n_iot])#n_time为行，n_iot为列

        #设备为智能体
        # ACTION: 0, local; 1, fog 0; 2, fog 1; ...; n, fog n - 1
        self.n_actions = 1 + num_fog
        # STATE: [A, t^{comp}, t^{tran}, [B^{fog}]]
        self.n_features = 1 + 1 + 1 + num_fog#任务大小，计算队列信息，传输队列信息，边缘节点前一时隙负载水平
        # LSTM STATE
        self.n_lstm_state = self.n_fog  # [fog1, fog2, ...., fogn]节点数量

        # TIME COUNT
        self.time_count = int(0)

        # QUEUE INITIALIZATION: size -> task size; time -> arrive time  队列初始化：大小->任务大小；时间->到达时间
        self.Queue_iot_comp = list() #本地计算队列
        self.Queue_iot_tran = list()#本地传输队列
        self.Queue_fog_comp = list()#节点计算队列

        for iot in range(self.n_iot):#为队列初始化
            self.Queue_iot_comp.append(queue.Queue())
            self.Queue_iot_tran.append(queue.Queue())
            self.Queue_fog_comp.append(list())
            for fog in range(self.n_fog):
                self.Queue_fog_comp[iot].append(queue.Queue())#每个fog的设备队列一个iot数量的队列

        # QUEUE INFO INITIALIZATION队列信息初始化
        self.t_iot_comp = - np.ones([self.n_iot])
        self.t_iot_tran = - np.ones([self.n_iot])
        self.b_fog_comp = np.zeros([self.n_iot, self.n_fog])

        # TASK INDICATOR任务指示器
        self.task_on_process_local = list()
        self.task_on_transmit_local = list()
        self.task_on_process_fog = list()
        self.fog_iot_m = np.zeros(self.n_fog)
        self.fog_iot_m_observe = np.zeros(self.n_fog)#没看懂这俩

        for iot in range(self.n_iot):
            self.task_on_process_local.append({'size': np.nan, 'time': np.nan, 'remain': np.nan})
            self.task_on_transmit_local.append({'size': np.nan, 'time': np.nan,
                                                'fog': np.nan, 'remain': np.nan})
            self.task_on_process_fog.append(list())
            for fog in range(self.n_fog):
                self.task_on_process_fog[iot].append({'size': np.nan, 'time': np.nan, 'remain': np.nan})

        # TASK DELAY
        self.process_delay = np.zeros([self.n_time, self.n_iot])    # total delay 总延误
        self.process_delay_unfinish_ind = np.zeros([self.n_time, self.n_iot])  # unfinished indicator未完成指示器
        self.process_delay_trans = np.zeros([self.n_time, self.n_iot])  # transmission delay (if applied)传输延迟（如适用）

        self.fog_drop = np.zeros([self.n_iot, self.n_fog])

    # reset the network scenario重置网络场景
    def reset(self, bitArrive):

        # test
        self.drop_trans_count = 0
        self.drop_fog_count = 0
        self.drop_iot_count = 0

        # BITRATE
        self.bitArrive = bitArrive

        # TIME COUNT
        self.time_count = int(0)

        # QUEUE INITIALIZATION
        self.Queue_iot_comp = list()
        self.Queue_iot_tran = list()
        self.Queue_fog_comp = list()

        for iot in range(self.n_iot):
            self.Queue_iot_comp.append(queue.Queue())
            self.Queue_iot_tran.append(queue.Queue())
            self.Queue_fog_comp.append(list())
            for fog in range(self.n_fog):
                self.Queue_fog_comp[iot].append(queue.Queue())

        # QUEUE INFO INITIALIZATION
        self.t_iot_comp = - np.ones([self.n_iot])
        self.t_iot_tran = - np.ones([self.n_iot])
        self.b_fog_comp = np.zeros([self.n_iot, self.n_fog])

        # TASK INDICATOR
        self.task_on_process_local = list()
        self.task_on_transmit_local = list()
        self.task_on_process_fog = list()

        for iot in range(self.n_iot):
            self.task_on_process_local.append({'size': np.nan, 'time': np.nan, 'remain': np.nan})
            self.task_on_transmit_local.append({'size': np.nan, 'time': np.nan,
                                                'fog': np.nan, 'remain': np.nan})
            self.task_on_process_fog.append(list())
            for fog in range(self.n_fog):
                self.task_on_process_fog[iot].append({'size': np.nan, 'time': np.nan, 'remain': np.nan})

        # TASK DELAY
        self.process_delay = np.zeros([self.n_time, self.n_iot])
        self.process_delay_unfinish_ind = np.zeros([self.n_time, self.n_iot])  # unfinished indicator
        self.process_delay_trans = np.zeros([self.n_time, self.n_iot])  # transmission delay (if applied)

        self.fog_drop = np.zeros([self.n_iot, self.n_fog])

        # INITIAL
        observation_all = np.zeros([self.n_iot, self.n_features])
        for iot_index in range(self.n_iot):
            # observation is zero if there is no task arrival如果没有任务到达，则观察值为零
            if self.bitArrive[self.time_count, iot_index] != 0:
                # state [A, B^{comp}, B^{tran}, [B^{fog}]]
                observation_all[iot_index, :] = np.hstack([
                    self.bitArrive[self.time_count, iot_index], self.t_iot_comp[iot_index],
                    self.t_iot_tran[iot_index],
                    np.squeeze(self.b_fog_comp[iot_index, :])]) #np.hstack():在水平方向上平铺
                # print("self.bitArrive[self.time_count, iot_index]")
                # print(self.bitArrive[self.time_count, iot_index])
                # print("self.t_iot_comp[iot_index]")
                # print(self.t_iot_comp[iot_index])
                # print("self.t_iot_tran[iot_index]")
                # print(self.t_iot_tran[iot_index])
                # print("np.squeeze(self.b_fog_comp[iot_index, :])")
                # print(np.squeeze(self.b_fog_comp[iot_index, :]))

        lstm_state_all = np.zeros([self.n_iot, self.n_lstm_state])
        print("observation_all")
        print(observation_all)
        return observation_all, lstm_state_all

    # perform action, observe state and delay (several steps later)执行操作、观察状态和延迟（几步之后）
    def step(self, action):

        # EXTRACT ACTION FOR EACH IOT提取每个物联网的操作
        iot_action_local = np.zeros([self.n_iot], np.int32)
        iot_action_fog = np.zeros([self.n_iot], np.int32)
        for iot_index in range(self.n_iot):
            iot_action = action[iot_index]
            iot_action_fog[iot_index] = int(iot_action - 1)
            if iot_action == 0:
                iot_action_local[iot_index] = 1

        # COMPUTATION QUEUE UPDATE ===================计算队列更新
        for iot_index in range(self.n_iot):

            iot_bitarrive = np.squeeze(self.bitArrive[self.time_count, iot_index])
            iot_comp_cap = np.squeeze(self.comp_cap_iot[iot_index])
            iot_comp_density = self.comp_density[iot_index]

            # INPUT
            if iot_action_local[iot_index] == 1:
                tmp_dict = {'size': iot_bitarrive, 'time': self.time_count}
                self.Queue_iot_comp[iot_index].put(tmp_dict)

            # TASK ON PROCESS
            if math.isnan(self.task_on_process_local[iot_index]['remain']) \
                    and (not self.Queue_iot_comp[iot_index].empty()):
                while not self.Queue_iot_comp[iot_index].empty():
                    # only put the non-zero task to the processor
                    get_task = self.Queue_iot_comp[iot_index].get()
                    # since it is at the beginning of the time slot, = self.max_delay is acceptable
                    if get_task['size'] != 0:
                        if self.time_count - get_task['time'] + 1 <= self.max_delay:
                            self.task_on_process_local[iot_index]['size'] = get_task['size']
                            self.task_on_process_local[iot_index]['time'] = get_task['time']
                            self.task_on_process_local[iot_index]['remain'] \
                                = self.task_on_process_local[iot_index]['size']
                            break
                        else:
                            self.process_delay[get_task['time'], iot_index] = self.max_delay
                            self.process_delay_unfinish_ind[get_task['time'], iot_index] = 1

            # PROCESS
            if self.task_on_process_local[iot_index]['remain'] > 0:
                self.task_on_process_local[iot_index]['remain'] = \
                    self.task_on_process_local[iot_index]['remain'] - iot_comp_cap / iot_comp_density
                # if no remain, compute processing delay
                if self.task_on_process_local[iot_index]['remain'] <= 0:
                    self.process_delay[self.task_on_process_local[iot_index]['time'], iot_index] \
                        = self.time_count - self.task_on_process_local[iot_index]['time'] + 1
                    self.task_on_process_local[iot_index]['remain'] = np.nan
                elif self.time_count - self.task_on_process_local[iot_index]['time'] + 1 == self.max_delay:
                    self.process_delay[self.task_on_process_local[iot_index]['time'], iot_index] = self.max_delay
                    self.process_delay_unfinish_ind[self.task_on_process_local[iot_index]['time'], iot_index] = 1
                    self.task_on_process_local[iot_index]['remain'] = np.nan

                    self.drop_iot_count = self.drop_iot_count + 1

            # OTHER INFO self.t_iot_comp[iot_index]
            # update self.t_iot_comp[iot_index] only when iot_bitrate != 0
            # 其他信息自身。物联网公司[物联网索引]
            # 仅当物联网比特率！=0
            if iot_bitarrive != 0:
                tmp_tilde_t_iot_comp = np.max([self.t_iot_comp[iot_index] + 1, self.time_count])
                self.t_iot_comp[iot_index] = np.min([tmp_tilde_t_iot_comp
                                                    + math.ceil(iot_bitarrive * iot_action_local[iot_index]
                                                     / (iot_comp_cap / iot_comp_density)) - 1,
                                                    self.time_count + self.max_delay - 1])

        # FOG QUEUE UPDATE =========================
        for iot_index in range(self.n_iot):

            iot_comp_density = self.comp_density[iot_index]

            for fog_index in range(self.n_fog):

                # TASK ON PROCESS
                if math.isnan(self.task_on_process_fog[iot_index][fog_index]['remain']) \
                        and (not self.Queue_fog_comp[iot_index][fog_index].empty()):
                    while not self.Queue_fog_comp[iot_index][fog_index].empty():
                        get_task = self.Queue_fog_comp[iot_index][fog_index].get()
                        if self.time_count - get_task['time'] + 1 <= self.max_delay:
                            self.task_on_process_fog[iot_index][fog_index]['size'] \
                                = get_task['size']
                            self.task_on_process_fog[iot_index][fog_index]['time'] \
                                = get_task['time']
                            self.task_on_process_fog[iot_index][fog_index]['remain'] \
                                = self.task_on_process_fog[iot_index][fog_index]['size']
                            break
                        else:
                            self.process_delay[get_task['time'], iot_index] = self.max_delay
                            self.process_delay_unfinish_ind[get_task['time'], iot_index] = 1

                # PROCESS
                self.fog_drop[iot_index, fog_index] = 0
                if self.task_on_process_fog[iot_index][fog_index]['remain'] > 0:
                    self.task_on_process_fog[iot_index][fog_index]['remain'] = \
                        self.task_on_process_fog[iot_index][fog_index]['remain'] \
                        - self.comp_cap_fog[fog_index] / iot_comp_density / self.fog_iot_m[fog_index]
                    # if no remain, compute processing delay
                    if self.task_on_process_fog[iot_index][fog_index]['remain'] <= 0:
                        self.process_delay[self.task_on_process_fog[iot_index][fog_index]['time'],iot_index] \
                            = self.time_count - self.task_on_process_fog[iot_index][fog_index]['time'] + 1
                        self.task_on_process_fog[iot_index][fog_index]['remain'] = np.nan
                    elif self.time_count - self.task_on_process_fog[iot_index][fog_index]['time'] + 1 == self.max_delay:
                        self.process_delay[self.task_on_process_fog[iot_index][fog_index]['time'], iot_index] = \
                            self.max_delay
                        self.process_delay_unfinish_ind[self.task_on_process_fog[iot_index][fog_index]['time'],
                                                        iot_index] = 1
                        self.fog_drop[iot_index, fog_index] = self.task_on_process_fog[iot_index][fog_index]['remain']
                        self.task_on_process_fog[iot_index][fog_index]['remain'] = np.nan

                        self.drop_fog_count = self.drop_fog_count + 1

                # OTHER INFO
                if self.fog_iot_m[fog_index] != 0:
                    self.b_fog_comp[iot_index, fog_index] \
                        = np.max([self.b_fog_comp[iot_index, fog_index]
                                  - self.comp_cap_fog[fog_index] / iot_comp_density / self.fog_iot_m[fog_index]
                                  - self.fog_drop[iot_index, fog_index], 0])

        # TRANSMISSION QUEUE UPDATE ===================
        for iot_index in range(self.n_iot):

            iot_tran_cap = np.squeeze(self.tran_cap_iot[iot_index,:])
            iot_bitarrive = np.squeeze(self.bitArrive[self.time_count, iot_index])

            # INPUT
            if iot_action_local[iot_index] == 0:
                tmp_dict = {'size': self.bitArrive[self.time_count, iot_index], 'time': self.time_count,
                            'fog': iot_action_fog[iot_index]}
                self.Queue_iot_tran[iot_index].put(tmp_dict)

            # TASK ON PROCESS
            if math.isnan(self.task_on_transmit_local[iot_index]['remain']) \
                    and (not self.Queue_iot_tran[iot_index].empty()):
                while not self.Queue_iot_tran[iot_index].empty():
                    get_task = self.Queue_iot_tran[iot_index].get()
                    if get_task['size'] != 0:
                        if self.time_count - get_task['time'] + 1 <= self.max_delay:
                            self.task_on_transmit_local[iot_index]['size'] = get_task['size']
                            self.task_on_transmit_local[iot_index]['time'] = get_task['time']
                            self.task_on_transmit_local[iot_index]['fog'] = int(get_task['fog'])
                            self.task_on_transmit_local[iot_index]['remain'] = \
                                self.task_on_transmit_local[iot_index]['size']
                            break
                        else:
                            self.process_delay[get_task['time'], iot_index] = self.max_delay
                            self.process_delay_unfinish_ind[get_task['time'], iot_index] = 1

            # PROCESS
            if self.task_on_transmit_local[iot_index]['remain'] > 0:
                self.task_on_transmit_local[iot_index]['remain'] = \
                    self.task_on_transmit_local[iot_index]['remain'] \
                    - iot_tran_cap[self.task_on_transmit_local[iot_index]['fog']]

                # UPDATE FOG QUEUE
                if self.task_on_transmit_local[iot_index]['remain'] <= 0:
                    tmp_dict = {'size': self.task_on_transmit_local[iot_index]['size'],
                                'time': self.task_on_transmit_local[iot_index]['time']}
                    self.Queue_fog_comp[iot_index][self.task_on_transmit_local[iot_index]['fog']].put(tmp_dict)

                    # OTHER INFO
                    fog_index = self.task_on_transmit_local[iot_index]['fog']
                    self.b_fog_comp[iot_index, fog_index] \
                        = self.b_fog_comp[iot_index, fog_index] + self.task_on_transmit_local[iot_index]['size']
                    self.process_delay_trans[self.task_on_transmit_local[iot_index]['time'], iot_index] \
                        = self.time_count - self.task_on_transmit_local[iot_index]['time'] + 1
                    self.task_on_transmit_local[iot_index]['remain'] = np.nan

                elif self.time_count - self.task_on_transmit_local[iot_index]['time'] + 1 == self.max_delay:
                    self.process_delay[self.task_on_transmit_local[iot_index]['time'], iot_index] = self.max_delay
                    self.process_delay_trans[self.task_on_transmit_local[iot_index]['time'], iot_index] \
                        = self.max_delay
                    self.process_delay_unfinish_ind[self.task_on_transmit_local[iot_index]['time'], iot_index] = 1
                    self.task_on_transmit_local[iot_index]['remain'] = np.nan

                    self.drop_trans_count = self.drop_trans_count + 1

            # OTHER INFO
            if iot_bitarrive != 0:
                tmp_tilde_t_iot_tran = np.max([self.t_iot_tran[iot_index] + 1, self.time_count])
                self.t_iot_comp[iot_index] = np.min([tmp_tilde_t_iot_tran
                                                    + math.ceil(iot_bitarrive * (1 - iot_action_local[iot_index])
                                                     / iot_tran_cap[iot_action_fog[iot_index]]) - 1,
                                                    self.time_count + self.max_delay - 1])

        # COMPUTE CONGESTION (FOR NEXT TIME SLOT)计算拥塞（下一个时隙）
        self.fog_iot_m_observe = self.fog_iot_m
        self.fog_iot_m = np.zeros(self.n_fog)
        for fog_index in range(self.n_fog):
            for iot_index in range(self.n_iot):
                if (not self.Queue_fog_comp[iot_index][fog_index].empty()) \
                        or self.task_on_process_fog[iot_index][fog_index]['remain'] > 0:
                    self.fog_iot_m[fog_index] += 1

        # TIME UPDATE
        self.time_count = self.time_count + 1
        done = False
        if self.time_count >= self.n_time:
            done = True
            # set all the tasks' processing delay and unfinished indicator
            # 设置所有任务的处理延迟和未完成指示器
            for time_index in range(self.n_time):
                for iot_index in range(self.n_iot):
                    if self.process_delay[time_index, iot_index] == 0 and self.bitArrive[time_index, iot_index] != 0:
                        self.process_delay[time_index, iot_index] = (self.time_count - 1) - time_index + 1
                        self.process_delay_unfinish_ind[time_index, iot_index] = 1

        # OBSERVATION
        observation_all_ = np.zeros([self.n_iot, self.n_features])
        lstm_state_all_ = np.zeros([self.n_iot, self.n_lstm_state])
        if not done:
            for iot_index in range(self.n_iot):
                # observation is zero if there is no task arrival
                if self.bitArrive[self.time_count, iot_index] != 0:
                    # state [A, B^{comp}, B^{tran}, [B^{fog}]]
                    observation_all_[iot_index, :] = np.hstack([
                        self.bitArrive[self.time_count, iot_index],
                        self.t_iot_comp[iot_index] - self.time_count + 1,
                        self.t_iot_tran[iot_index] - self.time_count + 1,
                        self.b_fog_comp[iot_index, :]])

                lstm_state_all_[iot_index, :] = np.hstack(self.fog_iot_m_observe) #np.hstack():在水平方向上平铺

        return observation_all_, lstm_state_all_, done
