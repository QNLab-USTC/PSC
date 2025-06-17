import copy
import random
import datetime
import time
from qns.entity.node.app import Application
from qns.entity.qchannel.qchannel import QuantumChannel
from qns.entity.node.node import QNode
from qns.network.network import QuantumNetwork
from typing import Dict, List, Optional, Tuple
from qns.network.topology import Topology
from qns.network.requests import Request
from gekko import GEKKO
import numpy as np
import math



def random_pick(probabilities):
    x = random.uniform(0, 1)
    if x <= probabilities:
        return 1
    else:
        return 0


p_in = 0.9
init_fidelity = 0
request_solver = []
nodes_number = 0
F_th = []
max_capacity = 100
request_num = 0
capacity_tot = 0

with open("temp_files/topo_information.txt", mode='r') as f_first:
    ss = f_first.readline().replace('nodes_list: ', '')
    ss = ss.replace('\n', '')
    ss = ss.replace('[', '')
    ss = ss.replace(']', '')
    ss = ss.replace('<', '')
    ss = ss.replace('>', '')
    ss = ss.replace('node ', '')
    ss = ss.replace(' ', '')
    ss = ss.split(',')
    nodes_list = ss
    nodes_number = len(nodes_list)
    while True:
        ss = f_first.readline()
        if ss.find('init_fidelity: ', 0, len(ss) - 1) != -1:
            line = ss
            break
    line = line.replace('init_fidelity: ', '').strip('\n')
    init_fidelity = float(line)

path = {}
output_path = {}
path_len = []
with open("temp_files/topo_information.txt", mode='r') as f_first:
    ss = f_first.readline().replace('nodes_list: ', '')
    while True:
        line = f_first.readline()
        if line.find('request_list', 0, len(line) - 1) != -1:
            break
    request_num = f_first.readline()
    request_num = int(request_num)
    for i in range(request_num):
        line = f_first.readline().replace(f'request{i}: ', '')
        line = line.replace('\n', '')
        line = line.replace('<', '')
        line = line.replace('<', '')
        line = line.replace('>', '')
        line = line.replace('node ', '')
        line = line.split('-')
        F_th.append(float(line[2]))
        line = f_first.readline().replace(f'request{i} path: ', '')
        line = line.replace('\n', '')
        line = line.replace('[', '')
        line = line.replace(']', '')
        line = line.replace('<', '')
        line = line.replace('>', '')
        line = line.replace('node ', '')
        line = line.replace(' ', '')
        new_line = line.replace('n', '')
        line = line.split(',')
        new_line = new_line.split(',')
        path[f'request{i}'] = line
        output_path[f'request{i}'] = new_line
    for key, value in path.items():
        path_len.append(len(value) - 1)

r_value = {}
result_r_value = {}
request_cnt = {}
for i in range(nodes_number + 1):
    request_cnt[i] = {}
    for j in range(nodes_number + 1):
        request_cnt[i][j] = 0

with open('temp_files/temp_heuristic.txt',
          mode='r') as f_third:
    line = f_third.readline().strip('\n').split(',')
    for i in range(request_num):
        r_value[i] = {}
        result_r_value[i] = {}
        line = f_third.readline().strip('\n').split(',')
        for j in range(len(line) - 1):
            r_value[i][j] = float(line[j])

new_F = init_fidelity * init_fidelity / (init_fidelity * init_fidelity + (1 - init_fidelity) * (1 - init_fidelity))
r_tot = []
for i in range(request_num):
    r_tot.append(int(max(0.0, np.ceil(
        (path_len[i] * math.log(init_fidelity, F_th[i]) - 1) / (math.log(init_fidelity, F_th[i]) - math.log(new_F, F_th[i]))))))

result_throught = 0.0
result_f_throught = 0.0
result_average_capacity_utilization = 0.0
result_x_value = []
max_iter_number = 50
cnt = 0
random_list = []
flag = 1
with open('temp_files/flag_heuristic.txt', mode='r') as f_flag:
    line = f_flag.readlines()
    flag = int(line[-1].strip('\n'))
if flag == -1:
    for i in range(request_num):
        for j in range(path_len[i]):
            n1 = int(output_path[f'request{i}'][j])
            n2 = int(output_path[f'request{i}'][j + 1])
            request_cnt[n1][n2] = request_cnt[n1][n2] + 1
    for i in range(request_num):
        for j in range(path_len[i]):
            n1 = int(output_path[f'request{i}'][j])
            n2 = int(output_path[f'request{i}'][j + 1])
            r_value[i][j] = float(0.6 / request_cnt[n1][n2])

starttime = time.time()
while cnt < max_iter_number:
    random.seed(str(datetime.datetime.now()))
    # 初始化
    m = GEKKO(remote=False)
    m.options.SOLVER = 1
    m.solver_options = ['minlp_maximum_iterations 10000']
    x_value = m.Array(m.Var, request_num, value=100, lb=0, ub=100)
    random_var = {}
    channel = {}
    use_channel = {}
    temp_r_value = {}
    for i in range(nodes_number + 1):
        random_var[i] = {}
        channel[i] = {}
        use_channel[i] = {}
        for j in range(nodes_number + 1):
            random_var[i][j] = {}
            for z in range(request_num):
                random_var[i][j][z] = -1
    with open("temp_files/topo_information.txt", mode='r') as f_first:
        f_first.readline()
        while True:
            line = f_first.readline()
            if line.find('request_list', 0, len(line) - 1) != -1:
                break
            line = line.replace('l_', '')
            line = line.strip('\n')
            line = line.replace(' ', '')
            line = line.split('=')
            line_line = line[0].split('_')
            n1_idx = int(line_line[0])
            n2_idx = int(line_line[1])
            capacity = int(line[1])
            channel[n1_idx][n2_idx] = channel[n2_idx][n1_idx] = capacity
            use_channel[n1_idx][n2_idx] = use_channel[n2_idx][n1_idx] = 0.0

    # 随机选取 r
    for i in range(request_num):
        l = [x for x in range(path_len[i])]
        temp_r_value[i] = {}
        if r_tot[i] == 0:
            for j in range(path_len[i]):
                n1 = int(output_path[f"request{i}"][j])
                n2 = int(output_path[f"request{i}"][j + 1])
                random_var[n1][n2][i] = random_var[n2][n1][i] = 0
                temp_r_value[i][j] = random_var[n2][n1][i]
            continue
        cnt_r = r_tot[i]
        flag_1 = False
        flag_0 = False
        for j in range(path_len[i]):
            n1 = int(output_path[f"request{i}"][j])
            n2 = int(output_path[f"request{i}"][j + 1])
            if path_len[i] - j == cnt_r:
                flag_1 = True
            if cnt_r == 0:
                flag_0 = True
            if flag_1:
                random_var[n1][n2][i] = random_var[n2][n1][i] = 1
            elif flag_0:
                random_var[n1][n2][i] = random_var[n2][n1][i] = 0
            else:
                random_var[n1][n2][i] = random_var[n2][n1][i] = random_pick(float(r_value[i][j]))
            cnt_r = cnt_r - random_var[n2][n1][i]
            temp_r_value[i][j] = random_var[n2][n1][i]

    temp_throught = 0.0
    temp_f_throught = 0.0
    temp_average_capacity_utilization = 0
    temp_x_value = []

    # 进行资源分配
    for i in range(request_num):
        for j in range(path_len[i]):
            path_tot = 0
            n1 = int(output_path[f'request{i}'][j])
            n2 = int(output_path[f'request{i}'][j + 1])
            for z in range(request_num):
                if random_var[n1][n2][z] == -1:
                    continue
                if random_var[n1][n2][z] == 0:
                    path_tot = path_tot + x_value[z]
                else:
                    path_tot = path_tot + (random_var[n1][n2][z] + 1) * x_value[z]
            m.Equation(path_tot <= channel[n1][n2])

    # LP 问题求解
    m.Maximize(m.sum([ri for ri in x_value]))
    m.solve(disp=False)
    diff = 0.0
    for i in range(request_num):
        temp_x_value.append(float(x_value[i].value[0]))
        diff = diff + temp_x_value[i] - np.floor(temp_x_value[i])
        temp_x_value[i] = np.floor(temp_x_value[i])
    # Rounding策略
    for ite in range(int(np.floor(diff))):
        for i in range(request_num):
            temp_x_value[i] = temp_x_value[i] + 1
            check_flag = True
            for j in range(path_len[i]):
                n1 = int(output_path[f'request{i}'][j])
                n2 = int(output_path[f'request{i}'][j + 1])
                path_tot = 0
                for z in range(request_num):
                    if random_var[n1][n2][z] == -1:
                        continue
                    if random_var[n1][n2][z] == 0:
                        path_tot = path_tot + temp_x_value[z]
                    else:
                        path_tot = path_tot + (random_var[n1][n2][z] + 1) * temp_x_value[z]
                if path_tot > channel[n1][n2]:
                    check_flag = False
                    break
            if not check_flag:
                temp_x_value[i] = temp_x_value[i] - 1
            else:
                break

    # for i in range(request_num):
    #     x = f"request{i}"
    #     min_path_flow = math.inf
    #     for j in range(path_len[i]):
    #         n1 = int(output_path[f"request{i}"][j])
    #         n2 = int(output_path[f"request{i}"][j + 1])
    #         min_path_flow = min(min_path_flow, np.floor(channel[n1][n2] / (random_var[n1][n2][i] + 1)))
    #     for j in range(path_len[i]):
    #         n1 = int(output_path[f"request{i}"][j])
    #         n2 = int(output_path[f"request{i}"][j + 1])
    #         channel[n1][n2] = channel[n1][n2] - (random_var[n1][n2][i] + 1) * min_path_flow
    #     temp_x_value.append(min_path_flow)

    temp_throught = np.sum(temp_x_value)

    for i in range(request_num):
        r_cnt = 0
        for j in range(path_len[i]):
            n1 = int(output_path[f"request{i}"][j])
            n2 = int(output_path[f"request{i}"][j + 1])
            if random_var[n1][n2][i] == 1:
                r_cnt = r_cnt + random_var[n1][n2][i]
        if math.pow(new_F, r_cnt) * math.pow(init_fidelity, path_len[i] - r_cnt) >= F_th[i]:
            temp_f_throught = temp_f_throught + temp_x_value[i]

    for i in range(request_num):
        for j in range(path_len[i]):
            n1 = int(output_path[f"request{i}"][j])
            n2 = int(output_path[f"request{i}"][j + 1])
            if random_var[n1][n2][i] == 1:
                use_channel[n1][n2] = use_channel[n1][n2] + (random_var[n1][n2][i]) * temp_x_value[i]
            use_channel[n2][n1] = use_channel[n1][n2]
    sum_len = 0
    tot = 0
    for i in range(request_num):
        sum_len = sum_len + path_len[i]
        for j in range(path_len[i]):
            n1 = int(output_path[f"request{i}"][j])
            n2 = int(output_path[f"request{i}"][j + 1])
            if (use_channel[n1][n2] + temp_x_value[i]) > channel[n1][n2]:
                print("!!!!!!!!!!!!!")
                print("!!!!!!!!!!!!!")
                print("!!!!!!!!!!!!!")

            tot = tot + temp_x_value[i]
    temp_average_capacity_utilization = tot
    if temp_f_throught >= result_f_throught:
        result_r_value = copy.deepcopy(temp_r_value)
        result_x_value = copy.deepcopy(temp_x_value)
        result_f_throught = temp_f_throught
        result_throught = temp_throught
        result_average_capacity_utilization = temp_average_capacity_utilization
    cnt = cnt + 1
result_throught = 0.0
result_f_throught = 0.0
result_average_capacity_utilization = 0.0
result_cnt = 0
capacity_sum = 0
capacity_utilization = 0.0
for i in range(request_num):
    # result_average_capacity_utilization = result_average_capacity_utilization - result_x_value[i]
    r_cnt = 0
    capacity_tot = 0.0
    for j in range(path_len[i]):
        if result_r_value[i][j] == 1:
            r_cnt = r_cnt + result_r_value[i][j]
        # cnt = cnt + 1
    capacity_tot = capacity_tot + r_cnt * result_x_value[i]
    capacity_sum = capacity_sum + r_cnt * result_x_value[i]
    result_x_value[i] = np.floor(np.dot(result_x_value[i], np.power(p_in, path_len[i] - 1)))
    result_throught = result_throught + result_x_value[i]
    if math.pow(new_F, r_cnt) * math.pow(init_fidelity, path_len[i] - r_cnt) >= F_th[i]:
        result_f_throught = result_f_throught + result_x_value[i]
        if result_x_value[i] > 0:
            result_cnt = result_cnt + 1
            result_average_capacity_utilization = result_average_capacity_utilization + capacity_tot / result_x_value[i]
capacity_utilization = result_average_capacity_utilization
if result_cnt != 0:
    result_average_capacity_utilization = result_average_capacity_utilization / result_cnt
endtime = time.time()
print(endtime - starttime)
with open('response_v1/node_failure_probability/result_0.30_Heuristic.txt', mode='a') as f_forth:
    f_forth.write('Weighted Throughput:' + str(round(float(result_throught), 2)))
    f_forth.write('\n')
    f_forth.write('Throught which >= F_th: ' + str(round(float(result_f_throught), 2)))
    f_forth.write('\n')
    f_forth.write('Capacity Sum:' + str(round(capacity_sum, 2)))
    f_forth.write('\n')
    f_forth.write('Capacity Utilization:' + str(round(capacity_utilization, 2)))
    f_forth.write('\n')
    f_forth.write('Average Capacity Utilization:' + str(round(result_average_capacity_utilization, 2)))
    f_forth.write('\n')
    f_forth.write('##########################################################################################')
    f_forth.write('\n')
