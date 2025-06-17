import numpy as np
import math
from gekko import GEKKO


request_num = 0
nodes_number = 0
F_min = []
F = 0

path = {}
output_path = {}
path_len = []

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
        F_min.append(float(line[2]))
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
        # print(line)
    line = f_first.readline().replace('init_fidelity:', '')
    line = line.replace('\n', '')
    F = float(line)
    for key, value in path.items():
        path_len.append(len(value) - 1)

with open("temp_files/path_output.txt", mode='w') as f_second:
    for i in range(request_num):
        x = f'request{i}'
        # print(len(output_path[x]))
        for j in range(len(output_path[x]) - 1):
            f_second.write(f'r_{output_path[x][j]}_{output_path[x][j + 1]}_{i}')
            f_second.write(',')
        f_second.write('\n')

channel = {}
path_len = [0] * request_num
var = {}
for i in range(nodes_number + 1):
    var[i] = {}
    channel[i] = {}
    for j in range(nodes_number + 1):
        var[i][j] = {}
        for z in range(request_num):
            var[i][j][z] = -1

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
    while True:
        line = f_first.readline()
        if line.find('init_fidelity: ', 0, len(line) - 1) != -1:
            ss = line
            break
    ss = ss.replace('init_fidelity: ', '').strip('\n')
    F = float(ss)

with open('temp_files/path_output.txt', mode='r') as f_second:
    for i in range(request_num):
        line = f_second.readline().strip('\n').split(',')
        path_len[i] = int(len(line) - 1)

A = 1 - math.log(F * F + (1 - F) * (1 - F), F)
B = []
for i in range(len(path_len)):
    B.append(math.log(F_min[i], F) - path_len[i])
m = GEKKO(remote=False)
m.options.SOLVER = 1
# m.solver_options = ['minlp_maximum_iterations 10000']
x = m.Array(m.Var, request_num, value=1, lb=0)
r = []
with open('temp_files/path_output.txt', mode='r') as f_third:
    for i in range(request_num):
        line = f_third.readline().strip('\n').split(',')
        tot = 0
        for j in range(path_len[i]):
            ss = line[j].replace('r_', '').split('_')
            n1_idx = int(ss[0])
            n2_idx = int(ss[1])
            v = m.Var(value=1, lb=0, ub=1)
            r.append(v)
            var[n1_idx][n2_idx][i] = v
            var[n2_idx][n1_idx][i] = v
            tot = tot + v
        m.Equation(A * tot <= B[i])
with open('temp_files/path_output.txt', mode='r') as f_forth:
    for i in range(request_num):
        line = f_forth.readline().strip('\n').split(',')
        for j in range(path_len[i]):
            tot = 0
            ss = line[j].replace('r_', '').split('_')
            n1_idx = int(ss[0])
            n2_idx = int(ss[1])
            for z in range(request_num):
                # if isinstance(var[n1_idx][n2_idx][z], int) == False:
                tot = tot + (var[n1_idx][n2_idx][z] + 1) * x[z]
            m.Equation(tot <= channel[n1_idx][n2_idx])

m.Maximize(m.sum([ri for ri in x]))
flag = 1
try:
    m.solve()
except:
    flag = -1
real_flow = []
for i in range(request_num):
    real_flow.append(x[i].value[0])
    # real_flow[i] = np.floor(np.dot(real_flow[i], np.power(p_in, path_len[i] - 1)))
with open('temp_files/flag_heuristic.txt', mode='a') as f_forth:
    f_forth.write(str(flag))
    f_forth.write('\n')
# with open('temp_files/ratio/flags_60_heuristic.txt', mode='a') as f_fifth:
#     f_fifth.write(str(flag))
#     f_fifth.write('\n')
with open('temp_files/temp_heuristic.txt', mode='w') as f_result:
    cnt = 0
    for i in range(request_num):
        f_result.write(str(real_flow[i]))
        f_result.write(',')
    f_result.write('\n')
    for i in range(request_num):
        for j in range(path_len[i]):
            f_result.write(str(r[cnt].value[0]))
            f_result.write(',')
            cnt = cnt + 1
        f_result.write('\n')
