import copy
import random

from qns.entity.node.app import Application
from qns.entity.qchannel.qchannel import QuantumChannel
from qns.entity.node.node import QNode
from qns.network.network import QuantumNetwork
from typing import Dict, List, Optional, Tuple
from qns.network.topology import Topology
from qns.network.requests import Request
import statistics as stat
import numpy as np
import math
from qns.utils import get_randint, get_rand

lines = 0
tot_channel = []
nodes_number = 100
max_capacity = 100
init_fidelity = 0
request_num = 10  # 随机产生请求，产生的请求的数目为 request_num
# request_length = 5  # 设定随机产生请求的路径长度为 request_length
p_in = 0.9
p_out = 0.8
p_in_greedy = 0.9  # 纠缠交换的成功概率
p_out_greedy = 0.8  # 纠缠建立的成功概率
link_failure_probability = 0.0  # 0.05, 0.10, 0.15, 0.20, 0.25, 0.30
node_failure_probability = 0.30  # 0.05, 0.10, 0.15, 0.20, 0.25, 0.30
capacity_tot_greedy = 0
capacity_tot = 0
result_cnt = 0
result_cnt_greedy = 0

t_request_fidelity = []
t_request_source = []
t_request_destination = []

with open('temp_files/temp_fidelity.txt', mode='r+') as f_init:
    lines = f_init.readlines()
    line = lines[0].strip('\n')
    init_fidelity = float(line)
    del lines[0]
with open('temp_files/temp_fidelity.txt', mode='w') as f_init2:
    for i in lines:
        f_init2.write(i)

with open("response_v1/topo_information_response.txt", mode='r') as f_first:
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
        tot_channel.append([n1_idx, n2_idx, capacity])

    f_first.readline()
    for i in range(request_num):
        line = f_first.readline().replace(f'request{i}: ', '')
        line = line.replace('\n', '')
        line = line.replace('<', '')
        line = line.replace('<', '')
        line = line.replace('>', '')
        line = line.replace('node ', '')
        line = line.split('-')
        t_request_fidelity.append(float(line[2]))
        t_request_source.append(int(line[0].replace('n', '')))
        t_request_destination.append(int(line[1].replace('n', '')))
        line = f_first.readline().replace(f'request{i} path: ', '')
        # print(line)
    line = f_first.readline().replace('init_fidelity:', '')

class RandomTopoQuantumRouting(Topology):
    def __init__(self, nodes_number, lines_number: int, qchannel_capacity: int, init_fidelity: float,
                 nodes_apps: List[Application] = [],
                 qchannel_args: Dict = {}, cchannel_args: Dict = {},
                 memory_args: Optional[List[Dict]] = {}):
        """
        Args:
            nodes_number: the number of Qnodes
            lines_number: the number of lines (QuantumChannel)
        """
        super().__init__(nodes_number, nodes_apps, qchannel_args, cchannel_args, memory_args)
        self.lines_number = lines_number
        self.qchannel_capacity = qchannel_capacity
        self.init_fidelity = init_fidelity

        # self.size = int(math.sqrt(self.nodes_number))

    def build(self) -> Tuple[List[QNode], List[QuantumChannel]]:
        nl: List[QNode] = []
        ll: List[QuantumChannel] = []

        for i in range(self.nodes_number):
            n = QNode(f"n{i + 1}")
            nl.append(n)
        for k in tot_channel:
            n = nl[k[0] - 1]
            pn = nl[k[1] - 1]
            link = QuantumChannel(name=f"l_{k[0]}_{k[1]}", **self.qchannel_args)
            link.capacity = k[2]
            link.fidelity = self.init_fidelity
            link.request_ID_on_edge = []
            link.flow_on_edge = []
            link.traffic_on_edge = []
            link.use_capacity = 0.0
            ll.append(link)
            pn.add_qchannel(link)
            n.add_qchannel(link)

        self._add_apps(nl)
        self._add_memories(nl)
        return nl, ll

class StaticTopoQuantumRouting(Topology):
    def __init__(self, nodes_number, lines_number: int, qchannel_capacity: int, init_fidelity: float,
                 nodes_apps: List[Application] = [],
                 qchannel_args: Dict = {}, cchannel_args: Dict = {},
                 memory_args: Optional[List[Dict]] = {}):
        """
        Args:
            nodes_number: the number of Qnodes
            lines_number: the number of lines (QuantumChannel)
        """
        super().__init__(nodes_number, nodes_apps, qchannel_args, cchannel_args, memory_args)
        self.lines_number = lines_number
        self.qchannel_capacity = qchannel_capacity
        self.init_fidelity = init_fidelity

        # self.size = int(math.sqrt(self.nodes_number))

    def build(self) -> Tuple[List[QNode], List[QuantumChannel]]:
        nl: List[QNode] = []
        ll: List[QuantumChannel] = []

        for i in range(self.nodes_number):
            n = QNode(f"n{i + 1}")
            nl.append(n)
        for k in tot_channel:
            n = nl[k[0] - 1]
            pn = nl[k[1] - 1]
            link = QuantumChannel(name=f"l_{k[0]}_{k[1]}", **self.qchannel_args)
            link.capacity = k[2]
            link.fidelity = self.init_fidelity
            link.request_ID_on_edge = []
            link.flow_on_edge = []
            link.traffic_on_edge = []
            link.use_capacity = 0.0
            ll.append(link)
            pn.add_qchannel(link)
            n.add_qchannel(link)

        self._add_apps(nl)
        self._add_memories(nl)
        return nl, ll


# 产生随机拓扑并将链路的信息、节点信息、请求信息保存在topo_information.txt中
request = []
request_greedy = []
request_solver = []
file_path = 'temp_files/topo_information.txt'


topo = RandomTopoQuantumRouting(nodes_number=nodes_number, lines_number=int(np.ceil(nodes_number * 1.5)),
                                qchannel_capacity=max_capacity,
                                init_fidelity=init_fidelity)
routing_net = QuantumNetwork(topo)

topo = StaticTopoQuantumRouting(nodes_number=nodes_number, lines_number=int(np.ceil(nodes_number * 1.5)),
                                qchannel_capacity=max_capacity,
                                init_fidelity=init_fidelity)
greedy_net = QuantumNetwork(topo)

routing_net.build_route()
greedy_net.build_route()

# 随机 request_num 个请求
cnt = 0
while cnt < request_num:
    src_idx = t_request_source[cnt] - 1
    dst_idx = t_request_destination[cnt] - 1
    new_request = Request(src=routing_net.nodes[src_idx], dest=routing_net.nodes[dst_idx])
    new_request.fidelity_threshold = t_request_fidelity[cnt]
    request_greedy.append(Request(src=greedy_net.nodes[src_idx], dest=greedy_net.nodes[dst_idx]))
    request.append(Request(src=routing_net.nodes[src_idx], dest=routing_net.nodes[dst_idx]))

    request_greedy[cnt].fidelity_threshold = new_request.fidelity_threshold
    request[cnt].fidelity_threshold = new_request.fidelity_threshold
    cnt = cnt + 1
# 对于 SPAR 实验方案的复现
max_num_path = 20  # maximum number of path that can take place on one edge
k_max = 1  # 对于每个请求选择的路径的最大数目
weight = [1] * request_num  # 各个请求所占的权重
Fid_th = 0.925
alpha = 1
beta = 1
min_capacity = max_capacity
f_min = max(np.floor(max_capacity / max_num_path), 1)
all_real_flow = []
all_path_pool = []
weighted_flow_sum = 0
weighted_flow_min = 0
average_capacity_utilization = 0
variance_capacity_utilization = 0


def Weighted_Adaptive_Cancellation(total_cancellation, max_cancellation, modify_weight):
    global max_capacity, init_fidelity, request, request_num, routing_net, max_num_path, k_max, weight, Fid_th, alpha, beta, p_in, p_out, min_capacity, f_min, all_real_flow, all_path_pool, weighted_flow_sum, weighted_flow_min, average_capacity_utilization, variance_capacity_utilization
    cancellation = [None] * len(max_cancellation)

    remaining_cancellation = total_cancellation

    # Neglect zero max_cancellation

    count_inf = 0
    remaining_vector = [ii for ii in range(len(max_cancellation))]
    for i in range(len(max_cancellation)):
        if max_cancellation[i] == 0:
            cancellation[i] = 0
            modify_weight[i] = 0
            max_cancellation[i] = np.inf
            count_inf = count_inf + 1
            remaining_vector.remove(i)

    to_do_order = list(np.argsort(max_cancellation))

    for i in range(len(to_do_order) - count_inf):  # start from the smallest max_cancellation

        fraction = modify_weight[to_do_order[i]] * max_cancellation[to_do_order[i]] / sum(
            [modify_weight[j] * max_cancellation[j] for j in remaining_vector])
        if fraction < 0:
            print('Warning! Fraction of allocation < 0')
        temp_cancellation = np.ceil(remaining_cancellation * fraction)  # delta_{i,j}^{r}

        if temp_cancellation > max_cancellation[to_do_order[i]]:
            temp_cancellation = max_cancellation[to_do_order[i]]

        cancellation[to_do_order[i]] = temp_cancellation

        # update
        remaining_cancellation = remaining_cancellation - temp_cancellation
        if remaining_cancellation < 0:
            print('Warning! Remaining_cancellation < 0.')
            remaining_cancellation = 0
        remaining_vector.remove(to_do_order[i])
    while remaining_cancellation > 0:  # still some cancellation remains
        for i in range(len(to_do_order) - count_inf):

            if cancellation[to_do_order[i]] < max_cancellation[to_do_order[i]]:
                diff_cancel = min(max_cancellation[to_do_order[i]] - cancellation[to_do_order[i]],
                                  remaining_cancellation)
                cancellation[to_do_order[i]] = cancellation[to_do_order[i]] + diff_cancel
                remaining_cancellation = remaining_cancellation - diff_cancel

            if remaining_cancellation == 0:
                break

    if remaining_cancellation > 0:  # still some cancellation remains
        print('Warning! Remaining_cancellation = ' + str(remaining_cancellation) + '. Allocation not complete.')

    if None in cancellation:
        print('Warning! Adapative cancellation unsuccessful.')
    return cancellation


def Step1_Capacity_Initialization():
    global max_capacity, init_fidelity, request, request_num, routing_net, max_num_path, k_max, weight, Fid_th, alpha, beta, p_in, p_out, min_capacity, f_min, all_real_flow, all_path_pool, weighted_flow_sum, weighted_flow_min, average_capacity_utilization, variance_capacity_utilization
    # 如果链路的保真度低于 Fid_th 那么就会进行一轮纯化操作
    global min_capacity, f_min
    lenn = len(list(routing_net.qchannels))
    k = 0
    for i in range(lenn):
        edge_fid = routing_net.qchannels[k].fidelity
        if edge_fid < Fid_th:
            routing_net.qchannels[k].use_capacity = routing_net.qchannels[k].capacity - routing_net.qchannels[
                k].capacity // 2
            routing_net.qchannels[k].capacity = routing_net.qchannels[k].capacity // 2
            routing_net.qchannels[k].fidelity = edge_fid ** 2 / (edge_fid ** 2 + (1 - edge_fid) ** 2)
            # routing_net.qchannels[k].capacity = np.random.binomial(routing_net.qchannels[k].capacity, p_out)
        # if edge_fid >= Fid_th:

        # routing_net.qchannels[k].capacity = np.random.binomial(routing_net.qchannels[k].capacity, p_out)

        if routing_net.qchannels[k].capacity < max_num_path:
            routing_net.qchannels.remove(routing_net.qchannels[k])
            k = k - 1
        if routing_net.qchannels[k].capacity < min_capacity:
            min_capacity = routing_net.qchannels[k].capacity
        k = k + 1
    f_min = max(np.floor(min_capacity / max_num_path), 1)


def Step2_Path_Determination():
    global max_capacity, init_fidelity, request, request_num, routing_net, max_num_path, k_max, weight, Fid_th, alpha, beta, p_in, p_out, min_capacity, f_min, all_real_flow, all_path_pool, weighted_flow_sum, weighted_flow_min, average_capacity_utilization, variance_capacity_utilization

    count_request = 0
    for r in request:
        count_request = count_request + 1
        temp_request_s = r.src
        temp_request_t = r.dest
        if temp_request_s not in routing_net.nodes or temp_request_t not in routing_net.nodes:
            path_pool = []
            path_length = math.inf
            all_path_pool.append([path_pool, [path_length] * len(path_pool)])
            print('Warning! Request out of grid:' + str(r))
            continue
        if not routing_net.query_route(temp_request_s, temp_request_t):  # Failure: No path in between
            path_pool = []
            path_length = math.inf
            all_path_pool.append([path_pool, [path_length] * len(path_pool)])
            print('Warning! No Path exist for request:' + str(r))
            continue
        path_pool = routing_net.query_route(temp_request_s, temp_request_t)[0][2]

        for ll in range(len(path_pool) - 1):  # each edge in this path
            n1 = path_pool[ll]
            n2 = path_pool[ll + 1]
            current_qchannel = n1.get_qchannel(n2)
            temp = list(current_qchannel.request_ID_on_edge)
            temp.append(
                [count_request, 1, len(path_pool) - 1, ll])  # append(request,path,path_length,order in the path)
            current_qchannel.request_ID_on_edge = temp

        all_path_pool.append([[path_pool], [len(path_pool) - 1]])


def Step3_Capacity_Allocation_PU(Step3_output):
    global max_capacity, init_fidelity, request, request_num, routing_net, max_num_path, k_max, weight, Fid_th, alpha, beta, p_in, p_out, min_capacity, f_min, all_real_flow, all_path_pool, weighted_flow_sum, weighted_flow_min, average_capacity_utilization, variance_capacity_utilization

    # P1 -- Initialize max_flow vector for each path
    max_flow_on_path = []  # = f^(r,j)_max

    for ii in range(len(all_path_pool)):  # for each request

        max_flow_on_path.append([])

        for j in range(len(all_path_pool[ii][0])):  # for each path
            temp_vec = [min_capacity] * (len(all_path_pool[ii][0][j]) - 1)
            max_flow_on_path[ii].append(temp_vec)

    # P2 -- Allocation
    count_no_improvement = 0  # iteration until no further change!

    head_count = 0  # number of iterations

    while count_no_improvement < len(routing_net.qchannels):
        k = routing_net.qchannels[head_count % len(routing_net.qchannels)]

        head_count = head_count + 1

        edge_capacity = k.capacity
        edge_request = k.request_ID_on_edge

        if not edge_request:  # Unused edge
            count_no_improvement = count_no_improvement + 1
            continue

        # Restrict the number of paths on every edge
        if len(edge_request) > max_num_path:
            num_cancel = len(edge_request) - max_num_path

            cancel_count = 0
            singular_request = []

            while cancel_count < num_cancel:  # recursive cancellation
                break_flag = 0

                for c in range(len(edge_request)):

                    if edge_request[c][0] in singular_request:
                        continue

                    if edge_request[c][2] == max(
                            [h[2] for h in [hh for hh in edge_request if hh[0] not in singular_request]]):

                        if [h[0] for h in edge_request].count(edge_request[c][0]) > 1:
                            edge_request.remove(
                                edge_request[c])  # change in node attribute 'request_ID_on_edge' as well
                            cancel_count = cancel_count + 1
                            break_flag = 1
                        else:
                            singular_request.append(edge_request[c][0])

                    if break_flag == 1:
                        break

        if len(edge_request) > max_num_path:  # Check: max path number realized
            print('Warning! Excessive Number of Path on edge ' + str(k))
            break

        total_max_flow = 0  # total max flow desired       --- edge specific
        request_max_flow = [0] * request_num  # max flow on each request     --- edge specific
        path_count = [0] * request_num  # path count on each request   --- edge specific
        max_flow_on_path_edge_specific = [[]] * request_num  # max flow on each path        --- edge specific [[],[]]

        for n in range(request_num):
            max_flow_on_path_edge_specific[n] = [0] * len(
                all_path_pool[n][0])  # All paths are counted. This is a sparse matrix.

        for ii in range(len(edge_request)):
            ind1 = edge_request[ii][0] - 1  # which request
            ind2 = edge_request[ii][1] - 1  # which path
            ind3 = edge_request[ii][3]  # which order

            total_max_flow = total_max_flow + max_flow_on_path[ind1][ind2][ind3]
            request_max_flow[ind1] = request_max_flow[ind1] + max_flow_on_path[ind1][ind2][ind3]
            path_count[ind1] = path_count[ind1] + 1
            max_flow_on_path_edge_specific[ind1][ind2] = max_flow_on_path[ind1][ind2][ind3]
        # Total cancellation

        total_cancellation = max(total_max_flow - edge_capacity, 0)  # delta_{i,j}
        # Stage 1 -- cancellation allocation on each request

        max_cancellation_request = [request_max_flow[tt] - f_min * path_count[tt] for tt in
                                    range(request_num)]  # delta_{i,j}^{r,max}
        weight_request = [tt ** -1 for tt in weight]  # multiply weight^-1 = divide weight

        for nn in range(len(max_cancellation_request)):  # non-used request, weight set to 0
            if max_cancellation_request[nn] < 0:
                weight_request[nn] = 0
                max_cancellation_request[nn] = 0

        # prime function

        # delta_{i,j}^{r}
        cancellation_on_request = Weighted_Adaptive_Cancellation(total_cancellation,
                                                                 list(max_cancellation_request),
                                                                 list(weight_request))

        if sum(n < 0 for n in cancellation_on_request) > 0:
            print('Warning! Negative concellation on request ' + str(ind1 + 1))

        if sum(cancellation_on_request) != total_cancellation:
            print('Warning! Sum of concellation allocation not match (Stage 1).')
            print('Allocated Cancellation:' + str(cancellation_on_request))
            print('Total Cancellation:' + str(total_cancellation))

        # Stage 2 -- cancellation allocation on each path

        cancellation_on_path = []
        for n in range(request_num):

            max_cancellation_path = [tt - f_min for tt in max_flow_on_path_edge_specific[n]]

            weight_path = [tt ** (alpha * -1) for tt in all_path_pool[n][1]]

            for nn in range(len(max_cancellation_path)):  # un-used path, weight set to 0, max cancellation set to 0
                if max_cancellation_path[nn] < 0:
                    weight_path[nn] = 0
                    max_cancellation_path[nn] = 0

            # prime function

            # delta_{i,j}^{r,l,max}
            temp_allocation = Weighted_Adaptive_Cancellation(cancellation_on_request[n], list(max_cancellation_path),
                                                             list(weight_path))
            cancellation_on_path.append(temp_allocation)

            if sum(temp_allocation) != cancellation_on_request[n]:
                print('Warning! Sum of concellation allocation not match (Stage 2).')
                print('Allocated Cancellation:' + str(temp_allocation))  # -----------
                print('Total Cancellation:' + str(cancellation_on_request[n]))  # -----------

        # Stage 3 -- Make deduction

        flow_on_edge_raw = []
        for ii in range(len(edge_request)):

            ind1 = edge_request[ii][0] - 1  # which request
            ind2 = edge_request[ii][1] - 1  # which path
            ind3 = edge_request[ii][3]  # which order

            if cancellation_on_path[ind1][ind2] < 0:
                print('Warning! Negative concellation on request ' + str(ind1 + 1) + ' path ' + str(ind2 + 1))

            temp_flow = max_flow_on_path[ind1][ind2][ind3] - cancellation_on_path[ind1][ind2]

            flow_on_edge_raw.append(int(temp_flow))

            # Update max_flow vector (only if further narrowing is needed)

            if temp_flow != max_flow_on_path[ind1][ind2][ind3]:
                max_flow_on_path[ind1][ind2][ind3] = temp_flow

                if ind3 < len(max_flow_on_path[ind1][ind2]) - 1:  # not last edge in the path
                    count_no_improvement = 0  # reset

            for ii in range(
                    len(max_flow_on_path[ind1][
                            ind2])):  # max flow information updates on EVERY edge in the path
                max_flow_on_path[ind1][ind2][ii] = min(temp_flow, max_flow_on_path[ind1][ind2][ii])

        # Allocation complete

        if len(flow_on_edge_raw) != len(edge_request):
            print('Warning! Edge' + str(k) + ' has unallocated request.')

        if flow_on_edge_raw != k.flow_on_edge and ind3 < len(
                max_flow_on_path[ind1][ind2]) - 1:  # not the last edge

            k.flow_on_edge = flow_on_edge_raw

        else:  # no improvement or last edge

            k.flow_on_edge = flow_on_edge_raw
            count_no_improvement = count_no_improvement + 1

        if sum(flow_on_edge_raw) > k.capacity:
            print('Warning! Edge' + str(k) + ' exceeds capacity.')

        if Step3_output == 1:
            print('Edge' + str(k) + '---' + str(flow_on_edge_raw))


def Step4_Routing_Performance(Step4_output):
    global max_capacity, init_fidelity, request, request_num, routing_net, max_num_path, k_max, weight, Fid_th, alpha, beta, p_in, p_out, min_capacity, f_min, all_real_flow, all_path_pool, weighted_flow_sum, weighted_flow_min, average_capacity_utilization, variance_capacity_utilization

    flow_prob = []  # unweighted flow rate, with p_in incorporated

    all_real_flow = []  # record all realized flow

    for i in range(request_num):  # for each request

        flow_path = []
        num_path = len(all_path_pool[i][0])

        for j in range(num_path):  # for each path

            list_edge = all_path_pool[i][0][j]  # 第 i 个请求的第 j 条路

            min_flow = max_capacity

            flag_path_cancellation = 0

            # specific path
            for k in range(len(list_edge) - 1):  # for each edge

                # find the index
                # list_edge不是请求i的
                n1 = list_edge[k]
                n2 = list_edge[k + 1]
                current_qchannel = n1.get_qchannel(n2)
                if [i + 1, j + 1] not in [w[:2] for w in current_qchannel.request_ID_on_edge]:
                    flag_path_cancellation = 1
                    break

                indd = [w[:2] for w in current_qchannel.request_ID_on_edge].index([i + 1, j + 1])

                if indd > len(current_qchannel.flow_on_edge):
                    print('Warning! Index does not match on edge: ' + str(list_edge[k]) + ',' + str(list_edge[k + 1]))

                if not current_qchannel.flow_on_edge:
                    print('Warning! Flow allocation incomplete on edge: ' + str(list_edge[k]) + ',' + str(
                        list_edge[k + 1]))
                # print(len(current_qchannel.flow_on_edge), current_qchannel.capacity, indd)
                temp = current_qchannel.flow_on_edge[indd]

                if temp < min_flow:
                    min_flow = temp

            if min_flow == max_capacity and flag_path_cancellation != 1:
                print('Warning! Min flow = Max flow on request ' + str(i + 1) + ', path ' + str(
                    j + 1) + ', check if error.')

            if flag_path_cancellation == 1:
                min_flow = 0

            flow_path.append(int(min_flow))

        path_length = all_path_pool[i][1]

        if Step4_output == 1:
            print('Request #' + str(i + 1) + '----' + str(request[i].src) + ' to ' + str(request[i].dest))
            print('Weight:' + str(weight[i]))
            print('Number of paths:' + str(num_path))
            print('Flow of each path:' + str(flow_path))
            print('Length of each path:' + str(path_length))
            print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')

        flow_prob.append(np.floor(np.dot(flow_path, list(np.power(p_in, [x - 1 for x in path_length])))))
        # flow_path[i] = flow_prob[i]
        all_real_flow.append([[np.floor(flow_prob[i])], path_length])  # [[flow_path, path_length],[ ] ]
        # all_real_flow.append([flow_path, path_length])  # [[flow_path, path_length],[ ] ]

    # objective function #1:
    # weighted_flow_sum = np.dot(weight, flow_prob)  # weighted sum flow
    weighted_flow_sum = np.floor(np.dot(weight, flow_prob))  # weighted sum flow

    # objective function #2:
    weighted_flow_min = min(np.multiply(weight, flow_prob))  # weighted min flow


def Step5_Capacity_Utilization(Step5_output):
    global max_capacity, init_fidelity, request, request_num, routing_net, max_num_path, k_max, weight, Fid_th, alpha, beta, p_in, p_out, min_capacity, f_min, all_real_flow, all_path_pool, weighted_flow_sum, weighted_flow_min, average_capacity_utilization, variance_capacity_utilization

    sum_capacity = 0
    count_used_edge = 0
    tot = 0

    for k in list(routing_net.qchannels):

        edge_capacity = k.capacity
        edge_request = k.request_ID_on_edge
        if not edge_request:
            continue

        real_capacity = 0
        for i in range(len(edge_request)):

            ind1 = k.request_ID_on_edge[i][0] - 1  # which request
            ind2 = k.request_ID_on_edge[i][1] - 1  # which path
            ind3 = k.request_ID_on_edge[i][2]  # path length

            if all_real_flow[ind1][1][ind2] != ind3:  # Check: path length does not match
                print('Warning! Path length does not match.')
                break

            real_capacity = real_capacity + all_real_flow[ind1][0][ind2]

        if Step5_output == 1:
            print('Edge--' + str(k) + '--' + str(round(real_capacity / edge_capacity * 100, 2)) + '%')

        if real_capacity / edge_capacity > 1:  # Check: capacity overflow
            print('Warning! Capacity Utilization > 1.')

        k.traffic_on_edge = round(real_capacity / edge_capacity * 100, 3)

        count_used_edge = count_used_edge + 1
        tot = tot + k.use_capacity

    average_capacity_utilization = tot
    list_utilization = [k.traffic_on_edge for k in routing_net.qchannels]
    list_utilization = list(filter(None, list_utilization))

    if len(list_utilization) > 1:
        variance_capacity_utilization = stat.variance(list_utilization) / 10000
    else:
        variance_capacity_utilization = 0


def Step6_Results():
    global max_capacity, init_fidelity, request, request_num, routing_net, max_num_path, k_max, weight, Fid_th, alpha, beta, p_in, p_out, min_capacity, f_min, all_real_flow, all_path_pool, weighted_flow_sum, weighted_flow_min, average_capacity_utilization, variance_capacity_utilization

    choice = 'PU'

    print('[Performance Report] -- ' + choice)
    print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
    print('Average Capacity Utilization:' + str(round(average_capacity_utilization * 100, 2)) + '%')
    print('Variance in Capacity Utilization:' + str(round(variance_capacity_utilization, 3)))
    print('Weighted Throughput:' + str(round(weighted_flow_sum, 2)))
    print('Weighted Min Throughput:' + str(round(weighted_flow_min, 2)))
    # print('Average Path Length:' + str(round(settings.ave_path_length, 2)))
    # print('Average Flow Variance:' + str(round(settings.ave_var_flow * 100, 2)) + '%')
    # print('Fairness over paths:' + str(round(settings.Jain_path, 2)))
    # print('Fairness over requests:' + str(round(settings.Jain_request, 2)))
    print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')

Step1_Capacity_Initialization()
print('[Minimum Capacity]')
print(str(min_capacity))
print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')

# for k in list(routing_net.qchannels):
#     print(k, k.capacity)
Step2_Path_Determination()
Step3_Capacity_Allocation_PU(Step3_output=0)
# for k in list(routing_net.qchannels):
#     print(k, k.flow_on_edge)
Step4_Routing_Performance(Step4_output=0)
Step5_Capacity_Utilization(Step5_output=0)
Step6_Results()
Throughput = 0
average_capacity_utilization = 0
normal_channel = []
result_cnt = 0
capacity_utilization = 0
capacity_sum = 0
for i in range(request_num):
    link_fidelity = 1.0
    capacity_tot = 0
    for j in range(len(all_path_pool[i][0][0]) - 1):
        n1 = all_path_pool[i][0][0][j]
        n2 = all_path_pool[i][0][0][j + 1]
        current_qchannel = n1.get_qchannel(n2)
        if current_qchannel not in normal_channel:
            capacity_sum = capacity_sum + current_qchannel.use_capacity
            normal_channel.append(current_qchannel)
            capacity_tot = capacity_tot + current_qchannel.use_capacity
        link_fidelity = link_fidelity * current_qchannel.fidelity
    if (link_fidelity >= request[i].fidelity_threshold):
        if all_real_flow[i][0][0] > 0:
            result_cnt = result_cnt + 1
            average_capacity_utilization = average_capacity_utilization + capacity_tot / all_real_flow[i][0][0]
        Throughput = Throughput + all_real_flow[i][0][0]
capacity_utilization = average_capacity_utilization
if result_cnt != 0:
    average_capacity_utilization = average_capacity_utilization / result_cnt

# 考虑链路失败概率
for k in routing_net.qchannels:
    key = get_rand()
    if key < link_failure_probability:
        k.capacity = 0
# 考虑节点失败概率
for i in routing_net.nodes:
    key = get_rand()
    if key < node_failure_probability:
        for k in i.qchannels:
            k.capacity = 0

for i in range(request_num):
    link_fidelity = 1.0
    capacity_tot = 0
    temp_flag = False
    for j in range(len(all_path_pool[i][0][0]) - 1):
        n1 = all_path_pool[i][0][0][j]
        n2 = all_path_pool[i][0][0][j + 1]
        current_qchannel = n1.get_qchannel(n2)
        if current_qchannel.capacity == 0:
            temp_flag = True
        link_fidelity = link_fidelity * current_qchannel.fidelity
    if (link_fidelity >= request[i].fidelity_threshold):
        if all_real_flow[i][0][0] > 0:
            result_cnt = result_cnt + 1
            average_capacity_utilization = average_capacity_utilization + capacity_tot / all_real_flow[i][0][0]
        if temp_flag:
            Throughput = Throughput - all_real_flow[i][0][0]


with open("response_v1/node_failure_probability/result_0.30_Pu.txt", mode='a') as fff:
    fff.write('Weighted Throughput:' + str(round(weighted_flow_sum, 2)))
    fff.write('\n')
    fff.write('Throught which >= F_th: ' + str(Throughput))
    fff.write('\n')
    fff.write('Capacity Sum:' + str(round(capacity_sum, 2)))
    fff.write('\n')
    fff.write('Capacity Utilization:' + str(round(capacity_utilization, 2)))
    fff.write('\n')
    fff.write('Average Capacity Utilization:' + str(round(average_capacity_utilization, 2)))
    fff.write('\n')
    fff.write('##########################################################################################')
    fff.write('\n')
print('Throught which >= F_th: ' + str(Throughput))

# 对于贪心策略的实现

max_num_path_greedy = 20  # maximum number of path that can take place on one edge
k_max_greedy = 1  # 对于每个请求选择的路径的最大数目
weight_greedy = [1] * request_num  # 各个请求所占的权重
Fid_th_greedy = 1
alpha_greedy = 1
beta_greedy = 1
min_capacity_greedy = max_capacity
f_min_greedy = max(np.floor(max_capacity / max_num_path_greedy), 1)
all_real_flow_greedy = []
all_path_pool_greedy = []
weighted_flow_sum_greedy = 0
weighted_flow_min_greedy = 0
average_capacity_utilization_greedy = 0
variance_capacity_utilization_greedy = 0


def Weighted_Adaptive_Cancellation_greedy(total_cancellation, max_cancellation, modify_weight):
    global max_capacity, init_fidelity, request, request_num, greedy_net, max_num_path_greedy, k_max_greedy, weight_greedy, Fid_th_greedy, alpha_greedy, beta_greedy, p_in_greedy, p_out_greedy, min_capacity_greedy, f_min_greedy, all_real_flow_greedy, all_path_pool_greedy, weighted_flow_sum_greedy, weighted_flow_min_greedy, average_capacity_utilization_greedy, variance_capacity_utilization_greedy
    cancellation = [None] * len(max_cancellation)

    remaining_cancellation = total_cancellation

    # Neglect zero max_cancellation

    count_inf = 0
    remaining_vector = [ii for ii in range(len(max_cancellation))]
    for i in range(len(max_cancellation)):
        if max_cancellation[i] == 0:
            cancellation[i] = 0
            modify_weight[i] = 0
            max_cancellation[i] = np.inf
            count_inf = count_inf + 1
            remaining_vector.remove(i)

    to_do_order = list(np.argsort(max_cancellation))

    for i in range(len(to_do_order) - count_inf):  # start from the smallest max_cancellation

        fraction = modify_weight[to_do_order[i]] * max_cancellation[to_do_order[i]] / sum(
            [modify_weight[j] * max_cancellation[j] for j in remaining_vector])
        if fraction < 0:
            print('Warning! Fraction of allocation < 0')
        temp_cancellation = np.ceil(remaining_cancellation * fraction)  # delta_{i,j}^{r}

        if temp_cancellation > max_cancellation[to_do_order[i]]:
            temp_cancellation = max_cancellation[to_do_order[i]]

        cancellation[to_do_order[i]] = temp_cancellation

        # update
        remaining_cancellation = remaining_cancellation - temp_cancellation
        if remaining_cancellation < 0:
            print('Warning! Remaining_cancellation < 0.')
            remaining_cancellation = 0
        remaining_vector.remove(to_do_order[i])
    while remaining_cancellation > 0:  # still some cancellation remains
        for i in range(len(to_do_order) - count_inf):

            if cancellation[to_do_order[i]] < max_cancellation[to_do_order[i]]:
                diff_cancel = min(max_cancellation[to_do_order[i]] - cancellation[to_do_order[i]],
                                  remaining_cancellation)
                cancellation[to_do_order[i]] = cancellation[to_do_order[i]] + diff_cancel
                remaining_cancellation = remaining_cancellation - diff_cancel

            if remaining_cancellation == 0:
                break

    if remaining_cancellation > 0:  # still some cancellation remains
        print('Warning! Remaining_cancellation = ' + str(remaining_cancellation) + '. Allocation not complete.')

    if None in cancellation:
        print('Warning! Adapative cancellation unsuccessful.')
    return cancellation


def Step1_Capacity_Initialization_greedy():
    global max_capacity, init_fidelity, request_greedy, request_num, greedy_net, max_num_path_greedy, k_max_greedy, weight_greedy, Fid_th_greedy, alpha_greedy, beta_greedy, p_in_greedy, p_out_greedy, min_capacity_greedy, f_min_greedy, all_real_flow_greedy, all_path_pool_greedy, weighted_flow_sum_greedy, weighted_flow_min_greedy, average_capacity_utilization_greedy, variance_capacity_utilization_greedy
    # 如果链路的保真度低于 Fid_th 那么就会进行一轮纯化操作
    global min_capacity_greedy, f_min_greedy
    lenn = len(list(greedy_net.qchannels))
    k = 0
    for i in range(lenn):
        edge_fid = greedy_net.qchannels[k].fidelity
        if edge_fid < Fid_th_greedy:
            greedy_net.qchannels[k].use_capacity = greedy_net.qchannels[k].capacity - greedy_net.qchannels[
                k].capacity // 2
            greedy_net.qchannels[k].capacity = greedy_net.qchannels[k].capacity // 2
            greedy_net.qchannels[k].fidelity = edge_fid ** 2 / (edge_fid ** 2 + (1 - edge_fid) ** 2)
            # greedy_net.qchannels[k].capacity = np.random.binomial(greedy_net.qchannels[k].capacity, p_out_greedy)
        # if edge_fid >= Fid_th_greedy:

        # greedy_net.qchannels[k].capacity = np.random.binomial(greedy_net.qchannels[k].capacity, p_out_greedy)
        if greedy_net.qchannels[k].capacity < max_num_path_greedy:
            greedy_net.qchannels.remove(greedy_net.qchannels[k])
            k = k - 1
        if greedy_net.qchannels[k].capacity < min_capacity_greedy:
            min_capacity_greedy = greedy_net.qchannels[k].capacity
        k = k + 1
    f_min_greedy = max(np.floor(min_capacity_greedy / max_num_path_greedy), 1)


def Step2_Path_Determination_greedy():
    # global max_capacity, init_fidelity, request_greedy, request_num, greedy_net, max_num_path_greedy, k_max_greedy, weight_greedy, Fid_th_greedy, alpha_greedy, beta_greedy, p_in_greedy, p_out_greedy, min_capacity_greedy, f_min_greedy, all_real_flow_greedy, all_path_pool_greedy, weighted_flow_sum_greedy, weighted_flow_min_greedy, average_capacity_utilization_greedy, variance_capacity_utilization_greedy
    count_request = 0
    for r in request_greedy:
        count_request = count_request + 1
        temp_request_s = r.src
        temp_request_t = r.dest
        if temp_request_s not in greedy_net.nodes or temp_request_t not in greedy_net.nodes:
            path_pool = []
            path_length = math.inf
            all_path_pool_greedy.append([path_pool, [path_length] * len(path_pool)])
            print('Warning! Request out of grid:' + str(r))
            continue
        if not greedy_net.query_route(temp_request_s, temp_request_t):  # Failure: No path in between
            path_pool = []
            path_length = math.inf
            all_path_pool_greedy.append([path_pool, [path_length] * len(path_pool)])
            print('Warning! No Path exist for request:' + str(r))
            continue
        path_pool = greedy_net.query_route(temp_request_s, temp_request_t)[0][2]

        for ll in range(len(path_pool) - 1):  # each edge in this path
            n1 = path_pool[ll]
            n2 = path_pool[ll + 1]
            current_qchannel = n1.get_qchannel(n2)
            temp = list(current_qchannel.request_ID_on_edge)
            temp.append(
                [count_request, 1, len(path_pool) - 1, ll])  # append(request,path,path_length,order in the path)
            current_qchannel.request_ID_on_edge = temp

        all_path_pool_greedy.append([[path_pool], [len(path_pool) - 1]])


def Step3_Capacity_Allocation_PU_greedy(Step3_output):
    global max_capacity, init_fidelity, request_greedy, request_num, greedy_net, max_num_path_greedy, k_max_greedy, weight_greedy, Fid_th_greedy, alpha_greedy, beta_greedy, p_in_greedy, p_out_greedy, min_capacity_greedy, f_min_greedy, all_real_flow_greedy, all_path_pool_greedy, weighted_flow_sum_greedy, weighted_flow_min_greedy, average_capacity_utilization_greedy, variance_capacity_utilization_greedy

    # P1 -- Initialize max_flow vector for each path
    max_flow_on_path = []  # = f^(r,j)_max

    for ii in range(len(all_path_pool_greedy)):  # for each request

        max_flow_on_path.append([])

        for j in range(len(all_path_pool_greedy[ii][0])):  # for each path
            temp_vec = [min_capacity_greedy] * (len(all_path_pool_greedy[ii][0][j]) - 1)
            max_flow_on_path[ii].append(temp_vec)

    # P2 -- Allocation
    count_no_improvement = 0  # iteration until no further change!

    head_count = 0  # number of iterations

    while count_no_improvement < len(greedy_net.qchannels):
        k = greedy_net.qchannels[head_count % len(greedy_net.qchannels)]

        head_count = head_count + 1

        edge_capacity = k.capacity
        edge_request = k.request_ID_on_edge

        if not edge_request:  # Unused edge
            count_no_improvement = count_no_improvement + 1
            continue

        # Restrict the number of paths on every edge
        if len(edge_request) > max_num_path_greedy:
            num_cancel = len(edge_request) - max_num_path_greedy

            cancel_count = 0
            singular_request = []

            while cancel_count < num_cancel:  # recursive cancellation
                break_flag = 0

                for c in range(len(edge_request)):

                    if edge_request[c][0] in singular_request:
                        continue

                    if edge_request[c][2] == max(
                            [h[2] for h in [hh for hh in edge_request if hh[0] not in singular_request]]):

                        if [h[0] for h in edge_request].count(edge_request[c][0]) > 1:
                            edge_request.remove(
                                edge_request[c])  # change in node attribute 'request_ID_on_edge' as well
                            cancel_count = cancel_count + 1
                            break_flag = 1
                        else:
                            singular_request.append(edge_request[c][0])

                    if break_flag == 1:
                        break

        if len(edge_request) > max_num_path_greedy:  # Check: max path number realized
            print('Warning! Excessive Number of Path on edge ' + str(k))
            break

        total_max_flow = 0  # total max flow desired       --- edge specific
        request_max_flow = [0] * request_num  # max flow on each request     --- edge specific
        path_count = [0] * request_num  # path count on each request   --- edge specific
        max_flow_on_path_edge_specific = [[]] * request_num  # max flow on each path        --- edge specific [[],[]]

        for n in range(request_num):
            max_flow_on_path_edge_specific[n] = [0] * len(
                all_path_pool_greedy[n][0])  # All paths are counted. This is a sparse matrix.

        for ii in range(len(edge_request)):
            ind1 = edge_request[ii][0] - 1  # which request
            ind2 = edge_request[ii][1] - 1  # which path
            ind3 = edge_request[ii][3]  # which order

            total_max_flow = total_max_flow + max_flow_on_path[ind1][ind2][ind3]
            request_max_flow[ind1] = request_max_flow[ind1] + max_flow_on_path[ind1][ind2][ind3]
            path_count[ind1] = path_count[ind1] + 1
            max_flow_on_path_edge_specific[ind1][ind2] = max_flow_on_path[ind1][ind2][ind3]
        # Total cancellation

        total_cancellation = max(total_max_flow - edge_capacity, 0)  # delta_{i,j}
        # Stage 1 -- cancellation allocation on each request

        max_cancellation_request = [request_max_flow[tt] - f_min_greedy * path_count[tt] for tt in
                                    range(request_num)]  # delta_{i,j}^{r,max}
        weight_request = [tt ** -1 for tt in weight_greedy]  # multiply weight^-1 = divide weight

        for nn in range(len(max_cancellation_request)):  # non-used request, weight set to 0
            if max_cancellation_request[nn] < 0:
                weight_request[nn] = 0
                max_cancellation_request[nn] = 0

        # prime function

        # delta_{i,j}^{r}
        cancellation_on_request = Weighted_Adaptive_Cancellation_greedy(total_cancellation,
                                                                        list(max_cancellation_request),
                                                                        list(weight_request))

        if sum(n < 0 for n in cancellation_on_request) > 0:
            print('Warning! Negative concellation on request ' + str(ind1 + 1))

        if sum(cancellation_on_request) != total_cancellation:
            print('Warning! Sum of concellation allocation not match (Stage 1).')
            print('Allocated Cancellation:' + str(cancellation_on_request))
            print('Total Cancellation:' + str(total_cancellation))

        # Stage 2 -- cancellation allocation on each path

        cancellation_on_path = []
        for n in range(request_num):

            max_cancellation_path = [tt - f_min_greedy for tt in max_flow_on_path_edge_specific[n]]

            weight_path = [tt ** (alpha_greedy * -1) for tt in all_path_pool_greedy[n][1]]

            for nn in range(len(max_cancellation_path)):  # un-used path, weight set to 0, max cancellation set to 0
                if max_cancellation_path[nn] < 0:
                    weight_path[nn] = 0
                    max_cancellation_path[nn] = 0

            # prime function

            # delta_{i,j}^{r,l,max}
            temp_allocation = Weighted_Adaptive_Cancellation_greedy(cancellation_on_request[n],
                                                                    list(max_cancellation_path),
                                                                    list(weight_path))
            cancellation_on_path.append(temp_allocation)

            if sum(temp_allocation) != cancellation_on_request[n]:
                print('Warning! Sum of concellation allocation not match (Stage 2).')
                print('Allocated Cancellation:' + str(temp_allocation))  # -----------
                print('Total Cancellation:' + str(cancellation_on_request[n]))  # -----------

        # Stage 3 -- Make deduction

        flow_on_edge_raw = []
        for ii in range(len(edge_request)):

            ind1 = edge_request[ii][0] - 1  # which request
            ind2 = edge_request[ii][1] - 1  # which path
            ind3 = edge_request[ii][3]  # which order

            if cancellation_on_path[ind1][ind2] < 0:
                print('Warning! Negative concellation on request ' + str(ind1 + 1) + ' path ' + str(ind2 + 1))

            temp_flow = max_flow_on_path[ind1][ind2][ind3] - cancellation_on_path[ind1][ind2]

            flow_on_edge_raw.append(int(temp_flow))

            # Update max_flow vector (only if further narrowing is needed)

            if temp_flow != max_flow_on_path[ind1][ind2][ind3]:
                max_flow_on_path[ind1][ind2][ind3] = temp_flow

                if ind3 < len(max_flow_on_path[ind1][ind2]) - 1:  # not last edge in the path
                    count_no_improvement = 0  # reset

            for ii in range(
                    len(max_flow_on_path[ind1][
                            ind2])):  # max flow information updates on EVERY edge in the path
                max_flow_on_path[ind1][ind2][ii] = min(temp_flow, max_flow_on_path[ind1][ind2][ii])

        # Allocation complete

        if len(flow_on_edge_raw) != len(edge_request):
            print('Warning! Edge' + str(k) + ' has unallocated request.')

        if flow_on_edge_raw != k.flow_on_edge and ind3 < len(
                max_flow_on_path[ind1][ind2]) - 1:  # not the last edge

            k.flow_on_edge = flow_on_edge_raw

        else:  # no improvement or last edge

            k.flow_on_edge = flow_on_edge_raw
            count_no_improvement = count_no_improvement + 1

        if sum(flow_on_edge_raw) > k.capacity:
            print('Warning! Edge' + str(k) + ' exceeds capacity.')

        if Step3_output == 1:
            print('Edge' + str(k) + '---' + str(flow_on_edge_raw))


def Step4_Routing_Performance_greedy(Step4_output):
    global max_capacity, init_fidelity, request_greedy, request_num, greedy_net, max_num_path_greedy, k_max_greedy, weight_greedy, Fid_th_greedy, alpha_greedy, beta_greedy, p_in_greedy, p_out_greedy, min_capacity_greedy, f_min_greedy, all_real_flow_greedy, all_path_pool_greedy, weighted_flow_sum_greedy, weighted_flow_min_greedy, average_capacity_utilization_greedy, variance_capacity_utilization_greedy

    flow_prob = []  # unweighted flow rate, with p_in incorporated

    all_real_flow_greedy = []  # record all realized flow

    for i in range(request_num):  # for each request

        flow_path = []
        num_path = len(all_path_pool_greedy[i][0])

        for j in range(num_path):  # for each path

            list_edge = all_path_pool_greedy[i][0][j]  # 第 i 个请求的第 j 条路

            min_flow = max_capacity

            flag_path_cancellation = 0

            # specific path
            for k in range(len(list_edge) - 1):  # for each edge

                # find the index
                # list_edge不是请求i的
                n1 = list_edge[k]
                n2 = list_edge[k + 1]
                current_qchannel = n1.get_qchannel(n2)
                if [i + 1, j + 1] not in [w[:2] for w in current_qchannel.request_ID_on_edge]:
                    flag_path_cancellation = 1
                    break

                indd = [w[:2] for w in current_qchannel.request_ID_on_edge].index([i + 1, j + 1])

                if indd > len(current_qchannel.flow_on_edge):
                    print('Warning! Index does not match on edge: ' + str(list_edge[k]) + ',' + str(list_edge[k + 1]))

                if not current_qchannel.flow_on_edge:
                    print('Warning! Flow allocation incomplete on edge: ' + str(list_edge[k]) + ',' + str(
                        list_edge[k + 1]))

                temp = current_qchannel.flow_on_edge[indd]

                if temp < min_flow:
                    min_flow = temp

            if min_flow == max_capacity and flag_path_cancellation != 1:
                print('Warning! Min flow = Max flow on request ' + str(i + 1) + ', path ' + str(
                    j + 1) + ', check if error.')

            if flag_path_cancellation == 1:
                min_flow = 0

            flow_path.append(int(min_flow))

        path_length = all_path_pool_greedy[i][1]

        if Step4_output == 1:
            print('Request #' + str(i + 1) + '----' + str(request_greedy[i].src) + ' to ' + str(request_greedy[i].dest))
            print('Weight:' + str(weight_greedy[i]))
            print('Number of paths:' + str(num_path))
            print('Flow of each path:' + str(flow_path))
            print('Length of each path:' + str(path_length))
            print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')

        flow_prob.append(np.floor(np.dot(flow_path, list(np.power(p_in_greedy, [x - 1 for x in path_length])))))
        # flow_path[i] = flow_prob[i]
        all_real_flow_greedy.append([[np.floor(flow_prob[i])], path_length])  # [[flow_path, path_length],[ ] ]

        # all_real_flow_greedy.append([flow_path, path_length])  # [[flow_path, path_length],[ ] ]

    # objective function #1:
    # weighted_flow_sum_greedy = np.dot(weight_greedy, flow_prob)  # weighted sum flow
    weighted_flow_sum_greedy = np.floor(np.dot(weight, flow_prob))  # weighted sum flow

    # objective function #2:
    weighted_flow_min_greedy = min(np.multiply(weight_greedy, flow_prob))  # weighted min flow


def Step5_Capacity_Utilization_greedy(Step5_output):
    global max_capacity, init_fidelity, request_greedy, request_num, greedy_net, max_num_path_greedy, k_max_greedy, weight_greedy, Fid_th_greedy, alpha_greedy, beta_greedy, p_in_greedy, p_out_greedy, min_capacity_greedy, f_min_greedy, all_real_flow_greedy, all_path_pool_greedy, weighted_flow_sum_greedy, weighted_flow_min_greedy, average_capacity_utilization_greedy, variance_capacity_utilization_greedy

    sum_capacity = 0
    count_used_edge = 0
    tot = 0
    for k in list(greedy_net.qchannels):

        edge_capacity = k.capacity
        edge_request = k.request_ID_on_edge
        if not edge_request:
            continue

        real_capacity = 0
        for i in range(len(edge_request)):

            ind1 = k.request_ID_on_edge[i][0] - 1  # which request
            ind2 = k.request_ID_on_edge[i][1] - 1  # which path
            ind3 = k.request_ID_on_edge[i][2]  # path length

            if all_real_flow_greedy[ind1][1][ind2] != ind3:  # Check: path length does not match
                print('Warning! Path length does not match.')
                break

            real_capacity = real_capacity + all_real_flow_greedy[ind1][0][ind2]

        if Step5_output == 1:
            print('Edge--' + str(k) + '--' + str(round(real_capacity / edge_capacity * 100, 2)) + '%')

        if real_capacity / edge_capacity > 1:  # Check: capacity overflow
            print('Warning! Capacity Utilization > 1.')

        k.traffic_on_edge = round(real_capacity / edge_capacity * 100, 3)

        count_used_edge = count_used_edge + 1
        tot = tot + k.use_capacity

    average_capacity_utilization_greedy = tot
    list_utilization = [k.traffic_on_edge for k in greedy_net.qchannels]
    list_utilization = list(filter(None, list_utilization))

    if len(list_utilization) > 1:
        variance_capacity_utilization_greedy = stat.variance(list_utilization) / 10000
    else:
        variance_capacity_utilization_greedy = 0


def Step6_Results_greedy():
    global max_capacity, init_fidelity, request_greedy, request_num, greedy_net, max_num_path_greedy, k_max_greedy, weight_greedy, Fid_th_greedy, alpha_greedy, beta_greedy, p_in_greedy, p_out_greedy, min_capacity_greedy, f_min_greedy, all_real_flow_greedy, all_path_pool_greedy, weighted_flow_sum_greedy, weighted_flow_min_greedy, average_capacity_utilization_greedy, variance_capacity_utilization_greedy

    choice = 'PU'

    print('[Performance Report] -- ' + choice)
    print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
    print('Average Capacity Utilization:' + str(round(average_capacity_utilization_greedy * 100, 2)) + '%')
    print('Variance in Capacity Utilization:' + str(round(variance_capacity_utilization_greedy, 3)))
    print('Weighted Throughput:' + str(round(weighted_flow_sum_greedy, 2)))
    print('Weighted Min Throughput:' + str(round(weighted_flow_min_greedy, 2)))
    # print('Average Path Length:' + str(round(settings.ave_path_length, 2)))
    # print('Average Flow Variance:' + str(round(settings.ave_var_flow * 100, 2)) + '%')
    # print('Fairness over paths:' + str(round(settings.Jain_path, 2)))
    # print('Fairness over requests:' + str(round(settings.Jain_request, 2)))
    print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')


Step1_Capacity_Initialization_greedy()
print('[Minimum Capacity]')
print(str(min_capacity_greedy))
print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')

# for k in list(greedy_net.qchannels):
#     print(k, k.capacity)
# for k in list(greedy_net.qchannels):
#     print(k, k.flow_on_edge)
Step2_Path_Determination_greedy()
Step3_Capacity_Allocation_PU_greedy(Step3_output=0)
Step4_Routing_Performance_greedy(Step4_output=0)
Step5_Capacity_Utilization_greedy(Step5_output=0)
Step6_Results_greedy()

Throughput = 0
average_capacity_utilization_greedy = 0
greedy_channel = []
result_cnt_greedy = 0
capacity_utilization_greedy = 0
capacity_sum_greedy = 0
for i in range(request_num):
    link_fidelity = 1.0
    capacity_tot_greedy = 0
    for j in range(len(all_path_pool_greedy[i][0][0]) - 1):
        n1 = all_path_pool_greedy[i][0][0][j]
        n2 = all_path_pool_greedy[i][0][0][j + 1]
        current_qchannel = n1.get_qchannel(n2)
        if current_qchannel not in greedy_channel:
            capacity_sum_greedy = capacity_sum_greedy + current_qchannel.use_capacity
            capacity_tot_greedy = capacity_tot_greedy + current_qchannel.use_capacity
            greedy_channel.append(current_qchannel)
        link_fidelity = link_fidelity * current_qchannel.fidelity
    if link_fidelity >= request_greedy[i].fidelity_threshold:
        if all_real_flow_greedy[i][0][0] > 0:
            result_cnt_greedy = result_cnt_greedy + 1
            average_capacity_utilization_greedy = average_capacity_utilization_greedy + capacity_tot_greedy / all_real_flow_greedy[i][0][0]
        Throughput = Throughput + all_real_flow_greedy[i][0][0]
capacity_utilization_greedy = average_capacity_utilization_greedy
average_capacity_utilization_greedy = average_capacity_utilization_greedy / result_cnt_greedy

for i in range(len(routing_net.qchannels)):
    greedy_net.qchannels[i].capacity = routing_net.qchannels[i].capacity

for i in range(request_num):
    link_fidelity = 1.0
    capacity_tot_greedy = 0
    temp_flag = False
    for j in range(len(all_path_pool_greedy[i][0][0]) - 1):
        n1 = all_path_pool_greedy[i][0][0][j]
        n2 = all_path_pool_greedy[i][0][0][j + 1]
        current_qchannel = n1.get_qchannel(n2)
        if current_qchannel.capacity == 0:
            temp_flag = True
        link_fidelity = link_fidelity * current_qchannel.fidelity
    if link_fidelity >= request_greedy[i].fidelity_threshold:
        if all_real_flow_greedy[i][0][0] > 0:
            result_cnt_greedy = result_cnt_greedy + 1
            average_capacity_utilization_greedy = average_capacity_utilization_greedy + capacity_tot_greedy / all_real_flow_greedy[i][0][0]
        if temp_flag:
            Throughput = Throughput - all_real_flow_greedy[i][0][0]

with open("response_v1/node_failure_probability/result_0.30_Greedy.txt", mode='a') as ffff:
    ffff.write('Weighted Throughput:' + str(round(weighted_flow_sum_greedy, 2)))
    ffff.write('\n')
    ffff.write('Throught which >= F_th: ' + str(Throughput))
    ffff.write('\n')
    ffff.write('Capacity Sum:' + str(round(capacity_sum_greedy, 2)))
    ffff.write('\n')
    ffff.write('Capacity Utilization:' + str(round(capacity_utilization_greedy, 2)))
    ffff.write('\n')
    ffff.write('Average Capacity Utilization:' + str(round(average_capacity_utilization_greedy, 2)))
    ffff.write('\n')
    ffff.write('##########################################################################################')
    ffff.write('\n')
print('Throught which >= F_th: ' + str(Throughput))
print('init_fidelity: ' + str(init_fidelity))

with open(file_path, mode='w', encoding='utf-8') as f:
    f.write("nodes_list: {}".format(routing_net.nodes))
    f.write('\n')
    for i in range(len(routing_net.qchannels)):
        cur_channel = routing_net.qchannels[i]
        if cur_channel.capacity == 0:
            f.write("{} = {}".format(cur_channel.name, 0))
        else:
            f.write("{} = {}".format(cur_channel.name, tot_channel[i][2]))
        f.write('\n')
    f.write('request_list:\n')
    f.write(str(request_num))
    f.write('\n')
    for idx, i in enumerate(request):
        f.write("request{}: {}-{}-{}".format(idx, i.src, i.dest, i.fidelity_threshold))
        f.write('\n')
        result = routing_net.query_route(i.src, i.dest)
        f.write("request{} path: {}".format(idx, result[0][2]))
        f.write('\n')
    f.write("init_fidelity: {}".format(str(init_fidelity)))