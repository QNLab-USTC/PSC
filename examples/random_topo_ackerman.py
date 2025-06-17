from qns.entity.node.app import Application
from qns.entity.qchannel.qchannel import QuantumChannel
from qns.entity.node.node import QNode
from qns.network.network import QuantumNetwork
from typing import Dict, List, Optional, Tuple
from qns.network.topology import Topology
from qns.network.requests import Request
from qns.utils.random import get_randint


# class RandomTopoAckerman(Topology):
#     def __init__(self, nodes_number, lines_number: int, qchannel_capacity: int, nodes_apps: List[Application] = [],
#                  qchannel_args: Dict = {}, cchannel_args: Dict = {},
#                  memory_args: Optional[List[Dict]] = {}):
#         """
#         Args:
#             nodes_number: the number of Qnodes
#             lines_number: the number of lines (QuantumChannel)
#         """
#         super().__init__(nodes_number, nodes_apps, qchannel_args, cchannel_args, memory_args)
#         self.lines_number = lines_number
#         self.qchannel_capacity = qchannel_capacity
#
#     def build(self) -> Tuple[List[QNode], List[QuantumChannel]]:
#         nl: List[QNode] = []
#         ll: List[QuantumChannel] = []
#
#         mat = [[0 for i in range(self.nodes_number)] for j in range(self.nodes_number)]
#
#         if self.nodes_number >= 1:
#             n = QNode(f"n{1}")
#             nl.append(n)
#
#         for i in range(self.nodes_number - 1):
#             n = QNode(f"n{i + 2}")
#             nl.append(n)
#
#             idx = get_randint(0, i)
#             pn = nl[idx]
#             mat[idx][i + 1] = 1
#             mat[i + 1][idx] = 1
#
#             link = QuantumChannel(name=f"l_{idx + 1}_{i + 2}", **self.qchannel_args)
#             link.capacity = get_randint(1, self.qchannel_capacity)
#             ll.append(link)
#             pn.add_qchannel(link)
#             n.add_qchannel(link)
#
#         if self.lines_number > self.nodes_number - 1:
#             for i in range(self.nodes_number - 1, self.lines_number):
#                 while True:
#                     a = get_randint(0, self.nodes_number - 1)
#                     b = get_randint(0, self.nodes_number - 1)
#                     if mat[a][b] == 0:
#                         break
#                 mat[a][b] = 1
#                 mat[b][a] = 1
#                 n = nl[a]
#                 pn = nl[b]
#                 link = QuantumChannel(name=f"l_{a + 1}_{b + 1}", **self.qchannel_args)
#                 link.capacity = get_randint(1, self.qchannel_capacity)
#                 ll.append(link)
#                 pn.add_qchannel(link)
#                 n.add_qchannel(link)
#
#         self._add_apps(nl)
#         self._add_memories(nl)
#         return nl, ll
class RandomTopoAckerman(Topology):
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

    def build(self) -> Tuple[List[QNode], List[QuantumChannel]]:
        nl: List[QNode] = []
        ll: List[QuantumChannel] = []

        mat = [[0 for i in range(self.nodes_number)] for j in range(self.nodes_number)]

        if self.nodes_number >= 1:
            n = QNode(f"n{1}")
            nl.append(n)

        for i in range(self.nodes_number - 1):
            n = QNode(f"n{i + 2}")
            nl.append(n)

            idx = get_randint(0, i)
            pn = nl[idx]
            mat[idx][i + 1] = 1
            mat[i + 1][idx] = 1

            link = QuantumChannel(name=f"l_{idx + 1}_{i + 2}", **self.qchannel_args)
            link.capacity = get_randint(1, self.qchannel_capacity)
            link.fidelity = self.init_fidelity
            link.request_ID_on_edge = []
            link.flow_on_edge = []
            ll.append(link)
            pn.add_qchannel(link)
            n.add_qchannel(link)

        if self.lines_number > self.nodes_number - 1:
            for i in range(self.nodes_number - 1, self.lines_number):
                while True:
                    a = get_randint(0, self.nodes_number - 1)
                    b = get_randint(0, self.nodes_number - 1)
                    if mat[a][b] == 0:
                        break
                mat[a][b] = 1
                mat[b][a] = 1
                n = nl[a]
                pn = nl[b]
                link = QuantumChannel(name=f"l_{a + 1}_{b + 1}", **self.qchannel_args)
                link.capacity = get_randint(1, self.qchannel_capacity)
                link.fidelity = self.init_fidelity
                link.request_ID_on_edge = []
                link.flow_on_edge = []
                ll.append(link)
                pn.add_qchannel(link)
                n.add_qchannel(link)

        self._add_apps(nl)
        self._add_memories(nl)
        return nl, ll


file_path = '../ackerman_examples/temp_files/topo_information.txt'
topo = RandomTopoAckerman(nodes_number=3, lines_number=1, qchannel_capacity=10, init_fidelity=0.9)
net = QuantumNetwork(topo)
net.build_route()
# for i in range(len(list(net.qchannels))):
#     print(net.qchannels[i].capacity)
request = []
request.append(Request(src=net.nodes[0], dest=net.nodes[len(net.nodes)-1]))
request.append(Request(src=net.nodes[0], dest=net.nodes[len(net.nodes)-1]))
all_path_pool_greedy = []
print(net.query_route(request[0].src, request[0].dest)[0][2])
path_pool = net.query_route(request[0].src, request[0].dest)[0][2]
all_path_pool_greedy.append([[path_pool], [len(path_pool) - 1]])
path_pool = net.query_route(request[1].src, request[1].dest)[0][2]
all_path_pool_greedy.append([[path_pool], [len(path_pool) - 1]])
for j in range(len(all_path_pool_greedy[0][0])):
    for k in range(len(all_path_pool_greedy[0][0][j]) - 1):
        n1 = all_path_pool_greedy[0][0][j][k]
        n2 = all_path_pool_greedy[0][0][j][k + 1]
        current_qchannel = n1.get_qchannel(n2)
        print(current_qchannel)
print(all_path_pool_greedy[0][0][0])
n1 = path_pool[0]
print(net.query_route(request.src, request.dest)[0][2])
for i in range(len(path_pool)-1):
    n1 = path_pool[i]
    n2 = path_pool[i+1]
    current_qchannel = n1.get_qchannel(n2)
    temp = list(current_qchannel.request_ID_on_edge)
    temp.append(
        [1, 1, len(path_pool) - 1, i])  # append(request,path,path_length,order in the path)
    current_qchannel.request_ID_on_edge = temp
for i in range(len(list(net.qchannels))):
    print(net.qchannels[i].request_ID_on_edge)

# with open(file_path, mode='w', encoding='utf-8') as f:
#     topo = RandomTopoAckerman(nodes_number=20, lines_number=40, qchannel_capacity=10)
#     net = QuantumNetwork(topo)
#     f.write("nodes_list: {}".format(net.nodes))
#     f.write('\n')
#     for i in net.qchannels:
#         f.write("{} = {}".format(i.name, i.capacity))
#         f.write('\n')
#     net.build_route()
#     f.write('request_list:\n')
#     # random generate request
#     request_num = 2
#     f.write(str(request_num+1))
#     f.write('\n')
#     request = []
#     request.append(Request(src=net.nodes[0], dest=net.nodes[len(net.nodes)-1]))
#     for i in range(request_num):
#         src_idx = get_randint(0, len(net.nodes)-1)
#         dst_idx = get_randint(0, len(net.nodes)-1)
#         new_request = Request(src=net.nodes[src_idx], dest=net.nodes[dst_idx])
#         if src_idx == dst_idx or new_request in request:
#             i = i - 1
#             continue
#         request.append(Request(src=net.nodes[src_idx], dest=net.nodes[dst_idx]))
#     for idx, i in enumerate(request):
#         f.write("request{}: {}-{}".format(idx, i.src, i.dest))
#         f.write('\n')
#         result = net.query_route(i.src, i.dest)
#         f.write("request{} path: {}".format(idx, result[0][2]))
#         f.write('\n')
