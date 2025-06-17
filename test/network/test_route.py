from qns.network.network import QuantumNetwork
from qns.network.topology import RandomTopology
from qns.network.topology.topo import ClassicTopology


def test_route_algorithm():
    topo = RandomTopology(nodes_number=20, lines_number=30)
    net = QuantumNetwork(topo=topo, classic_topo=ClassicTopology.All)

    net.build_route()
    print(net.nodes)
    for link in net.qchannels:
        print(link, link.node_list)


def test_request():
    topo = RandomTopology(nodes_number=20, lines_number=30)
    net = QuantumNetwork(topo=topo, classic_topo=ClassicTopology.Follow)
    n1 = net.get_node("n1")
    n4 = net.get_node("n4")
    print(net.route.route_table)
    print(net.query_route(n1, n4))

    net.add_request(n1, n4)
    print(net.requests)

    net.random_requests(5, allow_overlay=False)
    print(net.requests)

    net.random_requests(5, allow_overlay=True)
    print(net.requests)
