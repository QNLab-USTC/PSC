Classic channel: the link to transmit classic packets
========================================================

Classic channels can transmit classic packets (``ClassicPacket``) from a node to another node.
It has the following attributions:

- ``name``: the channel's name.
- ``length``: the physcial length of the channel. Default length is 0
- ``delay``: the propagation delay. The time delay from sending to receiving. Default delay is 0s.
- ``drop_rate``: the probability of losing the packets. Default drop rate is 0.
- ``bandwidth``: The bandwidth is the maximum sending bytes per second. If the ``bandwidth`` is reached, further packets will be put into a buffer (and causes a buffer delay). Default bandwidth is ``None`` (infinite).
-  ``max_buffer_size``: the maximum send buffer size in bytes. If the buffer is full, further packets will be dropped. Default buffer size is ``None`` (infinite).

Since SimQN focus on the quantum network, it is reasonable to assume that the classic channel is reliable. No errors will occur during the transmitting or errors will be checked. The following codes shows how to generate a classic channel:

.. code-block:: python

    from qns.entity.node.node import QNode
    from qns.entity.cchannel.cchannel import ClassicChannel

    n2 = QNode("n2")
    n1 = QNode("n1")
    l1 = ClassicChannel(name="l1", bandwidth=3, delay=0.2, drop_rate=0.1, max_buffer_size=5)

    # add the cchannel
    n1.add_cchannel(l1)
    n2.add_cchannel(l1)

    # get_cchannel can return the classic channel by its destination
    assert(l1 == n1.get_cchannel(n2))

Classic packets
----------------------------

The classic packets is implemented in class `ClassicPacket`, with the following fields:

- ``src``: the sending node
- ``dest``: the destination node
- ``msg``: the sending message.

The message can be a ``str``, ``bytes`` and any python object that can be dumped to a ``json`` string. If the message is a ``str`` or ``bytes`` object, the message length equals to the string or bytes length. Otherwise, the object will be firstly dumped to a json string and the packet length is the json string's length.

``get`` method will recovery the python object if the message is dumped to json. Or, it will return the messages as it is.

.. code-block:: python

    from qns.entity.cchannel.cchannel import ClassicPacket

    packet1 = ClassicPacket(msg="hello,world")
    msg1 = packet1.get() # msg1 = "hello,world"

    packet2 = ClassicPacket(msg={"key": "value"})
    msg2 = packet2.get() # msg2 = {"key": "value"}

SimQN provides the automatic json dump and load process for classic messages so that users will not worry about the 
tedious details of encoding of packets. ``ClassicPacket`` can transmit python ``dict`` as packets, and it is enough to express control the instructions.

Send and receive classic packets
----------------------------------------

It is easy to send classic packets using ``send`` method of the classic channel. For the receiver, it will be noticed by an event called ``RecvClassicPacket``. This event has the following fields:

- ``t``: the receiving time
- ``cchannel``: the related classic channel
- ``packet``: the receiving classic packet
- ``dest``: the destination

This packet needs to be processed in the ``handle`` method of the receiving applications:

.. code-block:: python

    # the send application
    class SendApp(Application):
        def __init__(self, dest: QNode, cchannel: ClassicChannel, send_rate=1):
            super().__init__()
            self.dest = dest
            self.cchannel = cchannel
            self.send_rate = send_rate

        # initiate: generate the first send event
        def install(self, node: QNode, simulator: Simulator):
            super().install(node, simulator)

            # get start time
            t = simulator.ts
            event = func_to_event(t, self.send_packet)
            self._simulator.add_event(event)

        def send_packet(self):
            # generate a packet
            packet = ClassicPacket(msg="Hello,world", src = self.get_node(), dest = self.dest)

            # send the classic packet
            self.cchannel.send(packet = packet, next_hop=self.dest)

            # calculate the next sending time
            t = self._simulator.current_time + \
                self._simulator.time(sec=1 / self.send_rate)
            
            # insert the next send event to the simulator
            event = func_to_event(t, self.send_packet)
            self._simulator.add_event(event)


    # the receiving application
    class RecvApp(Application):
        def handle(self, node: QNode, event: Event):
            if isinstance(event, RecvClassicPacket):
                packet = event.packet
                cchannel = event.cchannel
                recv_time = event.t

                # get the packet message
                msg = packet.get()

                # handling the receiving packet
                # ...

    # generate quantum nodes
    n1 = QNode("n1")
    n2 = QNode("n2") # add the RecvApp

    # generate a classic channel
    l1 = ClassicChannel(name="l1")
    n1.add_cchannel(l1)
    n2.add_cchannel(l1)

    # add apps
    n1.add_apps(SendApp(dest = n2, cchannel = l1))
    n2.add_apps(RecvApp())

    # initiate the simulator 
    s = Simulator(0, 10, 10000) # from  0 to 10 seconds
    n1.install(s)
    n2.install(s)

    # run the simulation
    s.run()

Forward classic packets
---------------------------

It is common that classic packets needs to be forwarded to the destination. SimQN provides ``ClassicPacketForwardApp`` to forward all classic packets if the node is not the destination. The classic routing table is generated from any ``RouteImpl`` object. In most cases, the ``DijkstraRouteAlgorithm`` is good enough.

.. note::
    The ``ClassicPacketForwardApp`` must be added to nodes before other applications so that it will handle all incoming classic packets first.

Here is an example of using the routing and forwarding classic packets:

.. code:: Python

    from qns.entity.cchannel.cchannel import ClassicPacket
    from qns.network.network import QuantumNetwork
    from qns.network.protocol.classicforward import ClassicPacketForwardApp
    from qns.network.route.dijkstra import DijkstraRouteAlgorithm
    from qns.network.topology.linetopo import LineTopology
    from qns.network.topology.topo import ClassicTopology
    from qns.simulator.simulator import Simulator

    s = Simulator(0, 10, accuracy=10000000)

    # the quantum network topology generator
    topo = LineTopology(nodes_number=10,
                        qchannel_args={"delay": 0.1},
                        cchannel_args={"delay": 0.1})

    # ``ClassicTopology.Follow`` means that the classic topology will follow the quantum topology
    net = QuantumNetwork(topo=topo, classic_topo=ClassicTopology.Follow)

    # build quantum routing table
    net.build_route()

    # generate the classic routing module
    classic_route = DijkstraRouteAlgorithm(name="classic route")

    # build classic routing table
    classic_route.build(net.nodes, net.cchannels)

    print(classic_route.route_table)

    # install the classic packet forward app to all nodes
    for n in net.nodes:
        n.add_apps(ClassicPacketForwardApp(classic_route))
