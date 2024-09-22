import os
import pickle
import sys
from ipaddress import IPv4Network

import numpy as np
import torch
from networkx import spring_layout, DiGraph

sys.path.append(os.path.dirname(__file__) + "/../")

import networkx as nx
from scipy.spatial import Delaunay
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt

from semantics import Semantics
import topologies
from nutils import prop, choose_random

from predicate_semantics import *

"""
This file contains the protocol semantics for BGP/OSPF. Using the BgpSemantics class
one can generate a dataset of fact bases encoding BGP/OSPF synthesis input/output examples.

This file also includes the simulation code required to simulate BGP/OSPF.
"""


def random_planar_graph(num_nodes, random_state, MAX_INTERFACE=4):
    pos = random_state.rand(num_nodes, 2)
    simps = Delaunay(pos).simplices
    G = nx.DiGraph()

    for i in range(num_nodes):
        G.add_node(i)
    def add_edge(src, dst):
        if G.out_degree(src) < (MAX_INTERFACE-1) and G.out_degree(dst) < (MAX_INTERFACE-1):
            G.add_edge(src, dst)
            G.add_edge(dst, src)

    for tri in simps:
        add_edge(tri[0], tri[1])
        add_edge(tri[1], tri[2])
        add_edge(tri[2], tri[0])

    return G, pos


class BgpRoute(object):
    def __init__(self, destination, local_preference, as_path_length, origin_type, med, is_ebgp_learned, bgp_speaker_id,
                 next_hop, igp_costs=None):
        self.destination = destination
        self.local_preference = local_preference
        self.as_path_length = as_path_length
        self.origin_type = origin_type
        self.med = med
        self.is_ebgp_learned = is_ebgp_learned
        self.bgp_speaker_id = bgp_speaker_id
        self.next_hop = next_hop

        self.igp_costs = igp_costs

    def copy(self):
        return BgpRoute(self.destination, self.local_preference, self.as_path_length, self.origin_type,
                        self.med, self.is_ebgp_learned, self.bgp_speaker_id, self.next_hop, self.igp_costs)

    def __repr__(self):
        return f"<BgpRoute destination={self.destination} LOCAL_PREFERENCE=" + str(self.local_preference) + \
            " AS_PATH_LENGTH=" + str(self.as_path_length) + \
            " ORIGIN_TYPE=" + str(self.origin_type) + \
            " MED=" + str(self.med) + \
            " IS_EBGP_LEARNED=" + str(self.is_ebgp_learned) + \
            " BGP_SPEAKER_ID=" + str(self.bgp_speaker_id) + \
            " NEXT_HOP=" + str(self.next_hop) + ">"

    # hash and compare by speaker ID only (TODO: what if more than one announcement per speaker e.g. redistributing IGP)
    def __eq__(self, other):
        if other is None:
            return False
        return self.bgp_speaker_id == other.bgp_speaker_id and self.destination == other.destination

    def __hash__(self):  # 计算哈希时只考虑bgp_speaker_id和destination
        return 10000 * self.bgp_speaker_id + self.destination


BgpRoute.ORIGIN_IGP = 0
BgpRoute.ORIGIN_EGP = 1
BgpRoute.ORIGIN_INCOMPLETE = 2


class AclRole():
    def __init__(self, dst, node_id, out_or_in, dst_ipv4: IPv4Network = None):
        self.dst = dst
        self.node_id = node_id
        self.out_or_in = out_or_in  # 0 out, 1 in
        self.dst_ipv4 = dst_ipv4

    def __eq__(self, other):
        if other is None:
            return False
        return self.dst == other.dst and self.node_id == other.node_id and self.out_or_in == other.out_or_in

    def __hash__(self):
        res = 10000 * self.dst + self.node_id + self.out_or_in
        return int(res)


def generate_random_route_announcement(destination, LOCAL_PREF=None, AS_PATH_LENGTH=None, ORIGIN=None, MED=None,
                                       IS_EBGP_LEARNED=None):
    LOCAL_PREF = LOCAL_PREF if LOCAL_PREF is not None else np.random.randint(0, 100)
    # AS_PATH_LENGTH = AS_PATH_LENGTH if AS_PATH_LENGTH is not None else np.random.randint(1, 10)
    AS_PATH_LENGTH = AS_PATH_LENGTH if AS_PATH_LENGTH is not None else 1  # 改为固定值1
    ORIGIN = ORIGIN if ORIGIN is not None else choose_random(
        [BgpRoute.ORIGIN_IGP, BgpRoute.ORIGIN_EGP, BgpRoute.ORIGIN_INCOMPLETE])
    MED = MED if MED is not None else choose_random([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    IS_EBGP_LEARNED = choose_random([True, False]) if IS_EBGP_LEARNED is None else IS_EBGP_LEARNED

    return BgpRoute(destination, LOCAL_PREF, AS_PATH_LENGTH, ORIGIN, MED, IS_EBGP_LEARNED, 0, 0)


def lowest(routes, prop_fct):
    v = float("inf")
    for r in routes: v = min(v, prop_fct(r))
    return v


def generate_random_route_announcements(destination, ROUTES_PER_CATEGORY=2, ROUTES_IN_LAST_CATEGORY=3):
    # 可设置的参数有local_preference和med
    announcements = []
    for i in range(ROUTES_PER_CATEGORY):
        LP = np.random.randint(0, 10) * 10
        announcements.append(generate_random_route_announcement(destination, LOCAL_PREF=LP))
    LOCAL_PREF = -lowest(announcements, lambda r: -r.local_preference)  # 最小local_preference

    for i in range(ROUTES_PER_CATEGORY):
        # APL = np.random.randint(1, 10)
        APL = 1  # 改为固定值1
        announcements.append(generate_random_route_announcement(destination, LOCAL_PREF=LOCAL_PREF, AS_PATH_LENGTH=APL))
    AS_PATH_LENGTH = lowest(announcements, lambda r: r.as_path_length)

    for i in range(ROUTES_PER_CATEGORY):
        M = np.random.randint(0, 10)
        announcements.append(
            generate_random_route_announcement(destination, LOCAL_PREF=LOCAL_PREF, AS_PATH_LENGTH=AS_PATH_LENGTH,
                                               MED=M))
    MED = lowest(announcements, lambda r: r.med)

    for i in range(ROUTES_IN_LAST_CATEGORY):
        announcements.append(
            generate_random_route_announcement(destination, LOCAL_PREF=LOCAL_PREF, AS_PATH_LENGTH=AS_PATH_LENGTH,
                                               MED=MED))

    return announcements


def prop(e, prop):
    if prop not in e.keys(): return None
    return e[prop]


def generate_graph_with_topology(filename, seed, NUM_NODES=None, NUM_NETWORKS=None, NUM_GATEWAY_NODES=None,
                                 NUM_ROUTE_REFLECTORS=None, FULLY_MESHED=False):
    MAX_WEIGHT = 32

    s = np.random.RandomState(seed=seed)
    graph = topologies.read_topology(filename)
    NUM_NODES = len(graph.nodes())

    print("generate sample based on topology", filename)

    if NUM_NETWORKS is None: NUM_NETWORKS = 1
    if NUM_GATEWAY_NODES is None: NUM_GATEWAY_NODES = choose_random([2, 3, 7], s)
    if NUM_ROUTE_REFLECTORS is None: NUM_ROUTE_REFLECTORS = 2

    # set node types
    nx.set_node_attributes(graph, "router", name="type")

    # initialise link weights
    for src, tgt in graph.edges():
        weight = s.randint(1, MAX_WEIGHT)
        graph[src][tgt]["weight"] = weight
        graph[tgt][src]["weight"] = weight

    nx.set_edge_attributes(graph, "ospf", name="type")

    router_nodes = set(graph.nodes())

    # add network nodes
    network_nodes = set()
    routes_per_network = {}

    for n in range(NUM_NETWORKS):
        node_id = len(graph.nodes())
        network_nodes.add(node_id)
        routes_per_network[node_id] = generate_random_route_announcements(node_id)  # 只对外部网络做通告
        graph.add_node(node_id, type="network")

    # BGP

    ## configure gateway nodes
    ebgp_nodes = set()
    for network_node in network_nodes:
        for n in range(NUM_GATEWAY_NODES):
            ebgp_node = len(graph.nodes())
            gateway_node = s.randint(0, NUM_NODES)
            graph.add_node(ebgp_node, type="external")
            ebgp_nodes.add(ebgp_node)

            # connect gateway and ebgp node
            graph.add_edge(ebgp_node, gateway_node, type="ebgp")
            graph.add_edge(gateway_node, ebgp_node, type="ebgp")
            # connect ebgp node and network
            graph.add_edge(ebgp_node, network_node, type="network")
            graph.add_edge(network_node, ebgp_node, type="network")

            # choose route to be advertised via gateway
            bgp_route = choose_random(routes_per_network[network_node], s).copy()
            bgp_route.bgp_speaker_id = ebgp_node
            bgp_route.is_ebgp_learned = True
            graph.nodes[ebgp_node]["bgp_route"] = bgp_route

    if not FULLY_MESHED:
        # add route reflector nodes
        route_reflector_nodes = set()
        for n in range(NUM_ROUTE_REFLECTORS):
            node_id = s.randint(0, NUM_NODES)
            route_reflector_nodes.add(node_id)
            graph.add_node(node_id)
            graph.nodes[node_id]["type"] = "route_reflector"

        # fully mesh route reflectors
        for rr in route_reflector_nodes:
            for other_rr in route_reflector_nodes:
                if rr == other_rr: continue
                graph.add_edge(rr, other_rr, type="ibgp")
                graph.add_edge(other_rr, rr, type="ibgp")

        # every router node is connected to one of the route reflectors
        for r in router_nodes:
            rr = choose_random(list(route_reflector_nodes), s)
            graph.add_edge(r, rr, type="ibgp")
            graph.add_edge(rr, r, type="ibgp")
    else:
        for r1 in router_nodes:
            for r2 in router_nodes:
                if r1 == r2: continue
                graph.add_edge(r1, r2, type="ibgp")
                graph.add_edge(r2, r1, type="ibgp")
        print("generating fully meshed ibgp session layout")

    return graph


def generate_graph(seed, NUM_NODES=None, NUM_NETWORKS=None, NUM_GATEWAY_NODES=None, NUM_ROUTE_REFLECTORS=None,
                   NUM_ACL_ROLE=None,
                   FULLY_MESHED=False, MAX_INTERFACE = 4):
    # 生成网络建模图，其中包括各种参数，权重，bgproute等
    if NUM_NODES is None: NUM_NODES = 10
    if NUM_NETWORKS is None: NUM_NETWORKS = 2
    if NUM_GATEWAY_NODES is None: NUM_GATEWAY_NODES = min(int(NUM_NODES / 2), 4)
    if NUM_ROUTE_REFLECTORS is None: NUM_ROUTE_REFLECTORS = 2
    if NUM_ACL_ROLE is None: NUM_ACL_ROLE = 4
    MAX_WEIGHT = 32

    s = np.random.RandomState(seed=seed)
    graph, pos = random_planar_graph(NUM_NODES, s, MAX_INTERFACE)

    #  1. 初始化AS内部的普通节点及链路，由ospf路由
    nx.set_node_attributes(graph, "router", name="type")

    # ospf权重
    for src, tgt in graph.edges():
        weight = s.randint(1, MAX_WEIGHT)
        graph[src][tgt]["weight"] = weight
        graph[tgt][src]["weight"] = weight

    nx.set_edge_attributes(graph, "ospf", name="type")

    router_nodes = set(graph.nodes())

    # 向graph添加 BGP相关节点（external、network、reflector）
    # add network nodes
    network_nodes = set()
    routes_per_network = {}

    for n in range(NUM_NETWORKS):
        node_id = len(graph.nodes())
        network_nodes.add(node_id)  # 2. 新节点，外部网络节点
        routes_per_network[node_id] = generate_random_route_announcements(node_id)  # 这个网络可能的通告，后面会随机选择一个
        graph.add_node(node_id, type="network")

    ## configure gateway nodes
    ebgp_nodes = set()
    for network_node in network_nodes:
        for n in range(NUM_GATEWAY_NODES):  # 每个网络都会与这三个GATEWAY_NODES建立联系？
            ebgp_node = len(graph.nodes())  # 新节点，每个网络有2个外部网关路由器

            while True:
                gateway_node = s.randint(0, NUM_NODES) # 随机找内部节点作为ebgp
                if graph.out_degree(gateway_node) < MAX_INTERFACE:break

            graph.add_node(ebgp_node, type="external")
            ebgp_nodes.add(ebgp_node)

            # connect gateway and ebgp node
            graph.add_edge(ebgp_node, gateway_node, type="ebgp")
            graph.add_edge(gateway_node, ebgp_node, type="ebgp")
            # connect ebgp node and network
            graph.add_edge(ebgp_node, network_node, type="network")
            graph.add_edge(network_node, ebgp_node, type="network")

            # 选择网关发布的路由
            bgp_route = choose_random(routes_per_network[network_node], s).copy()
            bgp_route.bgp_speaker_id = ebgp_node
            bgp_route.is_ebgp_learned = True
            graph.nodes[ebgp_node]["bgp_route"] = bgp_route

    if FULLY_MESHED:
        for r1 in router_nodes:
            for r2 in router_nodes:
                if r1 == r2: continue
                graph.add_edge(r1, r2, type="ibgp")
                graph.add_edge(r2, r1, type="ibgp")
        print("generating fully meshed ibgp session layout")
    else:
        # add route reflector nodes
        route_reflector_nodes = set()
        for n in range(NUM_ROUTE_REFLECTORS):
            node_id = s.randint(0, NUM_NODES)
            route_reflector_nodes.add(node_id)
            graph.add_node(node_id)
            graph.nodes[node_id]["type"] = "route_reflector"

        # fully mesh route reflectors
        for rr in route_reflector_nodes:
            for other_rr in route_reflector_nodes:
                if rr == other_rr: continue
                graph.add_edge(rr, other_rr, type="ibgp")
                graph.add_edge(other_rr, rr, type="ibgp")

        # every router node is connected to one of the route reflectors
        for r in router_nodes:
            if r in route_reflector_nodes: continue
            rr = choose_random(list(route_reflector_nodes), s)
            graph.add_edge(r, rr, type="ibgp")
            graph.add_edge(rr, r, type="ibgp")

    for n in range(NUM_ACL_ROLE):
        net_id = choose_random(list(network_nodes), s)
        ospf_edges = [(src, dst) for src, dst in graph.edges() if "weight" in graph[src][dst].keys()]
        edge = choose_random(ospf_edges, s)
        for i in range(len(edge)):
            if edge[i] not in route_reflector_nodes:
                if not "acl_role" in graph[edge[0]][edge[1]].keys():
                    graph[edge[0]][edge[1]]["acl_role"] = set([AclRole(net_id, edge[i], i)])  # 抽象图是在边上添加ACL规则
                else:
                    graph[edge[0]][edge[1]]["acl_role"].add(AclRole(net_id, edge[i], i))
                break

    return graph


def draw_graph(G, pos=None, destination=-1, figsize=(10, 10), label="", use_node_labels=True):
    G = G.copy()

    pos = pos if pos is not None else nx.drawing.layout.spring_layout(G, weight=None)

    labels = {}
    edge_color = []
    node_color = []

    for src, tgt in list(G.edges()):
        is_forwarding = destination in G[src][tgt]["is_forwarding"].keys() \
            if destination != -1 and prop(G[src][tgt], "is_forwarding") is not None \
            else destination == -1
        if not is_forwarding and G[src][tgt]["type"] != "network":
            G.remove_edge(src, tgt)

    for src, tgt in G.edges():
        edge_type = G[src][tgt]["type"]
        is_ospf_link = "weight" in G[src][tgt].keys()
        is_bgp_link = edge_type == "ibgp" or edge_type == "ebgp"
        is_network_link = edge_type == "network"

        if is_ospf_link:
            w = G[src][tgt]["weight"]
            labels[(src, tgt)] = f"{w}"
            edge_color.append("gray")
        elif is_bgp_link:
            labels[(src, tgt)] = edge_type
            edge_color.append("gray")
        elif is_network_link:
            labels[(src, tgt)] = "network"
            edge_color.append("yellow")
        else:
            assert False, f"unknown network graph edge type {edge_type}"

    for i in G.nodes():
        if prop(G.nodes[i], "type") == "route_reflector":
            node_color.append("green")
        elif prop(G.nodes[i], "bgp_route") is not None:
            node_color.append("yellow")
        elif prop(G.nodes[i], "type") == "network":
            node_color.append("red")
        else:
            node_color.append("red" if i == destination else "lightblue")

    fig = plt.Figure(figsize=figsize)
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    plt.axis('off')

    def node_label(n):
        if not use_node_labels: return str(n)
        return prop(G.nodes[n], "label") or str(n)

    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels, ax=ax)
    nx.draw_networkx_labels(G, labels=dict([(n, node_label(n)) for n in G.nodes()]), pos=pos, ax=ax)
    nx.draw(G, pos=pos, edge_color=edge_color, node_color=node_color, ax=ax, arrowsize=20,
            arrowstyle="-|>" if destination != -1 else "-")

    ax.set_title(label)
    return canvas

def draw_dev_graph(G, pos=None, destination=-1, figsize=(10, 10), label="", use_node_labels=True):
    G = G.copy()

    pos = pos if pos is not None else nx.drawing.layout.spring_layout(G, weight=None)

    labels = {}
    edge_color = []
    node_color = []

    for src, tgt in list(G.edges()):
        is_forwarding = destination in G[src][tgt]["is_forwarding"].keys() \
            if destination != -1 and prop(G[src][tgt], "is_forwarding") is not None \
            else destination == -1
        if not is_forwarding and G[src][tgt]["type"] != "network":
            G.remove_edge(src, tgt)

    interface_pos = {}
    interface_labe = {}
    for edge in G.edges():
        u, v = edge
        if G.edges[u, v]['type'] == 'ibgp' and 'weight' not in G.edges[u, v].keys(): continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        out_pos = (x1 + (x2 - x1) / 6, y1 + (y2 - y1) / 6)

        interface_labe[edge] = G.edges[u, v]['interface_info']['out']
        interface_pos[edge] = out_pos

    for i in G.nodes():
        if prop(G.nodes[i], "type") == "route_reflector":
            node_color.append("green")
        elif prop(G.nodes[i], "bgp_route") is not None:
            node_color.append("yellow")
        elif prop(G.nodes[i], "type") == "network":
            node_color.append("red")
        else:
            node_color.append("red" if i == destination else "lightblue")

    fig = plt.Figure(figsize=figsize)
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    plt.axis('off')

    def node_label(n):
        if not use_node_labels: return str(n)
        return prop(G.nodes[n], "label") or str(n)

    nx.draw_networkx_nodes(G, pos,node_color=node_color, ax=ax)
    nx.draw_networkx_edge_labels(G,label_pos=0.8, pos=pos, edge_labels=interface_labe,ax=ax)
    nx.draw_networkx_labels(G, labels=dict([(n, node_label(n)) for n in G.nodes()]), pos=pos, ax=ax)
    filtered_edges = [(u, v) for u, v, d in G.edges(data=True) if ('type' in d and d['type'] != 'ibgp') or 'weight' in d]
    nx.draw_networkx_edges(G, pos, edgelist=filtered_edges, ax=ax)

    # nx.draw(G, pos=pos, node_color=node_color, ax=ax, arrowsize=20,
    #         arrowstyle="-|>" if destination != -1 else "-")

    ax.set_title(label)
    return canvas

def top_group_for_attr(routes, fct, lower_is_better=True):  # 在节点所接收的bgp-公告中根据属性选择最优的
    best_val = float("inf") if lower_is_better else -float("inf")
    for r in routes:
        val = fct(r)
        if lower_is_better and val < best_val:
            best_val = val
        elif not lower_is_better and val > best_val:
            best_val = val

    return list(filter(lambda r: abs(fct(r) - best_val) < 0.00001, routes))


def bgp_select(node_id, routes, dist_to_destination):  # 选择最优来自一个dest的bgp宣告
    if len(routes) < 1: return None

    # remove duplicate entries
    routes = list(set(routes))

    # LOCAL_PREFERENCE
    highest_local_preference = top_group_for_attr(routes, lambda r: r.local_preference, lower_is_better=False)
    # AS_PATH_LENGTH
    lowest_as_path_length = top_group_for_attr(highest_local_preference, lambda r: r.as_path_length)
    # ORIGIN_TYPE (skipped for now)

    # best_origin = top_group_for_attr(lowest_as_path_length, lambda r: r.origin_type)

    # MED (skipped for now)
    best_med = top_group_for_attr(lowest_as_path_length, lambda r: r.med)

    # external routes over internal routes  (skipped for now)
    # best_external_over_internal = top_group_for_attr(best_med, lambda r: 0 if r.is_ebgp_learned else 1)

    # IGP cost
    def igp_cost(r):
        dists = dist_to_destination[r.next_hop]
        if not node_id in dists.keys():
            return 9999
        return dists[node_id]

    best_by_igp_cost = top_group_for_attr(best_med, igp_cost)
    # BGP speaker ID
    best_by_bgp_speaker_id = top_group_for_attr(best_by_igp_cost, lambda r: r.bgp_speaker_id)

    if len(best_by_bgp_speaker_id) > 1:
        print("fatal error: BGP decision criteria did not yield a unique result at node", node_id)
        for r in best_by_bgp_speaker_id:
            print(r)

    assert len(best_by_bgp_speaker_id) <= 1, "fatal error: BGP decision criteria did not yield a unique result"
    if len(best_by_bgp_speaker_id) < 1:
        print("warning: BGP decision criteria filtered all routes")
        return None

    return best_by_bgp_speaker_id[0]


def propagate(g, bgp_state, edges):
    for src, dst in edges:
        src_route = bgp_state[src].outbox
        if src_route is not None: bgp_state[dst].available_routes.add(src_route)  # 将bgp-route从src传播到dst
        dst_route = bgp_state[dst].outbox
        if dst_route is not None: bgp_state[src].available_routes.add(dst_route)
        # 传播了两遍


def update_node(node_id, node, dist_to_destination, g):  # 更新第node_id个节点
    available_routes = set(([node.locRib] if node.locRib is not None else []) + list(node.available_routes))
    best_route: BgpRoute = bgp_select(node_id, available_routes, dist_to_destination)

    # continue propagation of best_route if it is better than the current node.adjRibOut， 如果就是原来的那就不需要传播了
    outbox = None
    if best_route != node.adjRibOut and (best_route.is_ebgp_learned or node.type == "route_reflector"):
        outbox = best_route.copy()
        if best_route.is_ebgp_learned:  # 只有这种情况才会修改next hop
            outbox.next_hop = node_id
        outbox.is_ebgp_learned = False

    return BgpNodeState(node.type, best_route, outbox, outbox or node.adjRibOut, set())


def update(g, bgp_state, dist_to_destination):
    num_updates = 0

    for n in g.nodes():
        s = bgp_state[n]
        if s.type == "network" or s.type == "external": continue
        before_speaker_id = (bgp_state[n].locRib.bgp_speaker_id if bgp_state[n].locRib is not None else -1)

        bgp_state[n] = update_node(n, s, dist_to_destination, g)

        updated_speaker_id = (bgp_state[n].locRib.bgp_speaker_id if bgp_state[n].locRib is not None else -1)
        if before_speaker_id != updated_speaker_id: num_updates += 1
    return num_updates


from dataclasses import dataclass


@dataclass
class BgpNodeState:
    type: str
    locRib: BgpRoute
    outbox: BgpRoute
    adjRibOut: BgpRoute
    available_routes: set


def compute_forwarding_state(g: DiGraph):  # 计算转发表
    external_nodes = [n for n in g.nodes() if g.nodes[n]["type"] == "external"]
    networks = [n for n in g.nodes() if g.nodes[n]["type"] == "network"]

    ospf_edges = [(src, dst) for src, dst in g.edges() if "weight" in g[src][dst].keys()]
    ebgp_edges = [(src, dst) for src, dst in g.edges() if g[src][dst]["type"] == "ebgp"]

    edge_routers = []  # 连接external的路由器
    for e in external_nodes:
        edge_routers += [n for n in g.neighbors(e) if g.nodes[n]["type"] == "router"]

    # initialise forwarding edge attribute
    for src, dst in ospf_edges + ebgp_edges: g[src][dst]["is_forwarding"] = {}

    # compute shortest paths
    dist_to_destination = {}  # dist_to_destination[D][n] := dist from n to D
    next_router_to_destination = {}  # next_router_to_destination[D][n] := next router on shortest path from n to D

    g_ospf = nx.DiGraph()  # AS 0内部路由
    for src, dst in ospf_edges:
        g_ospf.add_edge(src, dst, weight=g[src][dst]["weight"])
        g_ospf.add_edge(dst, src, weight=g[src][dst]["weight"])
    for src, dst in ebgp_edges:
        g_ospf.add_edge(src, dst, weight=1)
        g_ospf.add_edge(dst, src, weight=1)

    igp_destinations = external_nodes + edge_routers + [n for n in g.nodes() if g.nodes[n]["type"] == "route_reflector"]

    for destination in igp_destinations:
        preds_per_node, dists = nx.algorithms.dijkstra_predecessor_and_distance(g_ospf, destination)
        dist_to_destination[destination] = dists
        next_router_to_destination[destination] = preds_per_node  # ospf的下一跳

    # BGP模拟
    for net in networks:
        # initialise BGP state
        bgp_state = {}
        for n in g.nodes():
            type = g.nodes[n]["type"]
            bgp_route = prop(g.nodes[n], "bgp_route")
            if bgp_route is not None:
                if bgp_route.destination != net:
                    bgp_route = None
                else:
                    bgp_route.next_hop = n
            bgp_state[n] = BgpNodeState(type, bgp_route, bgp_route, None, set())

        bgp_edges = [(src, dst) for src, dst in g.edges() if
                     g[src][dst]["type"] == "ibgp" or g[src][dst]["type"] == "ebgp"]  # BGP 传播边
        max_iteration = 10

        num_iterations = 0
        num_changes = 1

        while num_changes > 0:
            if max_iteration <= num_iterations: break
            num_iterations += 1

            propagate(g, bgp_state, bgp_edges)
            num_changes = update(g, bgp_state, dist_to_destination)
        # print(f"BGP simulation finished after {num_iterations} iterations")

        for n in g.nodes():
            s = bgp_state[n]
            type = g.nodes[n]["type"]
            if type != "router" and type != "route_reflector": continue  #只从这两种节点向外找

            if s.locRib is None:
                # print(f"warning: network {net} is not reachable from node {n}")
                continue
            # assert s.locRib is not None, f"node {n} has not received any of the advertised BGP routes for network {net}"

            # store info on next hop per node
            if "next_hop" not in g.nodes[n].keys(): g.nodes[n]["next_hop"] = {}
            g.nodes[n]["next_hop"][net] = s.locRib.next_hop

            if "peer" not in g.nodes[n].keys(): g.nodes[n]["peer"] = {}
            g.nodes[n]["peer"][net] = s.locRib.bgp_speaker_id

            if s.locRib.next_hop != n:
                next_router = next_router_to_destination[s.locRib.next_hop][n][0]
                if 'acl_role' in g[n][next_router].keys():
                    if net in [acl.dst for acl in g[n][next_router]['acl_role']]:
                        continue  # 不转发
                g[n][next_router]["is_forwarding"][net] = 1
            else:
                next_router = n

        # #
        # for src, dst in ospf_edges + ebgp_edges:
        #     if net in g[src][dst]["is_forwarding"].keys() and net in g[dst][src]["is_forwarding"].keys():
        #         if g[src][dst]["is_forwarding"][net] == g[dst][src]["is_forwarding"][net]:
        #             print(f"{src}-{dst}：{net}")
        #
        # with open(f"data/0/{net}_state.pkl", 'wb') as f:
        #     record = {'bgp_state': bgp_state, 'ospf_sp': next_router_to_destination,
        #               'dist': dist_to_destination, "g": g}
        #     pickle.dump(record, f)
        # print('stop')

    network_label_mapping = {}
    for n in networks:
        if not "net_label" in g.nodes[n].keys():
            # not using labeled_networks
            return
        net_label = g.nodes[n]["net_label"]
        network_label_mapping[n] = net_label

    for src, dst in g.edges():
        if "is_forwarding" in g[src][dst]:
            g[src][dst]["is_forwarding"] = dict(
                [(network_label_mapping[n], holds) for n, holds in g[src][dst]["is_forwarding"].items()])


class BgpSemantics(Semantics):
    def __init__(self, with_groundtruth=False, with_real_world_topologies_p=0.0, labeled_networks=False):
        self.with_groundtruth = with_groundtruth
        self.with_real_world_topologies_p = with_real_world_topologies_p

        self.predicate_semantics = [
            ForwardingPathPredicateSemantics(),
            # FullForwardingPlanePredicateSemantics(),
            TrafficIsolationPredicateSemantics(),
            ReachablePredicateSemantics()
        ]
        self.predicate_semantics_sample_config = {
            "fwd": {
                "n": 32
            },
            "reachable": {
                "n": 12
            },
            "trafficIsolation": {
                "n": 12
            }
        }
        self.labeled_networks = labeled_networks

    def sample(self, num_nodes=None, num_networks=None, NUM_GATEWAY_NODES=None, NUM_ACL_ROLE=None, seed=None, real_world_topology=False,
               skip_fwd_facts_p=0.0,
               predicate_semantics_sample_config_overrides={}, basedOnGraph=None, FULLY_MESHED=False,
               use_topology_file=None, MAX_INTERFACE = 4):
        s = np.random.RandomState(seed=seed)

        if basedOnGraph is not None:
            self.graph = basedOnGraph
        elif not real_world_topology:  # 生成随机拓扑
            self.graph = generate_graph(seed=seed, NUM_NODES=num_nodes, NUM_NETWORKS=num_networks,
                                   NUM_GATEWAY_NODES=NUM_GATEWAY_NODES, NUM_ACL_ROLE=NUM_ACL_ROLE, FULLY_MESHED=FULLY_MESHED, MAX_INTERFACE = MAX_INTERFACE)

        else:
            topology_file = choose_random(topologies.all_topology_files, s=s)
            if use_topology_file is not None: topology_file = use_topology_file
            self.graph = generate_graph_with_topology(topology_file, seed=seed, NUM_NETWORKS=num_networks,
                                                 NUM_GATEWAY_NODES=NUM_GATEWAY_NODES, FULLY_MESHED=FULLY_MESHED)
            num_nodes = len(self.graph.nodes())

        router_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n]["type"] == "router"]
        network_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n]["type"] == "network"]
        external_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n]["type"] == "external"]
        route_reflector_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n]["type"] == "route_reflector"]

        def c(r):
            return Constant(f"c{r}")

        # mapping of traffic classes / networks
        self.network_mapping = {}
        for n in network_nodes:
            self.network_mapping[c(n).name] = len(self.network_mapping)

        # set names in network graph
        for n in self.graph.nodes():
            self.graph.nodes[n]["label"] = c(n).name

        # compute forwarding state
        compute_forwarding_state(self.graph)

        def get_overrides(pred_s):
            if pred_s.predicate_name in predicate_semantics_sample_config_overrides.keys():
                return predicate_semantics_sample_config_overrides[
                    pred_s.predicate_name]  # {"n": choose_random([8, 10, 12], s)}
            return {}

        facts = []

        # 从计算的转发平面派生规范谓词
        for pred_s in self.predicate_semantics:
            config = self.sampling_config(pred_s, overrides=get_overrides(pred_s))  # config就是一个谓词抽样的个数 {"n": 10}
            derived = pred_s.sample(self.graph, random=s, **config)

            if self.labeled_networks:
                for f in derived:
                    def network_constants_to_label(a):
                        if type(a) is Constant and a.name in self.network_mapping.keys(): return self.network_mapping[a.name]
                        return a

                    f.args = [network_constants_to_label(a) for a in f.args]

            facts += derived
        return self.graph, facts

    def sampling_config(self, predicate_semantics, overrides={}):
        if predicate_semantics.predicate_name in self.predicate_semantics_sample_config.keys():
            config = dict(self.predicate_semantics_sample_config[predicate_semantics.predicate_name])
        else:
            config = {}
        config.update(overrides)
        return config
