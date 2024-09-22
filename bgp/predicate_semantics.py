from dataclasses import dataclass
from networkx.algorithms.isolate import is_isolate
import numpy as np
from nutils import choose_random

"""
该文件包含提取和检查支持的谓词的逻辑
通过我们的规范语言:fwd、可达和流量隔离。
"""


class Constant:  # 常量
    def __init__(self, name, id=None):
        self.name = name
        self.id = id

    def __repr__(self):
        return self.name


class PredicateAst:
    def __init__(self, name, args, is_negated, is_query):
        self.name = name
        self.args = args
        self.is_negated = is_negated
        self.is_query = is_query

    def __repr__(self):
        def argstr(a): return str(a) if a != -2 else "?"

        args_string = ",".join([argstr(a) for a in self.args])
        is_negated = "not " if self.is_negated else ""
        is_query = "? " if self.is_query else ""
        return f"{is_query}{is_negated}{self.name}({args_string})"


class PredicateSemantics:
    def check(self, network, predicate):
        """
        检查所提供的谓词是否适用于network
        """
        pass

    def check_all(self, network, predicates):
        results = np.zeros([len(predicates)])
        for i, pred in enumerate(predicates):
            results[i] = self.check(network, pred)
        return results.mean(), list(zip(predicates, results))

    def sample(self, network, n=0, random: np.random.RandomState = None):
        """
        对于所提供的网络，采样n个支持或不支持的谓词的实例。确保正面实例的数量与负面实例的数量大致匹配。
        """
        pass

    @property
    def predicate_name(self):
        return "<none>"


class FullForwardingPlanePredicateSemantics(PredicateSemantics):
    @property
    def predicate_name(self):
        return "fwd"

    def check(self, network, predicate: PredicateAst):
        assert predicate.name == "fwd", "can only check fwd predicates with ForwardingPathPredicateSemantics"
        assert len(
            predicate.args) == 3, f"expected fwd predicate of the form fwd(src, net, dst), got {predicate} instead"

        nodes_by_label = dict([(network.nodes[n]["label"], n) for n in network.nodes()])

        src = nodes_by_label[predicate.args[0].name]
        net = nodes_by_label[predicate.args[1].name] if type(predicate.args[1]) is Constant else predicate.args[1]
        dst = nodes_by_label[predicate.args[2].name]

        assert src is not None and net is not None and dst is not None, f"failed to resolve arguments of {predicate}"
        is_forwarding = network[src][dst]["is_forwarding"]
        if predicate.is_negated:
            return not (net in is_forwarding.keys())
        else:
            return net in is_forwarding.keys()

    def sample(self, network, random: np.random.RandomState, n=0):
        destinations = [n for n in network.nodes() if
                        network.nodes[n]["type"] == "network" or network.nodes[n]["type"] == "external"]

        result = []
        for src, dst in network.edges():
            if not "is_forwarding" in network[src][dst]: continue

            is_forwarding = network[src][dst]["is_forwarding"]
            for net in destinations:
                is_negated = not net in is_forwarding
                result.append(
                    PredicateAst("fwd", [Constant(network.nodes[n]["label"]) for n in [src, net, dst]], is_negated,
                                 False))
        return result


class ForwardingPathPredicateSemantics(PredicateSemantics):
    def __init__(self):
        self.included_fw_predicates = set()

    @property
    def predicate_name(self):
        return "fwd"

    def check(self, network, predicate: PredicateAst):
        assert predicate.name == "fwd", "can only check fwd predicates with ForwardingPathPredicateSemantics"
        assert len(
            predicate.args) == 3, f"expected fwd predicate of the form fwd(src, net, dst), got {predicate} instead"

        nodes_by_label = dict([(network.nodes[n]["label"], n) for n in network.nodes()])

        src = nodes_by_label[predicate.args[0].name]
        net = nodes_by_label[predicate.args[1].name] if type(predicate.args[1]) is Constant else predicate.args[1]
        dst = nodes_by_label[predicate.args[2].name]

        assert src is not None and net is not None and dst is not None, f"failed to resolve arguments of {predicate}"
        is_forwarding = network[src][dst]["is_forwarding"]
        if predicate.is_negated:
            return not (net in is_forwarding.keys())
        else:
            return net in is_forwarding.keys()

    # 这个采样是只基于转发表吗？
    def sample(self, network, n=0, random: np.random.RandomState = None, per_network=True):
        if random is None: random = np.random

        router_nodes = [n for n in network.nodes() if network.nodes[n]["type"] == "router"]
        destinations = [n for n in network.nodes() if
                        network.nodes[n]["type"] == "network"]  # or network.nodes[n]["type"] == "external"]

        # 提取转发表
        next_hop = dict(
            [(net, {}) for net in destinations])  # next_hop[network][node] = list of successor nodes in forwarding tree
        not_next_hop = dict(
            [(net, {}) for net in destinations])  # next_hop[network][node] = list of successor nodes in forwarding tree

        # 构造转发树，起始点为网络节点
        for src, dst in network.edges():
            if not "is_forwarding" in network[src][dst]: continue
            forwarding_table = network[src][dst]["is_forwarding"]
            for net in destinations:
                if net in forwarding_table:
                    if not src in next_hop[net].keys():
                        next_hop[net][src] = set()
                    next_hop[net][src] = set([*list(next_hop[net][src]), dst])
                else:
                    if not src in not_next_hop[net].keys():
                        not_next_hop[net][src] = set()
                    not_next_hop[net][src] = set([*list(not_next_hop[net][src]), dst])

        # walk and track random paths
        path_length = [2, 3, 4]
        is_destination = set(destinations)

        result = []

        num_paths = n

        def generate_fwd_path_req(d):  # 随机采样
            l = choose_random(path_length)
            n = choose_random(router_nodes)
            is_negative_sample = np.random.random() < 0.5

            path = [n]
            j = 0

            while not n in is_destination and j < l:
                if not is_negative_sample:
                    next_set = next_hop[d]
                else:
                    next_set = not_next_hop[d]

                if n not in next_set.keys(): break

                next_n = choose_random(list(next_set[n]))

                fwd_hash = (n, d, next_n, is_negative_sample)
                if fwd_hash not in self.included_fw_predicates:
                    result.append(PredicateAst("fwd", [Constant(network.nodes[n]["label"]) for n in [n, d, next_n]],
                                               is_negative_sample, False))
                    self.included_fw_predicates.add(fwd_hash)

                j += 1
                n = next_n
                path.append(n)

        if per_network:
            for d in destinations:
                for i in range(num_paths):
                    generate_fwd_path_req(d)
        else:
            for i in range(num_paths):
                generate_fwd_path_req(choose_random(destinations))

        return result


class ReachablePredicateSemantics(PredicateSemantics):
    def __init__(self):
        self.included_reachable_predicates = set()

    @property
    def predicate_name(self):
        return "reachable"

    def check(self, network, predicate: PredicateAst):
        assert predicate.name == "reachable", "can only check reachable predicates with ReachPredicateSemantics"
        assert len(
            predicate.args) == 3, f"expected reach predicate of the form reachable(src, net, dst), got {predicate} instead"

        nodes_by_label = dict([(network.nodes[n]["label"], n) for n in network.nodes()])

        src = nodes_by_label[predicate.args[0].name]
        net = nodes_by_label[predicate.args[1].name] if type(predicate.args[1]) is Constant else predicate.args[1]
        dst = nodes_by_label[predicate.args[2].name]

        assert src is not None and net is not None and dst is not None, f"failed to resolve arguments of {predicate}"

        destinations = [n for n in network.nodes() if network.nodes[n]["type"] == "network"]
        is_destination = set([net])

        # extract forwarding tables
        next_hop = self._extract_forwarding_table(network, destinations)

        n = src
        reachable = set([])
        while True:
            if n not in next_hop[net].keys(): break

            next_n = next_hop[net][n]
            n = next_n
            # cycle detection
            if n in reachable: break
            if n in is_destination: break
            reachable.add(n)

        if dst in reachable:
            return not predicate.is_negated
        return predicate.is_negated

    def _extract_forwarding_table(self, network, destinations):
        destinations = [network.nodes[d]["net_label"] if "net_label" in network.nodes[d].keys() else d for d in
                        destinations]
        next_hop = dict(
            [(net, {}) for net in destinations])  # next_hop[network][node] = list of successor nodes in forwarding tree

        for src, dst in network.edges():
            if not "is_forwarding" in network[src][dst]: continue

            forwarding_table = network[src][dst]["is_forwarding"]
            for net in destinations:
                if net in forwarding_table:
                    if src in next_hop[net].keys():
                        next_hop[net][src] = max(dst, next_hop[net][src])
                    else:
                        next_hop[net][src] = dst
        return next_hop

    def sample(self, network, n=0, random: np.random.RandomState = None, per_network=True):
        if random is None: random = np.random

        router_nodes = [n for n in network.nodes() if network.nodes[n]["type"] == "router"]
        destinations = [n for n in network.nodes() if network.nodes[n]["type"] == "network"]

        # extract forwarding tables
        next_hop = self._extract_forwarding_table(network, destinations)

        # walk and track random paths
        is_destination = set(destinations)
        result = []

        num_reachable_paths = n

        def gen_reachable_req(d):
            n_start = choose_random(router_nodes, s=random)
            reachable = set([])

            n = n_start
            while True:
                if n not in next_hop[d].keys(): break

                next_n = next_hop[d][n]
                n = next_n
                # cycle detection
                if n in reachable: break
                if n in is_destination: break
                reachable.add(n)

            if len(reachable) == 0: return

            via_n = choose_random(list(reachable), s=random)
            is_negative_sample = random.random() < 0.5

            if is_negative_sample:
                reachable.add(n_start)
                not_reachable = set(router_nodes).difference(reachable)
                if len(not_reachable) == 0: return
                via_n = choose_random(list(not_reachable), s=random)
                reachable_hash = (n_start, d, via_n, is_negative_sample)
                if reachable_hash not in self.included_reachable_predicates:
                    result.append(PredicateAst("reachable",
                                               [Constant(network.nodes[n]["label"]) for n in [n_start, d, via_n]],
                                               True, False))
                    self.included_reachable_predicates.add(reachable_hash)
            else:
                reachable_hash = (n_start, d, via_n, is_negative_sample)
                if reachable_hash not in self.included_reachable_predicates:
                    result.append(PredicateAst("reachable",
                                               [Constant(network.nodes[n]["label"]) for n in [n_start, d, via_n]],
                                               False, False))
                    self.included_reachable_predicates.add(reachable_hash)

        if per_network:
            for d in destinations:
                for i in range(num_reachable_paths):
                    gen_reachable_req(d)
        else:
            for i in range(num_reachable_paths):
                gen_reachable_req(choose_random(destinations))

        return result


class TrafficIsolationPredicateSemantics(PredicateSemantics):
    def __init__(self):
        self.included_iso_predicates = set()

    @property
    def predicate_name(self):
        return "trafficIsolation"

    def check(self, network, predicate: PredicateAst):
        assert predicate.name == "trafficIsolation", "can only check trafficIsolation predicates with TrafficIsolationPredicateSemantics"
        assert len(
            predicate.args) == 4, f"expected reach predicate of the form trafficIsolation(src, dst, net1, net2), got {predicate} instead"

        nodes_by_label = dict([(network.nodes[n]["label"], n) for n in network.nodes()])

        src = nodes_by_label[predicate.args[0].name]
        dst = nodes_by_label[predicate.args[1].name]
        net1 = nodes_by_label[predicate.args[2].name] if type(predicate.args[2]) is Constant else predicate.args[2]
        net2 = nodes_by_label[predicate.args[3].name] if type(predicate.args[3]) is Constant else predicate.args[3]

        assert src is not None and dst is not None and net1 is not None and net2 is not None, f"failed to resolve arguments of {predicate}"

        is_forwarding = network[src][dst]["is_forwarding"]
        is_isolated = not (net1 in is_forwarding.keys() and net2 in is_forwarding.keys())

        if predicate.is_negated:
            return not is_isolated
        else:
            return is_isolated

    def sample(self, network, n=0, random: np.random.RandomState = None, per_network=True):
        if random is None: random = np.random

        destinations = [n for n in network.nodes() if
                        network.nodes[n]["type"] == "network" or network.nodes[n]["type"] == "external"]

        forwarding_edges = dict(())

        for src, dst in network.edges():
            if not "is_forwarding" in network[src][dst]: continue

            forwarding_table = network[src][dst]["is_forwarding"]
            for net in destinations:
                if network.nodes[src]["type"] != "router" or network.nodes[dst]["type"] != "router": continue

                if net in forwarding_table:
                    edges = forwarding_edges[net] if net in forwarding_edges.keys() else set()
                    edges.add((src, dst))
                    forwarding_edges[net] = edges

        # limit set of destinations to those occuring in forwarding tables
        destinations = list(forwarding_edges.keys())

        isolated_edges = []
        non_isolated_edges = []

        for net1 in destinations:
            for net2 in destinations:
                if net1 == net2: continue

                common_edges = forwarding_edges[net1].intersection(forwarding_edges[net2])
                for src, dst in common_edges: non_isolated_edges.append((src, dst, net1, net2))

                exclusive_edges = forwarding_edges[net1].difference(forwarding_edges[net2])
                for src, dst in exclusive_edges: isolated_edges.append((src, dst, net1, net2))

        result = []

        for i in range(n):
            if len(isolated_edges) == 0 or len(non_isolated_edges) == 0: break

            is_negative_sample = random.rand() < 0.5
            args = choose_random(non_isolated_edges if is_negative_sample else isolated_edges, random)

            iso_hash = (args, is_negative_sample)
            if iso_hash not in self.included_iso_predicates:
                result.append(PredicateAst("trafficIsolation",
                                           [Constant(network.nodes[n]["label"]) for n in args],
                                           is_negative_sample, False))
                self.included_iso_predicates.add(iso_hash)

                src, dst, net1, net2 = args

                if is_negative_sample:
                    assert (net1 in network[src][dst]["is_forwarding"].keys() and net2 in network[src][dst][
                        "is_forwarding"].keys())
                else:
                    assert (net2 in network[src][dst]["is_forwarding"].keys() and not net1 in network[src][dst][
                        "is_forwarding"].keys()) or \
                           (net1 in network[src][dst]["is_forwarding"].keys() and not net2 in network[src][dst][
                               "is_forwarding"].keys())

        return result
