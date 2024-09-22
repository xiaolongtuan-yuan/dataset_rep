# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/20 10:24
@Auth ： xiaolongtuan
@File ：BgpBuilder.py
"""
import copy
import json
import math
import pickle
import random
import shutil
from ipaddress import IPv4Network

import pandas as pd
from torch_geometric.data import Data
import numpy as np

from bgp.predicate_semantics import Constant
from bgp_semantics import BgpSemantics, draw_graph, draw_dev_graph, compute_forwarding_state
from bgp_device import *
import networkx as nx
from networkx import spring_layout
import matplotlib.pyplot as plt


class QueryType(Enum):
    ROUTE = 1
    INTERFACE = 2
    NETWORK = 3


ORIGIN_TYPE = ['ibp', 'egp', 'incomplete']


class IpDistributor:
    '''
    全部使用C类网络
    '''

    def __init__(self):
        self.counter = 0
        self.counter2 = 0
        self.base_add = ipaddress.IPv4Address("1.0.0.0")
        self.mask = ipaddress.IPv4Address("255.255.255.0")

        self.loop_add = ipaddress.IPv4Address("1.1.1.1")
        self.loop_mask = ipaddress.IPv4Address("255.255.255.255")

    def assign_address(self, interface_type: Interface_type):
        if interface_type == Interface_type.GigabitEthernet:
            network_add = ipaddress.IPv4Address((int(self.base_add) + self.counter * 256))
            self.counter += 1
            return (network_add, self.mask)
        elif interface_type == Interface_type.Loopback:
            network_add = ipaddress.IPv4Address((int(self.loop_add) + self.counter2))
            self.counter2 += 1
            return (network_add, self.loop_mask)
        else:
            raise Exception("分配接口网络类型不存在")


class BgpBuilder:
    def __init__(self, net_seq, network_root, need_embeded=False):
        self.network_root = network_root
        config_path = f'{self.network_root}/configs'

        if os.path.exists(self.network_root):
            shutil.rmtree(self.network_root)
        os.makedirs(self.network_root)
        os.makedirs(config_path)
        self.ip_distributor = IpDistributor()
        self.need_embeded = need_embeded
        self.max_weight = 1000
        self.weight_matrix = None
        self.AS_device_map = {1: []}
        self.device_AS_map = {}
        self.net_seq = net_seq
        self.channel_num = 0
        self.MAX_WEIGHT = 32
        self.node_dst_map = {}

    def build_graph(self, num_nodes, real_world_topology, num_networks, sample_config_overrides, seed,
                    num_gateway_nodes, mun_acl_role, max_interface=4):  # 节点从0开始
        self.seed = seed
        self.bgp_semantics = BgpSemantics()
        self.G, self.facts = self.bgp_semantics.sample(num_nodes=num_nodes, real_world_topology=real_world_topology,
                                                       num_networks=num_networks,
                                                       predicate_semantics_sample_config_overrides=sample_config_overrides,
                                                       seed=seed, NUM_GATEWAY_NODES=num_gateway_nodes,
                                                       NUM_ACL_ROLE=mun_acl_role,
                                                       MAX_INTERFACE=max_interface)
        node_list = list(self.G.nodes())
        for n in node_list:
            if self.G.nodes[n]['type'] == 'network':
                AS_num = len(self.AS_device_map) + 1
                self.AS_device_map[AS_num] = [n]
                self.device_AS_map[n] = AS_num
                self.G.nodes[n]["device"] = BgpDevice(host_name=self.G.nodes[n]["label"],
                                                      device_type=Device_type.NETWORK, AS_num=AS_num)

            else:
                self.AS_device_map[1].append(n)
                self.device_AS_map[n] = 0
                if self.G.nodes[n]['type'] == 'route_reflector':
                    self.G.nodes[n]["device"] = BgpDevice(host_name=self.G.nodes[n]["label"],
                                                          device_type=Device_type.REFLECTOR, AS_num=1)
                elif self.G.nodes[n]['type'] == 'external':
                    self.G.nodes[n]["device"] = BgpDevice(host_name=self.G.nodes[n]["label"],
                                                          device_type=Device_type.EXTERNAL, AS_num=1)
                else:
                    self.G.nodes[n]["device"] = BgpDevice(host_name=self.G.nodes[n]["label"],
                                                          device_type=Device_type.ROUTER, AS_num=1)
        self.address_distribution()

    def drive_facts(self, predicate_semantics_sample_config_overrides):
        s = np.random.RandomState(seed=self.seed)

        def get_overrides(pred_s):
            if pred_s.predicate_name in predicate_semantics_sample_config_overrides.keys():
                return predicate_semantics_sample_config_overrides[
                    pred_s.predicate_name]  # {"n": choose_random([8, 10, 12], s)}
            return {}

        facts = []

        # 从计算的转发平面派生规范谓词
        for pred_s in self.bgp_semantics.predicate_semantics:
            config = self.bgp_semantics.sampling_config(pred_s, overrides=get_overrides(
                pred_s))  # config就是一个谓词抽样的个数 {"n": 10}
            derived = pred_s.sample(self.G, random=s, per_network=False, **config)

            if self.bgp_semantics.labeled_networks:
                for f in derived:
                    def network_constants_to_label(a):
                        if type(a) is Constant and a.name in self.bgp_semantics.network_mapping.keys(): return \
                            self.bgp_semantics.network_mapping[a.name]
                        return a

                    f.args = [network_constants_to_label(a) for a in f.args]
            facts += derived

        return facts

    def get_route_table(self):
        route_table = np.full((len(self.G.nodes), len(self.G.nodes)), -1)
        for node1, node2, data in self.G.edges(data=True):
            if "is_forwarding" in data.keys():
                for net in data['is_forwarding'].keys():
                    route_table[node1][net] = data['interface_info']['out']  # 出端口
            if data['type'] == 'network' and self.G.nodes[node1]['type'] == 'external':
                route_table[node1][node2] = data['interface_info']['out']
        np.savetxt(os.path.join(self.network_root, 'route_table.txt'), route_table, fmt='%d')
        return route_table

    def get_forward_table(self):
        self.forward_table = np.full((len(self.G.nodes), len(self.G.nodes)), -1)
        for node1, node2, data in self.G.edges(data=True):
            if "is_forwarding" in data.keys():
                for net in data['is_forwarding'].keys():
                    self.forward_table[node1][net] = node2  # 出端口
            if data['type'] == 'network' and self.G.nodes[node1]['type'] == 'external':
                self.forward_table[node1][node2] = node2
        np.savetxt(os.path.join(self.network_root, 'forward_table.txt'), self.forward_table, fmt='%d')
        return self.forward_table

    def random_data(self):
        old_forward_table = self.get_forward_table()
        num_edges = len(self.G.edges())

        update_record = {'ospf': [], 'bgp_route': []}
        while True:
            change_edges = random.sample(list(self.G.edges(data=True)), math.ceil(num_edges * 0.05))  # 采样20分之一的边改变
            for u, v, data in change_edges:
                if 'weight' in data:
                    old_value = {
                        'cost': data['weight']
                    }
                    weight_int = random.randint(1, self.MAX_WEIGHT)  # 随机权重整数
                    data['weight'] = weight_int
                    # update_record.append(
                    #     {
                    #         'type': 'ospf_weight',
                    #         'location': (u, v),
                    #         'old_value': old_value,
                    #         'new_value': {
                    #             'cost': weight_int
                    #         }
                    #     }
                    # )
                    update_record['ospf'].append([u, v, weight_int])

            external_nodes = [n for n in self.G.nodes() if self.G.nodes[n]["type"] == "external"]
            changed_external = random.sample(external_nodes, math.ceil(len(external_nodes) // 4))

            for external_node in changed_external:
                bgp_route = self.G.nodes[external_node]["bgp_route"]
                old_value = {
                    'local_preference': bgp_route.local_preference,
                    'med': bgp_route.med
                }

                bgp_route.local_preference = np.random.randint(0, 10) * 10
                bgp_route.med = np.random.randint(0, 10)

                # update_record.append({
                #     'type': 'bgp_route',
                #     'location': external_node,
                #     'old_value': old_value,
                #     'new_value': {
                #         'local_preference': bgp_route.local_preference,
                #         'med': bgp_route.med
                #     }
                # })
                update_record['bgp_route'].append([external_node, bgp_route.local_preference, bgp_route.med])
            compute_forwarding_state(self.G)
            new_forward_table = self.get_forward_table()
            if not np.array_equal(old_forward_table, new_forward_table):  # 如果转发平面未发生变化，则重新改变权重
                break
        # 刷新网络快照
        path_changed_node_pairs = {}
        net_node = [n for n in self.G.nodes if self.G.nodes[n]['type'] == 'network']
        router_node = [n for n in self.G.nodes if self.G.nodes[n]['type'] == 'router' or self.G.nodes[n][
            'type'] == 'route_reflector']  # router, route_reflector
        for i in router_node:  # 起始节点
            path_changed_node_pairs[i] = []
            for j in net_node:
                if not old_forward_table[i, j] == new_forward_table[i, j]:
                    path_changed_node_pairs[i].append(j)
            if len(path_changed_node_pairs[i]) == 0:  # 随机选择目的地节点，切changed=0
                dst = i
                while dst == i:
                    dst = random.sample(list(net_node), 1)[0]
                self.node_dst_map[i] = dst
            else:
                dst = random.sample(path_changed_node_pairs[i], 1)[0]
                self.node_dst_map[i] = dst

        return update_record

    def draw_grap(self):
        canvas0 = draw_graph(self.G)
        canvas0.print_figure(f'{self.network_root}/network_graph.png', dpi=300)

        canvas1 = draw_dev_graph(self.G)
        canvas1.print_figure(f'{self.network_root}/network_inf_graph.png', dpi=300)

        return

    def address_distribution(self):
        '''
        为所有设备的物理接口和逻辑接口分配网络
        :return:
        '''
        num_nodes = len(self.G.nodes())
        ospf_edges = [(src, dst) for src, dst in self.G.edges() if "weight" in self.G[src][dst].keys()]

        for node in self.G.nodes:  # 对逻辑接口进行赋值
            prefix, mask = self.ip_distributor.assign_address(Interface_type.Loopback)
            device = self.G.nodes[node]['device']
            device.add_interface(interface_type=Interface_type.Loopback, network_add=str(prefix), prefix=str(prefix),
                                 mask=mask, weight=0)

        visited = np.zeros((num_nodes, num_nodes), dtype=bool)
        for node1, node2 in ospf_edges:  # 对物理接口进行赋值  node1 -> node2  将这条边的权重作为node1的接口权重
            if visited[node1][node2]: continue

            prefix, mask = self.ip_distributor.assign_address(Interface_type.GigabitEthernet)
            ip1 = str(ipaddress.IPv4Address(int(prefix) + 1))
            ip2 = str(ipaddress.IPv4Address(int(prefix) + 2))

            # 在AS内，有权重的才是物理链路(除ebgp的边)
            device1 = self.G.nodes[node1]['device']
            weight1 = self.G[node1][node2].get('weight')
            interface_index1 = device1.add_interface(interface_type=Interface_type.GigabitEthernet, network_add=ip1,
                                                     prefix=str(prefix),
                                                     mask=mask, weight=weight1)

            device2 = self.G.nodes[node2]['device']
            weight2 = self.G[node2][node1].get('weight')
            interface_index2 = device2.add_interface(interface_type=Interface_type.GigabitEthernet, network_add=ip2,
                                                     prefix=str(prefix),
                                                     mask=mask, weight=weight2)

            self.G[node1][node2]['interface_info'] = {'out': interface_index1, 'in': interface_index2}  # 记录边连接的端口
            self.G[node2][node1]['interface_info'] = {'out': interface_index2, 'in': interface_index1}

            visited[node2][node1] = True

        for node1, node2 in ospf_edges:
            if 'acl_role' in self.G[node1][node2].keys():
                for acl in self.G[node1][node2]['acl_role']:
                    dst_add = self.G.nodes[acl.dst]['device'].loopback_add
                    acl.dst_ipv4 = IPv4Network(dst_add + '/32')
                    if acl.out_or_in == 0:
                        device1.add_acl_role(acl, self.G[node1][node2]['interface_info']['out'])
                    else:
                        device2.add_acl_role(acl, self.G[node1][node2]['interface_info']['in'])

        # 处理ebgp边
        visited = np.zeros((num_nodes, num_nodes), dtype=bool)
        ebgp_edges = [(src, dst) for src, dst in self.G.edges() if self.G[src][dst]["type"] == "ebgp"]
        for src, dst in ebgp_edges:
            if visited[src][dst]: continue
            prefix, mask = self.ip_distributor.assign_address(Interface_type.GigabitEthernet)
            ip1 = str(ipaddress.IPv4Address(int(prefix) + 1))
            ip2 = str(ipaddress.IPv4Address(int(prefix) + 2))
            interface_index1 = self.G.nodes[src]['device'].add_interface(interface_type=Interface_type.GigabitEthernet,
                                                                         network_add=ip1,
                                                                         prefix=str(prefix),
                                                                         mask=mask, weight=1)

            interface_index2 = self.G.nodes[dst]['device'].add_interface(interface_type=Interface_type.GigabitEthernet,
                                                                         network_add=ip2,
                                                                         prefix=str(prefix),
                                                                         mask=mask, weight=1)

            self.G[src][dst]['interface_info'] = {'out': interface_index1, 'in': interface_index2}  # 记录边连接的端口
            self.G[dst][src]['interface_info'] = {'out': interface_index2, 'in': interface_index1}

            visited[dst][src] = True

        network_edges = [(src, dst) for src, dst in self.G.edges() if self.G[src][dst]["type"] == "network"]
        for src, dst in network_edges:
            if visited[src][dst]: continue
            prefix, mask = self.ip_distributor.assign_address(Interface_type.GigabitEthernet)
            ip1 = str(ipaddress.IPv4Address(int(prefix) + 1))
            ip2 = str(ipaddress.IPv4Address(int(prefix) + 2))
            interface_index1 = self.G.nodes[src]['device'].add_interface(interface_type=Interface_type.GigabitEthernet,
                                                                         network_add=ip1,
                                                                         prefix=str(prefix),
                                                                         mask=mask)
            self.G.nodes[src]['device'].add_ebgp_neighbor(ip2, self.G.nodes[dst]['device'].AS_num,
                                                          True if self.G.nodes[dst][
                                                                      'type'] == 'route_reflector' else False)

            interface_index2 = self.G.nodes[dst]['device'].add_interface(interface_type=Interface_type.GigabitEthernet,
                                                                         network_add=ip2,
                                                                         prefix=str(prefix),
                                                                         mask=mask)
            self.G.nodes[dst]['device'].add_ebgp_neighbor(ip1, self.G.nodes[src]['device'].AS_num,
                                                          True if self.G.nodes[src][
                                                                      'type'] == 'route_reflector' else False)
            self.G[src][dst]['interface_info'] = {'out': interface_index1, 'in': interface_index2}  # 记录边连接的端口
            self.G[dst][src]['interface_info'] = {'out': interface_index2, 'in': interface_index1}

            visited[dst][src] = True

        visited = np.zeros((num_nodes, num_nodes), dtype=bool)
        ibgp_edges = [(src, dst) for src, dst in self.G.edges() if
                      self.G[src][dst]["type"] == "ibgp" or self.G[src][dst]["type"] == "ebgp"]
        for src, dst in ibgp_edges:
            if visited[src][dst]: continue
            if src == dst:
                print(f"{src}-{dst}")
            self.G.nodes[src]['device'].add_ibgp_neighbor(self.G.nodes[dst]['device'].loopback_add,
                                                          self.G.nodes[dst]['device'].AS_num, True if self.G.nodes[src][
                                                                                                          'type'] == 'route_reflector' else False)
            self.G.nodes[dst]['device'].add_ibgp_neighbor(self.G.nodes[src]['device'].loopback_add,
                                                          self.G.nodes[src]['device'].AS_num, True if self.G.nodes[dst][
                                                                                                          'type'] == 'route_reflector' else False)
            visited[dst][src] = True

        for node in self.G.nodes:
            if self.G.nodes[node]['type'] == 'external':
                bgp_route = self.G.nodes[node]['bgp_route']
                self.G.nodes[node]['device'].add_bgp_route(
                    destination=self.G.nodes[bgp_route.destination]['device'].loopback_add,
                    local_preference=bgp_route.local_preference,
                    med=bgp_route.med)

    def gen_config(self):
        for node in self.G.nodes:
            device = self.G.nodes[node]['device']
            device.make_config_file(self.network_root, False)

    def channel_register(self):
        delay_leval = [['1ms', '10ms', '20ms'], ['40ms', '80ms', '150ms'], ['200ms', '400ms', '1s']]  # 低，中，高延迟
        data_rate_leval = [['20kbps', '100kbps', '500kbps'], ['1Mbps', '5Mbps', '25Mbps'],
                           ['100Mbps', '400Mbps', '1Gbps']]  # 低，中，高速
        per_leval = [['0.001', '0.004', '0.08'], ['0.01', '0.04', '0.08'], ['0.1', '0.2', '0.4']]
        random_choose = [delay_leval, data_rate_leval, per_leval]
        with open("resource/edge_attribute_tem.txt", 'r') as file:
            channel_attribute_tem = file.read()

        channel_name = f"C{self.channel_num}"
        channel_value = []
        for data_levals in random_choose:
            leval = random.choice(data_levals)
            value = random.choice(leval)
            channel_value.append(value)
        channel_attribute_str = channel_attribute_tem.format(channel_name, channel_value[0], channel_value[1],
                                                             channel_value[2])
        self.channel_num += 1
        channel_attribute = {
            'delay': channel_value[0],
            'datarate': channel_value[1],
            'per': channel_value[2]
        }
        return channel_name, channel_attribute_str, channel_attribute

    def gen_ned_file(self):
        edge_tem = 'rte[{}].port[{}] <--> {} <--> rte[{}].port[{}];'  # 中间的channel要大变
        with open("resource/ned_template.txt", 'r') as ned_file:
            ned_template = ned_file.read()

        channel_str_list = []  # 初始化channel
        edge_str_list = []  # 边连接，携带channel变量
        num_nodes = len(self.G.nodes())
        visited = np.zeros((num_nodes, num_nodes), dtype=bool)
        columns = ['node1', 'node2', 'delay', 'datarate', 'per']
        edge_df = pd.DataFrame(columns=columns)
        for node1, node2, data in self.G.edges(data=True):  # 遍历物理边
            if self.G.edges[node1, node2]['type'] == 'ibgp' and 'weight' not in self.G.edges[
                node1, node2].keys(): continue  # 逻辑边
            if visited[node1][node2]: continue
            channel_name, channel_attribute_str, channel_attribute = self.channel_register()

            channel_str_list.append(channel_attribute_str)
            edge_str_list.append(
                edge_tem.format(node1, data['interface_info']['out'], channel_name, node2,
                                data['interface_info']['in']))
            channel_attribute['node1'] = node1
            channel_attribute['node2'] = node2
            edge_df.loc[len(edge_df)] = channel_attribute
            visited[node2][node1] = True
        # channel 注册
        channel_str = "\n        ".join(channel_str_list)
        # 边
        edge_str = '\n        '.join(edge_str_list)
        ned_str = ned_template.format(str(self.net_seq), channel_str, str(num_nodes), edge_str)

        os.makedirs(os.path.join("omnet_file", "networks"), exist_ok=True)
        with open(os.path.join("omnet_file", "networks", f"Myned{str(self.net_seq)}.ned"), 'w') as f:
            f.write(ned_str)

        edge_df.to_csv(os.path.join(self.network_root, "topology.csv"), index=False)

    def gen_ini_file(self, flow_num):
        route_tabel = self.get_route_table().astype(str)
        table_str = '\\n'.join([' '.join(line) for line in route_tabel])
        with open("resource/ini_template.txt", 'r') as ini_file:
            ini_format = ini_file.read()

        node_info = []
        flow_count = 0
        for n in self.G.nodes:
            node_info.append(f"Myned{str(self.net_seq)}.rte[{n}].appType = \"{self.G.nodes[n]['type']}\"")
            node_info.append(f"Myned{str(self.net_seq)}.rte[{n}].address = {n}")
            if n in self.node_dst_map and flow_count < flow_num:
                node_info.append(f"Myned{str(self.net_seq)}.rte[{n}].destAddresses = \"{self.node_dst_map[n]}\"")
                flow_count += 1
            else:
                node_info.append(f"Myned{str(self.net_seq)}.rte[{n}].destAddresses = \"{-1}\"")

        node_type_str = '\n'.join(node_info)
        ini_str = ini_format.format(str(self.net_seq), str(self.net_seq), str(self.net_seq), table_str,
                                    str(self.net_seq), str(self.net_seq), node_type_str)
        os.makedirs(os.path.join("omnet_file", "ini_dir"), exist_ok=True)
        with open(os.path.join("omnet_file", "ini_dir", f"omnetpp{str(self.net_seq)}.ini"), "w") as ini_file:
            ini_file.write(ini_str)

    def get_edges(self):
        edge_list = []
        for edge in self.G.edges():
            device1 = self.G.nodes[edge[0]]['device']
            device2 = self.G.nodes[edge[1]]['device']
            edge_list.append(f"[{device1.host_name},{device2.host_name}]")
        return ",".join(edge_list)

    def save_pyg(self):
        edge_index = torch.tensor(list(self.G.edges())).t().contiguous()
        x = torch.arange(self.G.number_of_nodes()).view(-1, 1).float()  # 假设节点特征为节点编号
        data = Data(x=x, edge_index=edge_index)
        torch.save(data, os.path.join(str(self.network_root), 'graph_edges.pt'))

    def get_path_from_start(self, start):
        while True:
            dest = random.sample(list(self.G.nodes), 1)[0]
            if not start == dest:
                if not self.G.has_edge(start, dest):  # 不是邻居节点
                    break
        path = nx.dijkstra_path(self.G, source=start, target=dest, weight='weight')
        return dest, path

    def get_path(self, start, dst):
        path = nx.dijkstra_path(self.G, source=start, target=dst, weight='weight')
        return path

    def record_edges(self, begin, edge_record: []):  # 用于将小图拼接为大图时，构造data.edge_index，向edge_record添加边
        edges = self.G.edges()
        for start, end in edges:
            edge_record[0].append(begin * 20 + start)
            edge_record[1].append(begin * 20 + end)

    def save_pkl(self):
        edges = self.G.edges()
        n = len(edges)
        start_nodes = []
        end_nodes = []

        for start, end in edges:
            start_nodes.append(start)
            end_nodes.append(end)

        edges = [start_nodes, end_nodes]
        # print(edges)

        with open(f'{self.network_root}/graph_edges.pkl', 'wb') as f:
            pickle.dump(edges, f)

        with open(f'{self.network_root}/net_model.pkl', 'wb') as f:
            pickle.dump(self.G, f)

    def load_pyg(self):
        data = torch.load(os.path.join(str(self.network_root), 'graph.pt'))
        # print(data.edge_index)

    def get_node_presentation(self, node):
        node_presentation = []
        connection_form = f"this {self.G.nodes[node]['device'].host_name}: the weights of the edges connecting this node to its neighbor nodes are as follows："
        edge_form = "to node {}, the weight is {}"
        node_presentation.append(connection_form)
        for neighbor in self.G.neighbors(node):
            node_presentation.append(edge_form.format(self.G.nodes[neighbor]['device'].host_name,
                                                      self.weight_matrix[node][neighbor]))
        return "\n".join(node_presentation)

    def get_node_presentation_type2(self, node):
        node_presentation = []
        connection_form = f"this {self.G.nodes[node]['device'].host_name}: the shortest forwarding path from this node to other nodes in the network are as follows："
        path_format = "to node {}, the path is {}"
        other_nodes = [i for i in self.G.nodes() if i != node]
        node_presentation.append(connection_form)
        for i in other_nodes:
            path = nx.dijkstra_path(self.G, node, i, weight='weight')
            path_str = ' -> '.join([self.G.nodes[j]['device'].host_name for j in path])
            node_presentation.append(path_format.format(self.G.nodes[i]['device'].host_name, path_str))
        return "\n".join(node_presentation)

    def get_query_and_answer(self, cur_node, query_type: QueryType):
        if query_type == QueryType.ROUTE.value:
            dest, path = self.get_path_from_start(cur_node)
            query = f"What are the shortest OSPF paths from {self.G.nodes[cur_node]['device'].host_name} and {self.G.nodes[dest]['device'].host_name}"
            hostname_path = [self.G.nodes[node]['device'].host_name for node in path]
            answer = ' - '.join(hostname_path)
            return query, answer, json.dumps(hostname_path)
        elif query_type == QueryType.INTERFACE.value:
            this_device = self.G.nodes[cur_node]['device']
            query = f"List all the interface info of the {this_device.host_name} device"
            answer_list = []
            label_list = []
            for interface in this_device.interface_list:
                inter_str = f"{interface.get_interface_name()}:Ipv4 address is {interface.address}，mask is {interface.mask}，OSPF cost is {interface.weight}"
                answer_list.append(inter_str)
                label_list.append({
                    'interface': interface.get_interface_name(),
                    'address': str(interface.address),
                    'mask': str(interface.mask)
                })
            answer = '\n'.join(answer_list)
            return query, answer, json.dumps(label_list)
        elif query_type == QueryType.NETWORK.value:
            this_device = self.G.nodes[cur_node]['device']
            query = f"List all the Ipv4 network of the {this_device.host_name} device"
            answer_list = []
            label_list = []
            answer_list.append(f"there are {len(this_device.networks)} networks:")
            for network in this_device.networks:
                answer_list.append(f"{network['prefix']} {network['mask']}")
                label_list.append({
                    'prefix': str(network['prefix']),
                    'mask': str(network['mask'])
                })
            answer = '\n'.join(answer_list)
            return query, answer, json.dumps(label_list)
        else:
            raise Exception("不存在的query_type!")


if __name__ == '__main__':
    builder = BgpBuilder(10, 0)
    builder.build_graph()
    builder.draw_graph()
    builder.gen_ospf_config()
    builder.save_pkl()
    # builder.load_pyg()
    builder.save_pkl()
