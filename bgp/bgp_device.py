# -*- coding: utf-8 -*-
"""
@Time ： 2024/1/10 11:19
@Auth ： xiaolongtuan
@File ：Device.py
"""
import os.path
from enum import Enum, auto
import ipaddress

import torch
from transformers import BertTokenizer, BertModel

from bgp_semantics import AclRole


class Interface_type(Enum):
    Loopback = auto()
    GigabitEthernet = auto()


class Device_type(Enum):
    ROUTER = auto()
    EXTERNAL = auto()
    REFLECTOR = auto()
    NETWORK = auto()


need_embeded = False
if need_embeded:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "google-bert/bert-base-uncased"
    local_path = './bert-base-uncased/models--google-bert--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594'
    tokenizer = BertTokenizer.from_pretrained(local_path)
    bert_model = BertModel.from_pretrained(local_path).to(device)


def bert_encoder(config):
    inputs = tokenizer(config, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # 使用BERT模型对编码后的文本进行forward传递，获取输出
    with torch.no_grad():
        outputs = bert_model(**inputs)

        # 获取最后一层的隐藏状态
        last_hidden_state = outputs.last_hidden_state

        # 获取 [CLS] 标记的隐藏状态作为整个句子的表示
        sentence_representation = last_hidden_state[:, 0, :].view(-1)
        return sentence_representation


class Config_Template:
    config_template = "interface {}{}\n ip address {} {}\n negotiation auto\n ip ospf cost {}"
    config_template_without_weight = "interface {}{}\n ip address {} {}\n negotiation auto"
    loop_config_template = "interface {}{}\n ip address {} {}\n negotiation auto"
    BGP_template = "router bgp {}"
    reflector_template = " bgp cluster-id {}"
    BGP_network_template = " network {} mask {}"
    BGP_neighbor_template = " neighbor {} remote-as {}"
    BGP_reflector_neighbor_template = " neighbor {} route-reflector-client\n neighbor {} next-hop-self"
    BGP_ibgp_neighbor_template = " neighbor {} remote-as {}\n neighbor {} update-source Loopback0"
    BGP_route_map_net = "route-map MODIFY_NETWORK permit 10\n match ip address {}"
    BGP_route_map_use = " neighbor {} route-map MODIFY_NETWORK out"
    OSPF_process = 'router ospf {}'
    OSPF_network_template = ' network {} {} area 0'
    ACL_title = 'access-list {}'
    ACL_iterm = ' deny ip any {} {}'
    ACL_end = ' permit ip any any'
    ACL_apply = ' ip access-group {} {}'


class Interface:
    def __init__(self, interface_type, seq: int, address: str, mask: str, weight: int):
        self.interface_type = interface_type
        self.seq = seq
        self.address = address
        self.mask = mask
        self.weight = weight

    def gen_interface_config(self, interface_acl_map):
        interface_config_list = []
        if self.interface_type == Interface_type.GigabitEthernet:
            if self.weight:
                interface_config_list.append(Config_Template.config_template.format(self.interface_type.name, self.seq, self.address,
                                                              self.mask,
                                                              self.weight))
            else:  # 没有ospf cost
                interface_config_list.append(Config_Template.config_template_without_weight.format(self.interface_type.name, self.seq,
                                                                             self.address, self.mask))
            if self.seq in interface_acl_map.keys():
                if 'out' in interface_acl_map[self.seq].keys():
                    interface_config_list.append(Config_Template.ACL_apply.format(str(100 + self.seq + 0), 'out'))
                if 'in' in interface_acl_map[self.seq].keys():
                    interface_config_list.append(Config_Template.ACL_apply.format(str(100 + self.seq + 1), 'in'))
        elif self.interface_type == Interface_type.Loopback:
            interface_config_list.append(Config_Template.loop_config_template.format(self.interface_type.name, 0, self.address,
                                                               self.mask))
        else:
            raise Exception('接口类型错误！')

        return '\n'.join(interface_config_list)

    def get_interface_name(self):
        if self.interface_type == Interface_type.GigabitEthernet:
            return f"{self.interface_type.name}{self.seq}/0"
        elif self.interface_type == Interface_type.Loopback:
            return f"{self.interface_type.name}0/0"


class BgpDevice:
    def __init__(self, device_type: Device_type, AS_num: int, host_name="", network_root='', cluster_id=100):
        self.network_root = network_root
        self.device_type = device_type
        self.AS_num = AS_num
        self.host_name = host_name
        self.interface_list = []
        self.interface_index = 0
        self.networks = []
        self.ebgp_neighbors = []
        self.ibgp_neighbors = []
        self.loopback_add = ''
        self.cluster_id = cluster_id
        self.interface_acl_map = {}

    def make_config_file(self, network_root, need_embeded=False):
        self.network_root = network_root
        basic_config = []
        basic_config.append(f"hostname {self.host_name}")
        # 接口配置
        for interface in self.interface_list:
            basic_config.append(interface.gen_interface_config(self.interface_acl_map))
        config = "\n!\n".join(basic_config) + "\n!"

        # ospf 网络宣告，只宣告有ospf cost的接口IP地址及回环
        if not self.device_type == Device_type.NETWORK:
            ospf_network_list = []
            ospf_network_list.append(Config_Template.OSPF_process.format(self.AS_num))
            for interface in self.interface_list:  # 精准宣告 这里之前没有宣告lp0，有bug
                ospf_network_list.append(
                    Config_Template.OSPF_network_template.format(str(interface.address), '0.0.0.0'))
            OSPF_config = "\n".join(ospf_network_list)
            basic_config.append(OSPF_config)

        # route-map定义
        if self.device_type == Device_type.EXTERNAL:
            basic_config.append(self.route_map_str)

        # 定义ACL规则
        acl_define_list = []
        for seq, acl_role_double in self.interface_acl_map.items():
            if 'out' in acl_role_double.keys():
                acl_define_list.append(Config_Template.ACL_title.format(str(100 + seq + 0)))  # 这个序号等于 interface seq + 100
                for acl in acl_role_double['out']:
                    acl_define_list.append(Config_Template.ACL_iterm.format(str(acl.dst_ipv4.network_address), str(acl.dst_ipv4.netmask)))
                acl_define_list.append(Config_Template.ACL_end)
            if 'in' in acl_role_double.keys():
                acl_define_list.append(
                    Config_Template.ACL_title.format(str(100 + seq + 1)))  # 这个序号等于 interface seq + 100 + 1
                for acl in acl_role_double['in']:
                    acl_define_list.append(
                        Config_Template.ACL_iterm.format(str(acl.dst_ipv4.network_address), str(acl.dst_ipv4.netmask)))
                acl_define_list.append(Config_Template.ACL_end)
        acl_define = '\n'.join(acl_define_list)
        basic_config.append(acl_define)

        # BGP配置
        bgp_config_list = []
        bgp_config_list.append(Config_Template.BGP_template.format(self.AS_num))
        if self.device_type == Device_type.REFLECTOR:  # 配置路由反射器
            bgp_config_list.append(Config_Template.reflector_template.format(self.cluster_id))

        # 建立邻居关系
        for neighbor in self.ebgp_neighbors:  # 用相应的物理接口
            bgp_config_list.append(
                Config_Template.BGP_neighbor_template.format(neighbor['ip_add'], neighbor["as_num"]))
            if self.device_type == Device_type.REFLECTOR:
                bgp_config_list.append(
                    Config_Template.BGP_reflector_neighbor_template.format(neighbor['ip_add'], neighbor['ip_add']))

        for neighbor in self.ibgp_neighbors:  # 用lp接口
            bgp_config_list.append(
                Config_Template.BGP_ibgp_neighbor_template.format(neighbor['ip_add'], neighbor["as_num"],
                                                                  neighbor['ip_add'])
            )
            if self.device_type == Device_type.REFLECTOR:
                bgp_config_list.append(
                    Config_Template.BGP_reflector_neighbor_template.format(neighbor['ip_add'], neighbor['ip_add']))

        # 宣告网络
        for network in self.networks:  # 只宣告lp0
            bgp_config_list.append(Config_Template.BGP_network_template.format(network['prefix'], network['mask']))

        # 应用route-map
        if self.device_type == Device_type.EXTERNAL:
            bgp_config_list.append(Config_Template.BGP_route_map_use.format(self.destination))
        bgp_config = "\n".join(bgp_config_list)
        basic_config.append(bgp_config)

        # 整合配置内容并保存
        config = "\n!\n".join(basic_config) + "\n!"
        self.config = config
        if need_embeded:
            self.embeded_config = bert_encoder(config).cpu()
            torch.cuda.empty_cache()
        with open(os.path.join(str(self.network_root), "configs", f"{str(self.device_type.name)}_{self.host_name}.cfg"),
                  "w") as config_file:
            config_file.write(config)
            # print(f"创建{self.host_name}配置文件")
            return config
        return ''

    def add_interface(self, interface_type: Interface_type, network_add: str, prefix: str, mask: str,
                      weight: int = None):
        interface = Interface(interface_type, self.interface_index, network_add, mask, weight)
        self.interface_index += 1
        self.interface_list.append(interface)

        if interface_type == Interface_type.Loopback:
            self.loopback_add = network_add
            self.networks.append({
                "prefix": prefix,
                "mask": mask
            })  # 记录要宣告的网络
        return interface.seq

    def add_acl_role(self, acl_role:AclRole, interface_index:int):
        if not interface_index in self.interface_acl_map.keys():
            if acl_role.out_or_in == 0:
                self.interface_acl_map[interface_index] = {'out': [acl_role]}
            else:
                self.interface_acl_map[interface_index] = {'in': [acl_role]}
        else:
            if acl_role.out_or_in == 0:
                if 'out' in self.interface_acl_map[interface_index].keys():
                    self.interface_acl_map[interface_index]['out'].append(acl_role)
                else:
                    self.interface_acl_map[interface_index]['out'] = [acl_role]
            else:
                if 'in' in self.interface_acl_map[interface_index].keys():
                    self.interface_acl_map[interface_index]['in'].append(acl_role)
                else:
                    self.interface_acl_map[interface_index]['in'] = [acl_role]


    def add_ebgp_neighbor(self, ip_add, as_num, is_reflector):
        self.ebgp_neighbors.append({
            "ip_add": ip_add,
            "as_num": as_num,
            "is_reflector": is_reflector
        })

    def add_ibgp_neighbor(self, ip_add, as_num, is_reflector):
        self.ibgp_neighbors.append({
            "ip_add": ip_add,
            "as_num": as_num,
            "is_reflector": is_reflector
        })

    def add_bgp_route(self, destination, local_preference=None, origin=None, med=None):
        if not self.device_type == Device_type.EXTERNAL:
            raise Exception("只有external可以添加bgp-route")
        else:
            self.destination = destination
            route_map_str = []
            route_map_str.append(Config_Template.BGP_route_map_net.format(destination))
            if local_preference:
                route_map_str.append(f' set local-preference {local_preference}')
            route_map_str.append(f' set as-path prepend {self.AS_num}')
            if origin:
                route_map_str.append(f' set origin {origin}')
            if med:
                route_map_str.append(f' set metric {med}')
            self.route_map_str = '\n'.join(route_map_str)
