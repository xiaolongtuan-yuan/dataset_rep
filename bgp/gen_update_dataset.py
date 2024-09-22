# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/19 20:14
@Auth ： xiaolongtuan
@File ：main.py
"""
import math
import os
import pickle

import pandas as pd
from tqdm import tqdm

from bgp_builder import BgpBuilder
from bgp_semantics import BgpSemantics
import numpy as np

from utils.routing_service import get_routing_sim


def choose_random(arr, s=None):
    s = np.random if s is None else s
    idx = s.randint(0, len(arr))
    return arr[idx]


total_account = 2


def add_qos_constraint(row):
    qos_constraint = "Now we need to ensure that the end-to-end delay of the traffic from node c{:.0f} to node c{:.0f} is less than {}ms, the average jitter is less than {}ms, and the packet loss rate is less than {:.2f}%".format(
        row['src'], row['dst'], math.ceil(row['avgDelay'] * 10) * 100, math.ceil(row['avgJitter'] * 100) * 10,
                                row['pkgLossRate'] * 100)
    return pd.Series([qos_constraint], index=['qos_constraint'])


if __name__ == '__main__':
    seed = os.getpid()
    pbar = tqdm(total=total_account)

    s = np.random.RandomState(seed=seed)
    real_world_topology = False
    # num_networks = choose_random(list(range(4, 5)), s)  # 外网个数
    num_networks = 3  # 7 6 4 3
    num_gateway_nodes = 2  # 每个外网连接网关个数 7 5 3 2
    mun_acl_role = 0  # ACL个数
    # num_nodes = choose_random(range(4, 10), s)  # ospf域内节点个数
    num_nodes = 5  # 44 24 19 5
    sample_config_overrides = {
        "fwd": {
            # "n": choose_random([1, 2], s)  # 每个外网抽样的fwd个数
            "n": 2  # 7 5 3 2
        },
        "reachable": {
            # "n": choose_random([2, 3], s)  # 总个数
            "n": 3  # 6 5 4 3
        },
        "trafficIsolation": {
            "n": 1  # 4 3 2 1
        }
    }
    data_dir = 'update_data_14'
    flow_num = 4  # 20 12 9 4

    for i in range(total_account):
        net_id = i
        builder = BgpBuilder(net_seq=net_id, network_root=f'{data_dir}/{i}')
        builder.build_graph(num_nodes, real_world_topology, num_networks, sample_config_overrides, seed,
                            num_gateway_nodes, mun_acl_role, max_interface=8)

        builder.gen_config()  # 收集旧的配置文件

        update_record = builder.random_data()  # 更新配置参数

        facts = builder.drive_facts(sample_config_overrides)  # 收集新配置的策略
        with open(os.path.join(f'{data_dir}/{i}', "facts.pkl"), 'wb') as f:
            pickle.dump(facts, f)
        with open(os.path.join(f'{data_dir}/{i}', "update_record.pkl"), 'wb') as f:
            pickle.dump(update_record, f)

        # 模拟新网络性能
        builder.gen_ned_file()
        builder.gen_ini_file(flow_num=flow_num)

        ned_path = os.path.join("omnet_file", "networks", f"Myned{str(net_id)}.ned")
        ini_path = os.path.join("omnet_file", "ini_dir", f"omnetpp{str(net_id)}.ini")

        qos_dic_list = get_routing_sim(net_id=str(net_id), ned_path=ned_path, ini_path=ini_path)
        qos_df = pd.DataFrame(qos_dic_list)

        qos_constraint_values = qos_df.apply(add_qos_constraint, axis=1)
        qos_df['qos_constraint'] = qos_constraint_values
        qos_df.to_json(f"{data_dir}/{str(net_id)}/end2end_qos.jsonl", orient="records", lines=True)
        pbar.update(1)
    print('finished')
    # 使用fact构造转发需求
