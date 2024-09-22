# -*- coding: utf-8 -*-
"""
@Time ： 2024/5/15 13:56
@Auth ： xiaolongtuan
@File ：data_analyse.py
"""
import argparse
import os
import subprocess
import shutil
import pandas as pd
import re


def delete_all_in_directory(directory_path):
    """
    删除指定目录下的所有文件和文件夹。

    参数:
    directory_path (str): 目标目录的路径
    """
    if not os.path.exists(directory_path):
        print(f"目录 {directory_path} 不存在")
        return

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 删除文件或符号链接
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 删除目录及其内容

            print("清理旧的data log")
        except Exception as e:
            print(f"删除 {file_path} 时出错: {e}")

def extract_end_to_end_delay_mean(filename):
    end_to_end_delay_means = []
    end_to_end_jitter_means = []
    package_sum = 0
    package_receive_sum = 0

    with open(filename, 'r') as file:
        file_str = file.read()
        # 使用正则表达式找到包含 endToEndDelay:mean 的行
        delay_matchs = re.findall(r'endToEndDelay:mean (\d+\.\d+)', file_str)
        jitter_matchs = re.findall(r'endToEndJitter:mean (\d+\.\d+)', file_str)
        package_nums = re.findall(r'sendCount:count (\d+)', file_str)
        package_receives = re.findall(r'receiveCount:count (\d+)', file_str)

        if delay_matchs:
            end_to_end_delay_means = [float(match) for match in delay_matchs]
        if jitter_matchs:
            end_to_end_jitter_means = [float(match) for match in jitter_matchs]
        if package_nums:
            package_sum = sum([int(match) for match in package_nums])
        if package_receives:
            package_receive_sum = sum([int(match) for match in package_receives])

        packet_length = re.search(r'packetLength\s+(\d+byte)', file_str)
        sendIaTime = re.search(r'sendIaTime (exponential\((.*?)\))', file_str)

        if packet_length and sendIaTime:
            packet_length = packet_length.group(1)
            sendIaTime = sendIaTime.group(1)
        else:
            print(packet_length)
            print(sendIaTime)
            raise Exception("未统计packet_length和sendIaTime")

    return packet_length, sendIaTime, sum(end_to_end_delay_means) / len(end_to_end_delay_means), sum(
        end_to_end_jitter_means) / len(
        end_to_end_jitter_means), (package_sum - package_receive_sum) / package_sum


def result_statistic(result_dir):
    columns = ['net_id', 'packet_length', 'sendIaTime', 'end_to_end_delay', 'end_to_end_jitter', 'package_loss_rate']
    df = pd.DataFrame(columns=columns, index=None)
    net_id_list = collect_integer_named_folders(result_dir)
    for i in net_id_list:
        data_path = f"results/{i}/General-#0.sca"
        packet_length, sendIaTime, end_to_end_delay_means, end_to_end_jitter_means, plr = extract_end_to_end_delay_mean(
            data_path)
        entry = {
            'net_id': i,
            'packet_length': packet_length,
            'sendIaTime': sendIaTime,
            'end_to_end_delay': end_to_end_delay_means,
            'end_to_end_jitter': end_to_end_jitter_means,
            'package_loss_rate': plr
        }
        df.loc[len(df)] = entry
    return df

def analyse_result_dir(result_dir, output):
    # 将所有网络的qos放在一张表里了
    df = result_statistic(result_dir)
    df.to_csv(os.path.join(output, "sim_result.csv"), index=False)

def analyse_datalog_dir(data_log_dir, output_dir):
    # 每个网络单独一张表
    net_id_list = collect_integer_named_folders(data_log_dir)
    for i in net_id_list:
        data_log_path =f'logs/{i}/pkg_log.txt'
        df = analyse_datalog_file(data_log_path)
        df.to_csv(os.path.join(output_dir,f'net{i}_path_qos.csv'), index=False)

    print("finished")

def analyse_datalog_file(file_path):
    file_path = file_path
    # 初始化数据存储
    data = []

    # 读取文件内容并解析
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 解析每一行并组织数据
    send_pkg_dict = {}
    node_pair_map = {}
    for line in lines:
        line = line.strip()
        if line.startswith("src:"):
            parts = line.split(", ")
            src = int(parts[0].split(": ")[1])
            dst = int(parts[1].split(": ")[1])
            if not (src, dst) in node_pair_map:
                node_pair_map[(src, dst)] = {}

            if 'sendPkgCount' in parts[2]:
                sendPkgCount = int(parts[2].split(": ")[1])
                node_pair_map[(src, dst)]['sendPkgCount'] = sendPkgCount

            else:
                avgDelay = float(parts[2].split(": ")[1])
                node_pair_map[(src, dst)]['avgDelay'] = avgDelay
                avgJitter = float(parts[3].split(": ")[1])
                node_pair_map[(src, dst)]['avgJitter'] = avgJitter

                pkgReceiveCount = int(parts[4].split(": ")[1])
                node_pair_map[(src, dst)]['pkgReceiveCount'] = pkgReceiveCount

    for k, v in node_pair_map.items():
        src, dst = k

        if sendPkgCount is not None:
            data.append({
                'src': src,
                'dst': dst,
                'avgDelay': v['avgDelay'],
                'avgJitter': v['avgJitter'],
                'pkgLossRate': v['pkgReceiveCount'] / v['sendPkgCount']
            })

    # 创建 DataFrame
    df = pd.DataFrame(data, columns=['src', 'dst', 'avgDelay', 'avgJitter', 'pkgLossRate'])
    # 打印 DataFrame
    return df

def collect_integer_named_folders(directory):
    # 正则表达式匹配整数命名的文件夹
    pattern = re.compile(r'^\d+$')

    # 收集文件夹名称列表
    integer_named_folders = []

    # 遍历目录中的所有文件和文件夹
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path) and pattern.match(item):
            integer_named_folders.append(item)
    integer_named_folders.sort()
    return integer_named_folders


def run_sim(ini_dir):
    net_id_list = collect_integer_named_folders(ini_dir)
    pattern = re.compile(r'omnetpp\d+\.ini')
    integer_named_folders = []
    # 遍历目录中的所有文件和文件夹
    for item in os.listdir(ini_dir):
        if pattern.match(item):
            integer_named_folders.append(item)
    integer_named_folders.sort()

    for item in integer_named_folders:
        ini_file = os.path.join(ini_dir, item)
        command = f'./routing -r 0 -m -u Cmdenv -n .:../inet4.5/src -l ../inet4.5/src/INET {ini_file}'
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f'sim {item} finished!')
            continue
        else:
            print(f'sim {item} ERROR! {result}')
            raise Exception("Command failed with error")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="示例程序描述")
    parser.add_argument('--ini_dir', type=str, required=False, help='ini目录路径', default='./ini_dir')
    parser.add_argument('--result_dir', type=str, required=False, help='模拟结果目录路径', default='./results')
    parser.add_argument('--data_log_dir', type=str, required=False, help='模拟结果datalog目录路径', default='./logs')
    parser.add_argument('--output_dir', type=str, required=False, help='输出csv目录路径', default='./net_path_df')
    args = parser.parse_args()

    delete_all_in_directory(args.data_log_dir)
    # 运行模拟
    run_sim(args.ini_dir)
    analyse_datalog_dir(args.data_log_dir, args.output_dir)


