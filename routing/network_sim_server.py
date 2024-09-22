# -*- coding: utf-8 -*-
"""
@Time ： 2024/5/25 19:25
@Auth ： xiaolongtuan
@File ：network_sim_server.py
"""
import logging
import os.path
import re
import subprocess
import shutil

import pandas as pd
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from starlette.responses import JSONResponse
import asyncio
from typing import List


def setup_logging():
    # 创建一个日志器
    logger = logging.getLogger("sim_logger")
    logger.setLevel(logging.INFO)  # 设置日志级别

    # 创建一个控制台处理器并设置级别
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)

    # 创建一个日志格式器
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    # 将处理器添加到日志器
    logger.addHandler(handler)
    return logger


app = FastAPI()
logger = setup_logging()

# 用于控制并发执行数量的信号量
semaphore = asyncio.Semaphore(10)


async def save_file(file: UploadFile, upload_path: str):
    with open(upload_path, "wb") as f:
        content = await file.read()
        f.write(content)
    return


def extract_unique_integer(string):
    # 使用正则表达式查找字符串中的所有整数
    numbers = re.findall(r'\d+', string)

    # 确保只有一个整数
    if len(numbers) == 1:
        return int(numbers[0])
    else:
        raise ValueError("The string does not contain exactly one unique integer")


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
        try:
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
        except Exception as e:
            logger.error(f"analyse_datalog_file: {e}")
    try:
        for k, v in node_pair_map.items():
            src, dst = k

            if 'pkgReceiveCount' in v:
                data.append({
                    'src': src,
                    'dst': dst,
                    'avgDelay': v['avgDelay'],
                    'avgJitter': v['avgJitter'],
                    'pkgLossRate': 1 - (v['pkgReceiveCount'] / v['sendPkgCount'])
                })
    except Exception as e:
        logger.error(f"analyse_datalog_file: {e}")

    # 创建 DataFrame
    df = pd.DataFrame(data, columns=['src', 'dst', 'avgDelay', 'avgJitter', 'pkgLossRate'])
    json_data = df.to_json(orient="records")
    # 打印 DataFrame
    return json_data


async def run_single_sim(net_id: str, ini_path: str):
    file_path = os.path.join('logs', net_id)

    if os.path.isfile(file_path) or os.path.islink(file_path):
        os.unlink(file_path)  # 删除文件或符号链接
    elif os.path.isdir(file_path):
        shutil.rmtree(file_path)  # 删除目录及其内容
    command = f'./routing -r 0 -m -u Cmdenv -n .:../inet4.5/src -l ../inet4.5/src/INET {ini_path}'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    logger.info(f"running sim command: {result.returncode}")

    if result.returncode == 0:
        log_path = os.path.join('logs', net_id, 'pkg_log.txt')
        json_data = analyse_datalog_file(log_path)
        logger.info(f'sim statistic result: {json_data}')
        # print(json_data)
        return json_data
    else:
        raise Exception(f"Command failed with error: {result.stderr}  {result.stdout}")


async def simulate_network_forwarding(net_id: str, ini_file, ned_file):
    # 这里可以是文件处理和网络转发仿真的代码
    # 模拟处理时间
    if net_id not in ini_file.filename:
        return {'status': "failed", 'message': f'net_id {net_id} not match {ini_file.filename}'}
    if net_id not in ned_file.filename:
        return {'status': "failed", 'message': f'net_id {net_id} not match {ned_file.filename}'}

    ini_path = os.path.join("ini_dir", ini_file.filename)
    ned_path = os.path.join("networks", ned_file.filename)

    if os.path.exists(ned_path):
        os.remove(ned_path)
    if os.path.exists(ini_path):
        os.remove(ini_path)
    await save_file(ini_file, ini_path)
    await save_file(ned_file, ned_path)  # 保存文件

    try:
        res_json = await run_single_sim(net_id, ini_path)  # 模拟网络
        return res_json
    except Exception as e:
        return {"status": "failed", 'message': str(e)}


@app.post("/network_simulation/")
async def network_simulation(ini_file: UploadFile = File(...), ned_file: UploadFile = File(...)):
    net_id = extract_unique_integer(ned_file.filename)
    print(f'reseive req {net_id}, {ini_file.filename}, {ned_file.filename}')
    res_json = await simulate_network_forwarding(str(net_id), ini_file, ned_file)
    return JSONResponse(status_code=202, content=res_json)


# 运行应用的命令，通常在生产环境中会使用更加健壮的服务器如Gunicorn
# 这里仅供示例
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
