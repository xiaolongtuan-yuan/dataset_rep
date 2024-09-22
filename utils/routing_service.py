# -*- coding: utf-8 -*-
"""
@Time ： 2024/5/31 13:14
@Auth ： xiaolongtuan
@File ：routing_service.py
"""
import json

import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

# 接口的URL
url = "http://10.112.14.208:8000/network_simulation/"


def get_routing_sim(net_id, ini_path, ned_path):
    m = MultipartEncoder(
        fields={
            'net_id': (None, net_id),
            'ini_file': (f'omnetpp{net_id}.ini', open(ini_path, 'rb'), 'application/octet-stream'),
            'ned_file': (f'Myned{net_id}.ned', open(ned_path, 'rb'), 'application/octet-stream'),
        }
    )
    # 发送POST请求
    response = requests.post(url, data=m, headers={'Content-Type': m.content_type})
    m.fields['ini_file'][1].close()
    m.fields['ned_file'][1].close()
    # 打印响应内容
    try:
        data = json.loads(response.json())
        return data
    except Exception as e:
        print(response.json())
        raise e
