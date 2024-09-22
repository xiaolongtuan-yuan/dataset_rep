# -*- coding: utf-8 -*-
"""
@Time ： 2024/5/9 15:09
@Auth ： xiaolongtuan
@File ：test.py
"""
import json
import math
import os


def delete_update_record_files(root_dir):
    """
    Traverse the directory and delete all files named 'update_record.pkl'.

    Parameters:
    - root_dir: The root directory to start traversing from.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == 'update_record.pkl':
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")


def remove_qos_constraint(file_path):
    """
    Remove the 'qos_constraint' field from each JSON object in the file and save the modified file.

    Parameters:
    - file_path: The path to the JSONL file to process.
    """
    # Read contents from the original file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()

    # Process each line (assuming each line is a valid JSON object)
    modified_data = []
    for line in data:
        json_obj = json.loads(line)
        json_obj[
            'qos_constraint'] = "The end-to-end delay of the traffic from node c{:.0f} to node c{:.0f} is less than {}ms, the average jitter is less than {}ms, and the packet loss rate is less than {:.2f}%".format(
            json_obj['src'], json_obj['dst'], math.ceil(json_obj['avgDelay'] * 10) * 100,
                                              math.ceil(json_obj['avgJitter'] * 100) * 10,
                                              json_obj['pkgLossRate'] * 100)
        modified_data.append(json.dumps(json_obj))  # Convert back to JSON string

    # Write the modified data back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(modified_data))


# 使用范例
# 指定要遍历的根目录

directory_paths = ["understand_data_14", "understand_data_35", "understand_data_70", "understand_data_100"]

for directory_path in directory_paths:
    for dirpath, dirnames, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith(".jsonl"):
                file_path = os.path.join(dirpath, filename)
                remove_qos_constraint(file_path)
    print(f"Removed 'qos_constraint' from {directory_path} and saved.")
