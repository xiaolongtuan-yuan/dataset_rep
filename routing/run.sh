#!/bin/zsh

# 循环变量 i 从 0 到 10
cd ini_dir
for i in {0..9}; do
    # 构建 ini 文件路径
    ini_file="omnetpp${i}.ini"

    # 执行命令
#    ./routing -u Cmdenv -n . "$ini_file"
    ../routing -r 0 -m -u Cmdenv -n ..:../../inet4.5/src -l ../../inet4.5/src/INET ${ini_file}
done

echo "*************running finish*******************"