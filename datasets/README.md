# CoderUJB数据集构建流程
这里的教程将帮助您从零开始从defects4j中构建CoderUJB数据集。
首先请确保您已经安装了defects4j，并已经成功运行了defects4j。
或者您也可以使用我们提供的docker环境进行构建。该docker镜像已经安装好了我们所需的工具和依赖项。
请注意CoderUJB数据集的构建流程涉及大量的文件读写操作，因此我们强烈建议您在SSD硬盘中运行后续代码。

## 构建流程
### 进入项目工作路径
```bash
cd ISSTA24-CoderUJB
```
### 从defects4j中提取每个项目的各版本数据
```bash
python datasets/extract_defects4j_info.py
```
### 从defects4j的项目中提取测试覆盖关系
```bash
# 这一步将占用大量的时间（>10h），并且会同时启动大量的进程，请参考下面的链接确保你的环境支持大量进程同时运行。
# https://stackoverflow.com/questions/32283003/python-cant-start-new-thread-100-active-threads
python datasets/extract_function_test_map.py
```

### 构建CoderUJB-FCG（Functional Code Generation）数据集
```bash
# CoderUJB-FCG数据集将被储存在ISSTA24-CoderUJB/datasets/data/task_complete_bench_def.json'
python datasets/extract_task_complete.py
```