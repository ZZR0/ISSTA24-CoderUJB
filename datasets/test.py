from concurrent.futures import ProcessPoolExecutor
import datetime
from multiprocessing import Manager
import os
import threading
import time

def get_test_relevant_methods_worker(shared_list, index):
    shared_list[index] = {"path": os.path.join("data", "func_test_map", "asd.jsonl"), "data": ""}

def process_results(shared_list, all_task_count):
    start_time = time.time()
    while True:
        time.sleep(0.5)
        count = sum(1 for item in shared_list if item is not None)  # 计算非 None 元素的数量
        if count % 10 == 0 and count != 0:
            # 打印完成数量
            print(f"{'='*10} {datetime.datetime.now()} {'='*10}")
            print(f"Finished: {count}/{all_task_count} ({count / all_task_count * 100:.2f}%)")
            # 计算剩余时间
            used_time = (time.time() - start_time)
            hour = int(used_time / 3600)
            minute = int((used_time % 3600) / 60)
            second = int(used_time % 60)
            print(f"Used Time Cost: {hour}h {minute}m {second}s")
            total_time = (time.time() - start_time) / count * all_task_count
            hour = int(total_time / 3600)
            minute = int((total_time % 3600) / 60)
            second = int(total_time % 60)
            print(f"Total Time Cost: {hour}h {minute}m {second}s")
        if count == all_task_count:
            print(f"{'='*10} {datetime.datetime.now()} {'='*10}")
            print(f"Finished: {count}/{all_task_count} ({count / all_task_count * 100:.2f}%)")
            break  # 如果所有任务都完成，退出循环


manager = Manager()
shared_list = manager.list([None] * 1000)  # 初始化共享列表

tasks = [(shared_list, i) for i in range(1000)]
result_thread = threading.Thread(target=process_results, args=(shared_list, len(tasks)))
result_thread.start()

with ProcessPoolExecutor(max_workers=64) as executor:
    # 提交任务到线程池并传递共享列表和索引
    [executor.submit(get_test_relevant_methods_worker, *task) for task in tasks]

result_thread.join()  # 等待结果处理线程完成
