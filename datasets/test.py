from concurrent.futures import ProcessPoolExecutor
import datetime
from multiprocessing import Manager, Queue
import os
import threading
import time

def get_test_relevant_methods_worker(item):
    result_queue = item
    result_queue.put({"path":os.path.join("data", "func_test_map", "asd.jsonl"), "data":""})

def process_results(result_queue, all_task_count):
    count = 0
    start_time = time.time()
    while True:
        time.sleep(0.5)
        result = result_queue.get()  # 阻塞直到从队列中获取一个结果
        count += 1
        if count % 10 == 0:
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
            break  # 如果收到 ENDING，表示没有更多的结果需要处理


manager = Manager()
result_queue = manager.Queue()

tasks = [(result_queue) for i in range(1000)]
result_thread = threading.Thread(target=process_results, args=(result_queue, len(tasks)))
result_thread.start()

# results = []
# for task in tqdm(tasks, desc="Processing Test Files"):
#     results.append(get_test_relevant_methods_worker(task))    

with ProcessPoolExecutor(max_workers=64) as executor:
    # 提交任务到线程池并传递队列
    [executor.submit(get_test_relevant_methods_worker, task) for task in tasks]

# process_results(result_queue, len(tasks))

# 所有任务提交后，向队列发送一个 None 以通知结果处理线程停止
result_thread.join()  # 等待结果处理线程完成