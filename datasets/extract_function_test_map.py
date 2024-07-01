import datetime
import fnmatch
import json
from multiprocessing import Manager
import os
import random
import string
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from code_parser import Code_AST
from tqdm import tqdm

from utils import read_file

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = f'{FILE_DIR}/tmp/'
os.makedirs(TMP_DIR, exist_ok=True)

def get_project_ids():
    cmd = ['defects4j', 'pids']
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    project_ids = result.stdout.splitlines()
    return project_ids

def get_latest_bug_id(project_id):
    save_path = os.path.join(TMP_DIR, "query_info.csv")
    query_str = "bug.id,revision.date.fixed"
    cmd = ['defects4j', 'query', '-p', project_id, '-q', query_str, '-o', save_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["sed", "-i", "1i"+query_str, save_path])
    
    # Read the CSV file without parsing the date column
    data = pd.read_csv(save_path)
    
    # Convert the revision.date.fixed column to datetime format, attempting to infer the format
    data['revision.date.fixed'] = pd.to_datetime(data['revision.date.fixed'].apply(lambda x: ' '.join(x.split(' ')[:2])))
    
    # Find the row with the latest revision date
    latest_bug = data.loc[data['revision.date.fixed'].idxmax()]
    
    # Return the bug id
    return latest_bug["bug.id"]

def process_extract_info(FILE_DIR):
    processed_data = []
    
    project_ids = get_project_ids()
    for project_id in project_ids:
        bug_id = int(get_latest_bug_id(project_id))
        
        checkout_path = os.path.join(FILE_DIR, "projects", project_id, str(bug_id) + 'f')
        
        cmd = ['defects4j', 'export', '-p', 'dir.src.classes', '-w', checkout_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # print(" ".join(cmd))
        src_classes = result.stdout.splitlines()[-1].strip()
        cmd = ['defects4j', 'export', '-p', 'dir.src.tests', '-w', checkout_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        src_tests = result.stdout.splitlines()[-1].strip()
    
        processed_data.append({
            "project_id": project_id,
            "bug_id": bug_id,
            "src_classes": src_classes,
            "src_tests": src_tests,
        })
    
    json.dump(processed_data, open(os.path.join(FILE_DIR, 'data', 'data_latest_bug.json'), 'w'), indent=4, ensure_ascii=False)

def find_java_files(folder_path):
    java_files = []

    # Recursive function to search for .java files.
    def search_files(path):
        for entry in os.listdir(path):
            entry_path = os.path.join(path, entry)

            if os.path.isdir(entry_path):
                search_files(entry_path)
            elif fnmatch.fnmatch(entry_path, '*.java'):
                java_files.append(entry_path)
    # print("Searching for .java files in " + folder_path)
    search_files(folder_path)
    return java_files

def process_extract_class_info(FILE_DIR):
    processed_data = []
    data = json.load(open(os.path.join(FILE_DIR, 'data', 'data_latest_bug.json')))
    for item in data:
        src_classes = item['src_classes']
        src_tests = item['src_tests']
        project_id = item['project_id']
        bug_id = item['bug_id']
        
        checkout_path = os.path.join(FILE_DIR, "projects", project_id, str(bug_id) + 'f')
        src_class_dir = os.path.join(checkout_path, src_classes)
        src_class_files = find_java_files(src_class_dir)
        src_class_files = [file.replace(src_class_dir, '').replace('.java', '').replace('/', '.')[1:] for file in src_class_files]
        item["src_class_files"] = src_class_files
        
        src_tests_dir = os.path.join(checkout_path, src_tests)
        src_test_files = find_java_files(src_tests_dir)
        src_test_files = [file.replace(src_tests_dir, '').replace('.java', '').replace('/', '.')[1:] for file in src_test_files if "test" in file.lower()]
        item["src_test_files"] = src_test_files
        
        processed_data.append(item)
    
    json.dump(processed_data, open(os.path.join(FILE_DIR, 'data', 'data_class_test_info.json'), 'w'), indent=4, ensure_ascii=False)                     

def get_test_relevant_methods_worker(item):
    def generate_random_string(length):
        characters = string.ascii_letters + string.digits  # 包含大写字母、小写字母和数字
        random_string = ''.join(random.choice(characters) for _ in range(length))
        return random_string
    item_data, test_file, file_dir, result_queue = item
    test_src = item_data["src_tests"]
    project_id, bug_id = item_data["project_id"], item_data["bug_id"]
    tmp_project_path = os.path.join(TMP_DIR, f"{project_id}-{bug_id}", generate_random_string(16))

    try:
        subprocess.run(['rm', '-rf', tmp_project_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        cmd = ['defects4j', 'checkout', '-p', project_id, '-v', str(bug_id) + 'f', '-w', tmp_project_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        increment_path = os.path.join(tmp_project_path, "increment.txt")
        with open(increment_path, 'w') as f:
            f.writelines([file+"\n" for file in item_data["src_class_files"]])
        
        test_relevant_methods = []
        test_file_path = test_file.replace(".", "/") + ".java"
        test_code = read_file(os.path.join(tmp_project_path, test_src, test_file_path))
        test_ast = Code_AST(code=test_code, lang="java").ast
        test_functions = test_ast.get_functions()
        test_functions_name = [func.get_function_name() for func in test_functions]
        
        # test_functions_name = random.sample(test_functions_name, 100) \
        #                         if len(test_functions_name) > 100 else test_functions_name
        
        # for test_function_name in tqdm(test_functions_name, desc="Test Function"):
        for test_function_name in test_functions_name:
            cmd = ["defects4j", "coverage", "-w", tmp_project_path, "-t", f"{test_file}::{test_function_name}", 
                "-i", increment_path]
            # print(" ".join(cmd))
            # exit()
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Specify the path to the 'coverage.xml' file
            xml_file_path = os.path.join(tmp_project_path, "coverage.xml")

            # Load and parse the XML file
            tree = ET.parse(xml_file_path)
            root = tree.getroot()

            # Iterate through the coverage data, searching for the target class and method
            for package in root.findall(".//package"):
                for class_element in package.findall(".//class"):
                    be_test_class_name = class_element.attrib["name"]
                    be_test_class_file = class_element.attrib["filename"]
                    
                    for method_element in class_element.findall(".//method"):
                        method_name = method_element.attrib["name"]
                        method_signature = method_element.attrib["signature"]
                        method_line_rate = method_element.attrib["line-rate"]
                        if float(method_line_rate) > 0:
                            line_numbers = []
                            for line_element in method_element.findall(".//line"):
                                line_numbers.append(line_element.attrib["number"])
                            if len(line_numbers) == 0: continue
                            test_relevant_methods.append({"test_file":test_file, "test_function":test_function_name, 
                                                        "be_test_class_name":be_test_class_name,
                                                        "be_test_class_file":be_test_class_file,
                                                        "be_test_function_name":method_name,
                                                        "be_test_function_signature":method_signature,
                                                        "line_numbers": line_numbers, 
                                                        "method_line_rate":float(method_line_rate)})
        item_data["test_relevant_methods"] = test_relevant_methods
        
        
        result_queue.put({"path":os.path.join(file_dir, "data", "func_test_map", project_id+".jsonl"), "data":item_data})
    finally:
        subprocess.run(['rm', '-rf', tmp_project_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)


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
        if isinstance(result, dict):
            with open(result["path"], "a", encoding="utf-8") as f:
                f.write(json.dumps(result["data"], ensure_ascii=False) + "\n")
        if count >= 0.95*all_task_count:
            print(f"{'='*10} {datetime.datetime.now()} {'='*10}")
            print(f"Finished: {count}/{all_task_count} ({count / all_task_count * 100:.2f}%)")
            break  # 如果收到 ENDING，表示没有更多的结果需要处理


def process_extract_test_coverage(FILE_DIR):
    os.makedirs(os.path.join(FILE_DIR, "data", "func_test_map"), exist_ok=True)
    data = json.load(open(os.path.join(FILE_DIR, 'data', 'data_class_test_info.json')))

    tasks = []
    manager = Manager()
    result_queue = manager.Queue()
    for item in data:
        test_files = item["src_test_files"]
        
        for test_file in test_files:
            tasks.append((item, test_file, FILE_DIR, result_queue))
            
        with open(os.path.join(FILE_DIR, "data", "func_test_map", item["project_id"]+".jsonl"), "w") as f:
            f.write("")
    # tasks = tasks[:20]
    random.shuffle(tasks)
    
    result_thread = threading.Thread(target=process_results, args=(result_queue, len(tasks)))
    result_thread.start()

    # results = []
    # for task in tqdm(tasks, desc="Processing Test Files"):
    #     results.append(get_test_relevant_methods_worker(task))    
    
    with ProcessPoolExecutor(max_workers=16) as executor:
        # 提交任务到线程池并传递队列
        [executor.submit(get_test_relevant_methods_worker, task) for task in tasks]

    # 所有任务提交后，向队列发送一个 None 以通知结果处理线程停止
    result_thread.join()  # 等待结果处理线程完成

def process_extract_mapping(FILE_DIR):
    project_ids = get_project_ids()
    processed_data = []
    
    for project_id in project_ids:
        func_test_map_path = os.path.join(FILE_DIR, 'data', "func_test_map", project_id+".jsonl")
        func_test_map = []
        with open(func_test_map_path, 'r') as f:
            for line in f:
                func_test_map.append(json.loads(line))
        
        merged_func_test_map = {
            "project_id": func_test_map[0]["project_id"],
            "bug_id": func_test_map[0]["bug_id"],
            "src_classes": func_test_map[0]["src_classes"],
            "src_tests": func_test_map[0]["src_tests"],
            "src_class_files": func_test_map[0]["src_class_files"],
            "src_test_files": func_test_map[0]["src_test_files"],
            "test_relevant_methods": []
        }
        for func_test in func_test_map:
            merged_func_test_map["test_relevant_methods"].extend(func_test["test_relevant_methods"])
        
        function_to_test = {}
        for mapping in merged_func_test_map["test_relevant_methods"]:
            key = "::".join([
                mapping["be_test_class_name"],
                mapping["be_test_function_name"],
                mapping["be_test_function_signature"]
            ])
            if key not in function_to_test:
                function_to_test[key] = []
            function_to_test[key].append(mapping)
            # function_to_test[key].append(f"{mapping['test_file']}::{mapping['test_function']}")
        merged_func_test_map["function_to_test"] = function_to_test
        
        test_to_function = {}
        for mapping in merged_func_test_map["test_relevant_methods"]:
            key = "::".join([
                mapping["test_file"],
                mapping["test_function"],
            ])
            if key not in test_to_function:
                test_to_function[key] = []
            test_to_function[key].append(mapping)
        merged_func_test_map["test_to_function"] = test_to_function
        
        processed_data.append(merged_func_test_map)
    json.dump(processed_data, open(os.path.join(FILE_DIR, 'data', "data_processed_mapping.json"), 'w'), indent=4)

def main():
    process_extract_info(FILE_DIR)
    process_extract_class_info(FILE_DIR)
    
    process_extract_test_coverage(FILE_DIR)
    
    process_extract_mapping(FILE_DIR)
    
if __name__ == '__main__':
    main()    