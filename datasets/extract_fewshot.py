import ast
import copy
import json
import os
import random
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

from code_parser import Code_AST
from tqdm import tqdm
from utils import (scoring_worker, gen_worker, check_is_complete_function,
                   get_prompt_with_comment, pretty_comment, 
                   read_file, hash_string, get_indent,
                   robust_multiprocessing, fast_multiprocessing)
from extract_task_testgenissue import (get_test_dir, get_src_dir, get_classes_modified,
                                       get_formatted_report_libro, get_function, get_code_prefix)
from extract_defects4j_info import (get_project_ids, get_bug_ids)
random.seed(42)

def process_final_bench_complete(FILE_DIR):
    data = json.load(open(os.path.join(FILE_DIR, 'data', 'task_complete_filtered_with_better_comment_and_score.json')))
    few_shot_exmaple = []
    for example in data:
        test_class = [testmethod.split("::")[0].split(".")[-1] for testmethod in example["testmethods"]]
        test_class = [tc.replace("Tests", "").replace("Test", "").replace("tests", "").replace("test", "") for tc in test_class]
        file_class = example["location"].replace(".java","").split("/")[-1]
        if file_class in test_class:
            # print(example["function_tested_rate"], example["score"])
            if example["function_tested_rate"] >= 0.6 and example["score"] >= 7.5:
                continue
            elif example["function_tested_rate"] >= 0.8 and example["score"] >= 7:
                continue
            elif example["function_tested_rate"] < 0.5:
                continue
            elif example["score"] < 7:
                continue
            few_shot_exmaple.append(
                {
                    "function_name": example["function_name"],
                    "comment": example["function_comment"],
                    "function": example["function"],
                    "function_signature":example["function_signature"],
                }
            )
    print("Total Task Case after filtering:", len(few_shot_exmaple))
    few_shot_exmaple.sort(key=lambda x: len(x["comment"]+x["function"]))
    few_shot_exmaple = few_shot_exmaple[:10]
    random.shuffle(few_shot_exmaple)
    print("Total Task Case after filtering:", len(few_shot_exmaple))
    json.dump(few_shot_exmaple, open(os.path.join(FILE_DIR, 'data/fewshot', 'task_complete_fewshot.json'), 'w'), indent=4, ensure_ascii=False)
    with open(os.path.join(FILE_DIR, 'data/fewshot', 'task_complete_fewshot.txt'), 'w') as f:
        for example in few_shot_exmaple:
            f.write(example["comment"] + "\n")
            f.write(example["function"] + "\n")
            f.write("\n\n")


def process_final_bench_testgen(FILE_DIR):
    data = json.load(open(os.path.join(FILE_DIR, 'data', 'task_testgen_filtered_with_better_comment_and_score.json')))
    few_shot_exmaple = []
    for example in data:
        if example["score"] >= 7.5:
            continue
        if example["score"] < 7:
            continue
        few_shot_exmaple.append(
            {
                "function_name": example["function_name"],
                "comment": example["function_comment"],
                "function": example["function"],
                "function_signature":example["function_signature"],
                "be_test_class_signature": example["be_test_class_signature"],
            }
        )
        
    print("Total Task Case after filtering:", len(few_shot_exmaple))
    few_shot_exmaple.sort(key=lambda x: len(x["comment"]+x["function"]))
    few_shot_exmaple = few_shot_exmaple[:10]
    random.shuffle(few_shot_exmaple)
    print("Total Task Case after filtering:", len(few_shot_exmaple))
    json.dump(few_shot_exmaple, open(os.path.join(FILE_DIR, 'data/fewshot', 'task_testgen_fewshot.json'), 'w'), indent=4, ensure_ascii=False)
    with open(os.path.join(FILE_DIR, 'data/fewshot', 'task_testgen_fewshot.txt'), 'w') as f:
        for example in few_shot_exmaple:
            f.write(example["be_test_class_signature"] + "\n")
            f.write(example["comment"] + "\n")
            f.write(example["function"] + "\n")
            f.write("\n\n")
 
def process_issues_info(FILE_DIR):
    all_issue_tests = []
    
    data = json.load(open(os.path.join(FILE_DIR, "data", "task_repair_bench_default|2048.json")))
    filter_out_example_id = []
    for example in data["code_ucb_repair"]:
        example_id = f"{example['project']}-{example['bug_id']}"
        filter_out_example_id.append(example_id)
    
    all_data = json.load(open(os.path.join(FILE_DIR, 'data', 'task_info.json')))
    filter_examples = []
    for example in all_data:
        example_id = f"{example['project_id']}-{example['bug_id']}" 
        if example_id in filter_out_example_id:
            continue
        
        filter_examples.append({"project":example["project_id"], "bug_id":example["bug_id"], 
                                "testmethods":example["tests_trigger"]})
        
    for example in filter_examples:
        testmethods = example["testmethods"]
        
        test_dir = get_test_dir(FILE_DIR, example["project"], example["bug_id"])
        src_dir = get_src_dir(FILE_DIR, example["project"], example["bug_id"])
        classes_modified = get_classes_modified(FILE_DIR, example["project"], example["bug_id"])
        # print(example.keys())
        issue_test = {
            "project_id": example["project"],
            "bug_id": example["bug_id"],
            "src_classes": src_dir,
            "test_classes": test_dir,
            "classes_modified": classes_modified,
            "testmethods": testmethods,
            "issue_url": "URL",
            "test_paths": [os.path.join(FILE_DIR, "projects", example["project"], 
                                      str(example["bug_id"]) + 'b', test_dir, 
                                      testmethod.split("::")[0].replace(".", "/") + ".java") for testmethod in testmethods],
        }
        all_issue_tests.append(issue_test)
    print("len(all_issue_tests)", len(all_issue_tests))
    json.dump(all_issue_tests, open(os.path.join(FILE_DIR, "data/fewshot", "task_testgenissue_info.json"), "w"), indent=4)

def extract_worker(task):
    # print(type(task))
    # if isinstance(task, set):
    #     print(task)
    task["test_classes"]
    task["testmethod"].split("::")[0].replace(".", "/")
    location = os.path.join(task["test_classes"], task["testmethod"].split("::")[0].replace(".", "/") + ".java")
    test_class_path_fixed = os.path.join(task["FILE_DIR"], "projects", task["project_id"], str(task["bug_id"]) + 'f',
                                   location)
    test_class_path_buggy = os.path.join(task["FILE_DIR"], "projects", task["project_id"], str(task["bug_id"]) + 'b',
                                   location)
    
    test_function_name = task["testmethod"].split("::")[-1]
    try:
        function_fixed, comment_fixed, class_file_fixed, class_ast_fixed = get_function(test_class_path_fixed, test_function_name)
        function_prefix_with_signature, _, function_signature = get_code_prefix(class_ast_fixed, function_fixed.source_line)
        test_class_context = class_ast_fixed.get_class_context_source()
        test_file_context = class_ast_fixed.get_file_context_source()
        indent = class_ast_fixed.get_indent()
        
        import_context = class_ast_fixed.get_import_context_source()
        class_signature = class_ast_fixed.get_class_signature_context_source()
        class_field_context = class_ast_fixed.get_class_field_context_source()
        class_function_signature_context = class_ast_fixed.get_class_functions_signature_context_source()
        # insert_line, class_file_buggy = get_insert_place(test_class_path_buggy)
        function_buggy, comment_buggy, class_file_buggy, class_ast_buggy = get_function(test_class_path_buggy, test_function_name)
    except:
        return None
    
    issue_id = f"{task['project_id']}-{task['report']['issue_id']}"
    issue_report = f"## Issue-ID: {issue_id}\n\n## Issue-Title: \n{task['report']['title']}\n\n## Issue-Description: \n{task['report']['discussion']}\n\n"
    return {
        "task_id": f"testgenissue|{task['project_id']}|{task['bug_id']}|{location}|{test_function_name}|{function_fixed.start_line}|{function_fixed.end_line}",
        "project_id": task["project_id"],
        "bug_id": task["bug_id"],
        "testmethod": task["testmethod"],
        "classes_modified": task["classes_modified"],
        "src_classes": task["src_classes"],
        "test_classes": task["test_classes"],
        "location": location,
        "test_class": task["testmethod"].split("::")[0],
        "function_comment": comment_fixed,
        "function_signature": function_signature,
        "function": function_fixed.source_line,
        "function_name": test_function_name,
        "function_start_line_fixed": function_fixed.start_line,
        "function_end_line_fixed": function_fixed.end_line,
        "function_start_line_buggy": function_buggy.start_line,
        "function_end_line_buggy": function_buggy.end_line,
        "test_source_fixed": class_file_fixed,
        "test_source_buggy": class_file_buggy,
        "import_context": import_context,
        "class_signature": class_signature,
        "class_field_context": class_field_context,
        "class_function_signature_context": class_function_signature_context,
        "indent": indent,
        "issue_id": issue_id,
        "issue_report": issue_report,
        "report": task["report"],
        "test_class_context": test_class_context,
        "test_file_context": test_file_context,
    }

def process_issues_task(FILE_DIR):
    processed_tasks = []
    test_tasks = []
    data = json.load(open(os.path.join(FILE_DIR, "data/fewshot", "task_testgenissue_info.json")))
    for issue in tqdm(data):
        # report_path, file_type = get_raw_report(FILE_DIR, issue["project_id"], issue["bug_id"], issue["issue_url"])
        # report = get_formatted_report(report_path, file_type)
        report_path = os.path.join(FILE_DIR, "data/bug_report", f"{issue['project_id']}-{issue['bug_id']}.json")
        if not os.path.exists(report_path):
            print(report_path)
            continue
        report = get_formatted_report_libro(report_path)
        
        for testmethod in issue["testmethods"][:1]:
            test_tasks.append(
                {
                    "FILE_DIR": FILE_DIR,
                    "project_id": issue["project_id"], 
                    "bug_id": issue["bug_id"], 
                    "report": report,
                    "testmethod": testmethod,
                    "test_classes": issue["test_classes"],
                    "src_classes": issue["src_classes"],
                    "classes_modified": issue["classes_modified"],
                }
            )
    
    # for task in tqdm(test_tasks):
        # processed_tasks.append(extract_worker(task))
        
    with ProcessPoolExecutor(max_workers=16) as executor:
        processed_tasks = list(tqdm(executor.map(extract_worker, test_tasks), total=len(test_tasks), desc="process_issues_task"))
    
    processed_tasks = [task for task in processed_tasks if task is not None]
    
    json.dump(processed_tasks, open(os.path.join(FILE_DIR, "data/fewshot", "task_testgenissue.json"), "w"), indent=4)

def process_final_bench_testgenissue(FILE_DIR):
    data = json.load(open(os.path.join(FILE_DIR, 'data/fewshot', 'task_testgenissue.json')))
    few_shot_exmaple = []
    for example in data:
        few_shot_exmaple.append(
            {
                "issue_report": example["issue_report"],
                "function": example["function"],
                "function_name": example["function_name"],
            }
        )
    
    few_shot_exmaple.sort(key=lambda x: len(x["issue_report"]+x["function"]))
    few_shot_exmaple = few_shot_exmaple[:10]
    random.shuffle(few_shot_exmaple)
    print("Total number of tasks after filtering: ", len(few_shot_exmaple))
    json.dump(few_shot_exmaple, open(os.path.join(FILE_DIR, 'data/fewshot', 'task_testgenissue_fewshot.json'), 'w'), indent=4, ensure_ascii=False)
    with open(os.path.join(FILE_DIR, 'data/fewshot', 'task_testgenissue_fewshot.txt'), 'w') as f:
        for example in few_shot_exmaple:
            f.write(example["issue_report"] + "\n")
            f.write(example["function"] + "\n")
            f.write("\n\n")

def extract_complete(FILE_DIR):
    process_final_bench_complete(FILE_DIR)
    

def extract_testgen(FILE_DIR):
    process_final_bench_testgen(FILE_DIR)
    
    
def extract_testgenissue(FILE_DIR):
    process_issues_info(FILE_DIR)
    process_issues_task(FILE_DIR)
    process_final_bench_testgenissue(FILE_DIR)

def main():
    FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    # extract_complete(FILE_DIR)
    extract_testgen(FILE_DIR)
    # extract_testgenissue(FILE_DIR)
    
if __name__ == '__main__':
    main()    