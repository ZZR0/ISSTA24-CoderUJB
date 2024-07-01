from concurrent.futures import ProcessPoolExecutor
import os
import json
import re
from collections import namedtuple
from difflib import unified_diff
import subprocess
from code_parser import Code_AST
from tqdm import tqdm
from utils import read_file, hash_string
import random

from transformers import AutoTokenizer

random.seed(42)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

ChangeInfo = namedtuple('ChangeInfo', ['line', 'action', 'content'])

def parse_diff_text(diff_text):
    # print(diff_text)
    # print('='*100)
    lines = diff_text.strip().split('\n')
    parsed_changes, parsed_changes_chunk = [], []
    
    for line in lines:
        if line.startswith('+++') or line.startswith('---'):
            continue
        elif line.startswith('@@'):
            line_numbers_sub = re.findall(r'\-\d+', line)
            line_numbers_add = re.findall(r'\+\d+', line)
            # print(line_numbers_sub)
            # print(line_numbers_add)
            current_line_sub = int(line_numbers_sub[0][1:])
            current_line_add = int(line_numbers_add[0][1:])
            if parsed_changes_chunk:
                parsed_changes.extend(parsed_changes_chunk)
            parsed_changes_chunk = []
        elif line.startswith('-'):
            line_content = " "+line[1:].strip()
            parsed_changes_chunk.append(ChangeInfo(current_line_sub, 'sub', line_content))
            parsed_changes_chunk.append(ChangeInfo(current_line_add, 'add', line_content))
            current_line_sub += 1
        elif line.startswith('+'):
            line_content = " "+line[1:].strip()
            parsed_changes_chunk.append(ChangeInfo(current_line_add, 'add', line_content))
            parsed_changes_chunk.append(ChangeInfo(current_line_sub, 'sub', line_content))
            current_line_add += 1
        else:
            parsed_changes_chunk.append(ChangeInfo(current_line_add, 'add-blank', line.strip()))
            parsed_changes_chunk.append(ChangeInfo(current_line_sub, 'sub-blank', line.strip()))
            current_line_add += 1
            current_line_sub += 1
    
    if parsed_changes_chunk:
        parsed_changes.extend(parsed_changes_chunk)
        
    return parsed_changes


def get_unified_diff(source, mutant):
    output = ""
    for line in unified_diff(source.split('\n'), mutant.split('\n'), lineterm=''):
        output += line + "\n"
    return output

def get_modified_function(functions, change):
    modified_functions = []
    for function in functions:
        if change.line >= function.start_line+1 and change.line <= function.end_line+1:
            modified_functions.append(function)
    
    if len(modified_functions) == 0:
        return None
    
    modified_function = modified_functions[0]
    for function in modified_functions:
        if modified_function.source in function.source:
            modified_function = function
    
    return modified_function

def check_is_repair_function(node):
    if node.path.endswith("class_body|method_declaration"):
        return True
    if node.path.endswith("enum_body_declarations|method_declaration"):
        return True
    if node.path.endswith("class_body|constructor_declaration"):
        return True
    return False

def process_worker(FILE_DIR, example):
    if len(example["classes_modified"]) > 1:
        return None
    
    example_processed = {}
    example_processed.update(example)
    example_processed["buggy_function"] = {}
    example_processed["fixed_function"] = {}
    example_processed["buggy_source"] = {}
    example_processed["fixed_source"] = {}
    
    project_fixed_path = os.path.join(FILE_DIR, "projects", example['project_id'], example["bug_id"]+"f")
    project_buggy_path = os.path.join(FILE_DIR, "projects", example['project_id'], example["bug_id"]+"b")
    
    os.system("rm -rf " + project_fixed_path)
    os.system("rm -rf " + project_buggy_path)
    cmd = ['defects4j', 'checkout', '-p', example['project_id'], '-v', example["bug_id"] + 'f', '-w', project_fixed_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cmd = ['defects4j', 'checkout', '-p', example['project_id'], '-v', example["bug_id"] + 'b', '-w', project_buggy_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    java_file_path = example["classes_modified"][0].replace(".", "/") + ".java"
    java_file_fixed_path = os.path.join(project_fixed_path, example["src_classes"], java_file_path)
    java_file_buggy_path = os.path.join(project_buggy_path, example["src_classes"], java_file_path)
    
    # print(java_file_fixed_path)
    # print(java_file_buggy_path)
    
    fixed_file = read_file(java_file_fixed_path)
    buggy_file = read_file(java_file_buggy_path)
    
    example_processed["buggy_source"][os.path.join(example["src_classes"], java_file_path)] = buggy_file
    example_processed["fixed_source"][os.path.join(example["src_classes"], java_file_path)] = fixed_file
    
    diff_text = get_unified_diff(buggy_file, fixed_file)
    
    code_ast_buggy = Code_AST(code=buggy_file, lang="java").ast
    code_ast_fixed = Code_AST(code=fixed_file, lang="java").ast
    
    functions_buggy = [func for func in code_ast_buggy.get_functions() if check_is_repair_function(func)]
    functions_fixed = [func for func in code_ast_fixed.get_functions() if check_is_repair_function(func)]
    
    # print(fixed_file)
    # print("------------------------------------------")
    # print(diff_text)
    for change in parse_diff_text(diff_text):
        
        if change.action == 'add':
            functions = functions_fixed
        elif change.action == 'sub':
            functions = functions_buggy
        else:
            continue
        
        modified_function = get_modified_function(functions, change)
        if modified_function is None:
            function_name = "not-function"
            function_body = ""
            function_comment = ""
            function_signature = ""
            function_star_line = 0
            function_end_line = 0
            is_nest_function = False
        else:
            function_body = modified_function.source_line
            function_name = modified_function.get_function_name()
            function_comment = modified_function.get_function_comment()
            function_signature = modified_function.get_function_signature_source()
            function_star_line = modified_function.start_point[0]
            function_end_line = modified_function.end_point[0]
            is_nest_function = modified_function.check_is_nest_function(modified_function)
        
        # print("change: ", change)
        # print(function_name, function_star_line, function_end_line)
        func_info = {
                "function_star_line": function_star_line,
                "function_end_line": function_end_line,
                "function_name": function_name,
                "function_body": function_body,
                "function_comment": function_comment,
                "function_signature": function_signature,
                "is_function": False if function_name == "not-function" else True,
                "have_comment": False if function_comment == "" else True,
                "is_nest_function": is_nest_function,
            }
        try:
            if change.action == 'add':
                func_info["import_context"] = code_ast_fixed.get_import_context_source()
                func_info["class_signature"] = code_ast_fixed.get_class_signature_context_source()
                func_info["class_field_context"] = code_ast_fixed.get_class_field_context_source()
                func_info["class_function_signature_context"] = code_ast_fixed.get_class_functions_signature_context_source()
                example_processed["fixed_function"][f"{function_name}|{function_star_line}|{function_end_line}"] = func_info
            elif change.action == 'sub':
                func_info["import_context"] = code_ast_buggy.get_import_context_source()
                func_info["class_signature"] = code_ast_buggy.get_class_signature_context_source()
                func_info["class_field_context"] = code_ast_buggy.get_class_field_context_source()
                func_info["class_function_signature_context"] = code_ast_buggy.get_class_functions_signature_context_source()
                example_processed["buggy_function"][f"{function_name}|{function_star_line}|{function_end_line}"] = func_info
        except:
            return None
        
    return example_processed

def process_extract_info(FILE_DIR):
    json_data = json.load(open(os.path.join(FILE_DIR, 'data', 'task_info.json')))
    processed_data = []
    tasks = [(FILE_DIR, example) for example in json_data]
    # tasks = [task for task in tasks if task[1]["project_id"] == "Math" and task[1]["bug_id"] == "90"]
    # print(tasks)
    
    # for task in tqdm(tasks):
    #     processed_data.append(process_worker(*task))
        
    with ProcessPoolExecutor(max_workers=16) as executor:
        processed_data = list(tqdm(executor.map(process_worker, *zip(*tasks)), total=len(tasks), desc="Processing"))

    processed_data = [example for example in processed_data if example is not None]
    json.dump(processed_data, open(os.path.join(FILE_DIR, 'data', 'task_defectdetection.json'), 'w'), indent=4, ensure_ascii=False)

def process_extract_function(FILE_DIR):
    
    task_defectdetection = json.load(open(os.path.join(FILE_DIR, 'data', 'task_defectdetection.json')))
    print("Total number of tasks: ", len(task_defectdetection))
    
    single_function_tasks = []
    multi_function_tasks = []
    
    for task in task_defectdetection:
        # print([function["is_function"] for function in task["fixed_function"].values()])
        # print([function["is_function"] for function in task["buggy_function"].values()])
        if not all([function["is_function"] for function in task["fixed_function"].values()]):
            continue
        if not all([function["is_function"] for function in task["buggy_function"].values()]):
            continue
        if any([function["is_nest_function"] for function in task["fixed_function"].values()]):
            continue
        if any([function["is_nest_function"] for function in task["buggy_function"].values()]):
            continue
        
        fixed_functions_name = [function.split("|")[0] for function in task["fixed_function"]]
        buggy_functions_name = [function.split("|")[0] for function in task["buggy_function"]]
        
        if fixed_functions_name != buggy_functions_name:
            continue
        # if not all([function["have_comment"] for function in task["fixed_function"].values()]):
        #     continue
        
        if len(task["fixed_function"]) == 1 and len(task["buggy_function"]) == 1:
            single_function_tasks.append(task)
        else:
            multi_function_tasks.append(task)
            
    print("Total number of single function tasks: ", len(single_function_tasks))
    print("Total number of multi function tasks: ", len(multi_function_tasks))
    
    json.dump(single_function_tasks, open(os.path.join(FILE_DIR, 'data', 'task_defectdetection_single_function.json'), 'w'), indent=4, ensure_ascii=False)
    json.dump(multi_function_tasks, open(os.path.join(FILE_DIR, 'data', 'task_defectdetection_multi_function.json'), 'w'), indent=4, ensure_ascii=False)

def get_test_source(project_dir, tests_dir, testmethods):
    sources = []
    for testmethod in testmethods:
        file_path, method_name = testmethod.split("::")
        file_path = file_path.replace(".", "/") + ".java"
        try:
            with open(os.path.join(project_dir, tests_dir, file_path), 'r') as f:
                source = f.read()
        except:
            with open(os.path.join(project_dir, tests_dir, file_path), 'r', encoding='ISO-8859-1') as f:
                source = f.read()
        sources.append({"file":file_path, "method":method_name, "source":source})
    return sources

def get_code_context(import_context, class_signature, 
                     class_field_context, class_function_signature_context,
                     context_length):
    code_context = "{import_context}\n\n{class_signature} {{\n{class_field_context}\n\n{class_function_signature_context}\n}}"
    code_context = code_context.format(
        import_context=import_context,
        class_signature=class_signature,
        class_field_context=class_field_context,
        class_function_signature_context=class_function_signature_context
    )
    code_context = code_context.strip()
    cut_code_context = tokenizer.convert_tokens_to_string(tokenizer.tokenize(code_context)[-context_length:])
    cut = cut_code_context != code_context
    return cut, cut_code_context

def process_get_prompt(FILE_DIR, context=None, context_length=1024, save_suffix=""):
    
    PROMPT_COMPLETE_FS0 = """{comment}\n{code}\nQuestion: Please determine whether the above-mentioned Java function has any defects?\nA. Yes, it has defects\nB. No, it doesn't have defects\nAnswer:"""
    PROMPT_CHAT_FS0 = """I want you to act as a code defect detector, where I'll provide you with a Java function and it will be your responsibility to analyze it for potential issues based on the provided function code. Please respond with either "A. Yes, there are defects" or "B. No, there are no defects" based on your assessment. Let's get started with our first potentially flawed Java function:\n\n```java\n{comment}\n{code}\n```\n"""
    PROMPT_COMPLETE_FS1 = """/**\n* Counts the number of set bits in the binary representation of a given integer.\n*/\npublic static int bitcount(int n) {{\n    int count = 0;\n    while (n != 0) {{\n        n = (n ^ (n - 1));\n        count++;\n    }}\n    return count;\n}}\nQuestion: Please determine whether the above-mentioned Java function has any defects?\nA. Yes, it has defects\nB. No, it doesn't have defects\nAnswer: A. Yes, it has defects\n\n{comment}\n{code}\nQuestion: Please determine whether the above-mentioned Java function has any defects?\nA. Yes, it has defects\nB. No, it doesn't have defects\nAnswer:"""
    PROMPT_CHAT_FS1 = """I want you to act as a code defect detector, where I'll provide you with a Java function and it will be your responsibility to analyze it for potential issues based on the provided function code. Please respond with either "A. Yes, there are defects" or "B. No, there are no defects" based on your assessment. Let's get started with our first potentially flawed Java function:\n\n```java\n{comment}\n{code}\n```\n"""
    PROMPT_COMPLETE_FS2 = """/**\n* Counts the number of set bits in the binary representation of a given integer.\n*/\npublic static int bitcount(int n) {{\n    int count = 0;\n    while (n != 0) {{\n        n = (n ^ (n - 1));\n        count++;\n    }}\n    return count;\n}}\nQuestion: Please determine whether the above-mentioned Java function has any defects?\nA. Yes, it has defects\nB. No, it doesn't have defects\nAnswer: A. Yes, it has defects\n\n/**\n * Reverses the order of a singly linked list.\n */\npublic static Node reverse_linked_list(Node node) {{\n    Node prevnode = null;\n    Node nextnode;\n    while (node != null) {{\n        nextnode = node.getSuccessor();\n        node.setSuccessor(prevnode);\n        prevnode = node;\n        node = nextnode;\n    }}\n    return prevnode;\n}}\nQuestion: Please determine whether the above-mentioned Java function has any defects?\nA. Yes, it has defects\nB. No, it doesn't have defects\nAnswer: B. No, it doesn't have defects\n\n{comment}\n{code}\nQuestion: Please determine whether the above-mentioned Java function has any defects?\nA. Yes, it has defects\nB. No, it doesn't have defects\nAnswer:"""
    PROMPT_CHAT_FS2 = """I want you to act as a code defect detector, where I'll provide you with a Java function and it will be your responsibility to analyze it for potential issues based on the provided function code. Please respond with either "A. Yes, there are defects" or "B. No, there are no defects" based on your assessment. Let's get started with our first potentially flawed Java function:\n\n```java\n{comment}\n{code}\n```\n"""
    PROMPT_COMPLETE_FS3 = """/**\n* Counts the number of set bits in the binary representation of a given integer.\n*/\npublic static int bitcount(int n) {{\n    int count = 0;\n    while (n != 0) {{\n        n = (n ^ (n - 1));\n        count++;\n    }}\n    return count;\n}}\nQuestion: Please determine whether the above-mentioned Java function has any defects?\nA. Yes, it has defects\nB. No, it doesn't have defects\nAnswer: A. Yes, it has defects\n\n/**\n * Reverses the order of a singly linked list.\n */\npublic static Node reverse_linked_list(Node node) {{\n    Node prevnode = null;\n    Node nextnode;\n    while (node != null) {{\n        nextnode = node.getSuccessor();\n        node.setSuccessor(prevnode);\n        prevnode = node;\n        node = nextnode;\n    }}\n    return prevnode;\n}}\nQuestion: Please determine whether the above-mentioned Java function has any defects?\nA. Yes, it has defects\nB. No, it doesn't have defects\nAnswer: B. No, it doesn't have defects\n\n/**\n * Computes the maximum sum of any contiguous subarray in a given integer array.\n */\npublic static int max_sublist_sum(int[] arr) {{\n    int max_ending_here = 0;\n    int max_so_far = 0;\n\n    for (int x : arr) {{\n        max_ending_here = Math.max(0,max_ending_here + x);\n        max_so_far = Math.max(max_so_far, max_ending_here);\n    }}\n\n    return max_so_far;\n}}\nQuestion: Please determine whether the above-mentioned Java function has any defects?\nA. Yes, it has defects\nB. No, it doesn't have defects\nAnswer: B. No, it doesn't have defects\n\n{comment}\n{code}\nQuestion: Please determine whether the above-mentioned Java function has any defects?\nA. Yes, it has defects\nB. No, it doesn't have defects\nAnswer:"""
    PROMPT_CHAT_FS3 = """I want you to act as a code defect detector, where I'll provide you with a Java function and it will be your responsibility to analyze it for potential issues based on the provided function code. Please respond with either "A. Yes, there are defects" or "B. No, there are no defects" based on your assessment. Let's get started with our first potentially flawed Java function:\n\n```java\n{comment}\n{code}\n```\n"""
    PROMPT_COMPLETE_FS4 = """/**\n* Counts the number of set bits in the binary representation of a given integer.\n*/\npublic static int bitcount(int n) {{\n    int count = 0;\n    while (n != 0) {{\n        n = (n ^ (n - 1));\n        count++;\n    }}\n    return count;\n}}\nQuestion: Please determine whether the above-mentioned Java function has any defects?\nA. Yes, it has defects\nB. No, it doesn't have defects\nAnswer: A. Yes, it has defects\n\n/**\n * Reverses the order of a singly linked list.\n */\npublic static Node reverse_linked_list(Node node) {{\n    Node prevnode = null;\n    Node nextnode;\n    while (node != null) {{\n        nextnode = node.getSuccessor();\n        node.setSuccessor(prevnode);\n        prevnode = node;\n        node = nextnode;\n    }}\n    return prevnode;\n}}\nQuestion: Please determine whether the above-mentioned Java function has any defects?\nA. Yes, it has defects\nB. No, it doesn't have defects\nAnswer: B. No, it doesn't have defects\n\n/**\n * Computes the maximum sum of any contiguous subarray in a given integer array.\n */\npublic static int max_sublist_sum(int[] arr) {{\n    int max_ending_here = 0;\n    int max_so_far = 0;\n\n    for (int x : arr) {{\n        max_ending_here = Math.max(0,max_ending_here + x);\n        max_so_far = Math.max(max_so_far, max_ending_here);\n    }}\n\n    return max_so_far;\n}}\nQuestion: Please determine whether the above-mentioned Java function has any defects?\nA. Yes, it has defects\nB. No, it doesn't have defects\nAnswer: B. No, it doesn't have defects\n\n/**\n * Computes the square root of a given number using the Newton's method, to within a given epsilon difference for convergence.\n */\npublic static double sqrt(double x, double epsilon) {{\n    double approx = x / 2f;\n    while (Math.abs(x-approx) > epsilon) {{\n        approx = 0.5f * (approx + x / approx);\n    }}\n    return approx;\n}}\nQuestion: Please determine whether the above-mentioned Java function has any defects?\nA. Yes, it has defects\nB. No, it doesn't have defects\nAnswer: A. Yes, it has defects\n\n{comment}\n{code}\nQuestion: Please determine whether the above-mentioned Java function has any defects?\nA. Yes, it has defects\nB. No, it doesn't have defects\nAnswer:"""
    PROMPT_CHAT_FS4 = """I want you to act as a code defect detector, where I'll provide you with a Java function and it will be your responsibility to analyze it for potential issues based on the provided function code. Please respond with either "A. Yes, there are defects" or "B. No, there are no defects" based on your assessment. Let's get started with our first potentially flawed Java function:\n\n```java\n{comment}\n{code}\n```\n"""
    
    PROMPT_COMPLETE_FS = [
        PROMPT_COMPLETE_FS0, PROMPT_COMPLETE_FS1, PROMPT_COMPLETE_FS2, PROMPT_COMPLETE_FS3, PROMPT_COMPLETE_FS4
    ]
    PROMPT_CHAT_FS = [
        PROMPT_CHAT_FS0, PROMPT_CHAT_FS1, PROMPT_CHAT_FS2, PROMPT_CHAT_FS3, PROMPT_CHAT_FS4
    ]
    
    PROMPT_COMPLETE = """{code_context}\n\n You are a professional Java programmer, please identify any defects in the function named `{function_name}` based on the provided abstract Java class context information and the following function code.\n{comment}\n{code}\nQuestion: Please determine whether the above-mentioned Java function has any defects?\nA. Yes, it has defects\nB. No, it doesn't have defects\nAnswer:"""
    PROMPT_CHAT = """```java\n{code_context}\n```\n\nI want you to act as a code defect detector, where I'll provide you with a Java function and it will be your responsibility to analyze it for potential issues based on the provided abstract Java class context information and the following function code. Let's get started with our first potentially flawed Java function:\n\n```java\n{comment}\n{code}\n```\nPlease respond with either "A. Yes, there are defects" or "B. No, there are no defects" based on your assessment."""
    
    PROMPT_COMPLETE_DEFAULT_FS = """/**\n* Counts the number of set bits in the binary representation of a given integer.\n*/\npublic static int bitcount(int n) {{\n    int count = 0;\n    while (n != 0) {{\n        n = (n ^ (n - 1));\n        count++;\n    }}\n    return count;\n}}\nQuestion: Please determine whether the above-mentioned Java function has any defects?\nA. Yes, it has defects\nB. No, it doesn't have defects\nAnswer: A. Yes, it has defects\n\n/**\n * Reverses the order of a singly linked list.\n */\npublic static Node reverse_linked_list(Node node) {{\n    Node prevnode = null;\n    Node nextnode;\n    while (node != null) {{\n        nextnode = node.getSuccessor();\n        node.setSuccessor(prevnode);\n        prevnode = node;\n        node = nextnode;\n    }}\n    return prevnode;\n}}\nQuestion: Please determine whether the above-mentioned Java function has any defects?\nA. Yes, it has defects\nB. No, it doesn't have defects\nAnswer: B. No, it doesn't have defects\n\n{code_context}\n\nYou are a professional Java programmer, please identify any defects in the function named `{function_name}` based on the provided abstract Java class context information and the following function code.\n{comment}\n{code}\nQuestion: Please determine whether the above-mentioned Java function has any defects?\nA. Yes, it has defects\nB. No, it doesn't have defects\nAnswer:"""
    PROMPT_CHAT_DEFAULT_FS = """```java\n{code_context}\n```\n\nI want you to act as a code defect detector, where I'll provide you with a Java function and it will be your responsibility to analyze it for potential issues based on the provided abstract Java class context information and the following function code. Let's get started with our first potentially flawed Java function:\n\n```java\n{comment}\n{code}\n```\nPlease respond with either "A. Yes, there are defects" or "B. No, there are no defects" based on your assessment."""
    
    # PROMPT_COMPLETE = """/**\n* Counts the number of set bits in the binary representation of a given integer.\n*/\npublic static int bitcount(int n) {{\n    int count = 0;\n    while (n != 0) {{\n        n = (n ^ (n - 1));\n        count++;\n    }}\n    return count;\n}}\nQuestion: Please determine whether the above-mentioned Java function has any defects?\nA. Yes, it has defects\nB. No, it doesn't have defects\nAnswer: A. Yes, it has defects\n\n/**\n * Reverses the order of a singly linked list.\n */\npublic static Node reverse_linked_list(Node node) {{\n    Node prevnode = null;\n    Node nextnode;\n    while (node != null) {{\n        nextnode = node.getSuccessor();\n        node.setSuccessor(prevnode);\n        prevnode = node;\n        node = nextnode;\n    }}\n    return prevnode;\n}}\nQuestion: Please determine whether the above-mentioned Java function has any defects?\nA. Yes, it has defects\nB. No, it doesn't have defects\nAnswer: B. No, it doesn't have defects\n\n{code_context}\n\n You are a professional Java programmer, please identify any defects in the function named `{function_name}` based on the provided abstract Java class context information and the following function code.\n{comment}\n{code}\nQuestion: Please determine whether the above-mentioned Java function has any defects?\nA. Yes, it has defects\nB. No, it doesn't have defects\nAnswer:"""
    # PROMPT_CHAT = """```java\n{code_context}\n```\n\nI want you to act as a code defect detector, where I'll provide you with a Java function and it will be your responsibility to analyze it for potential issues based on the provided abstract Java class context information and the following function code. Please respond with either "A. Yes, there are defects" or "B. No, there are no defects" based on your assessment. Let's get started with our first potentially flawed Java function:\n\n```java\n{comment}\n{code}\n```\n"""
    
    def build_example(project, bug_id, example, comment, code, defective, context, context_length):
        if context["few_shot"] in [-1, -2]:
            import_context = example["import_context"] if context["import_context"] else ""
            class_signature = example["class_signature"] if context["class_signature"] else ""
            class_field_context = example["class_field_context"] if context["class_field_context"] else ""
            class_function_signature_context = example["class_function_signature_context"] if context["class_function_signature_context"] else ""
            
            becut, code_context = get_code_context(import_context, class_signature, 
                                            class_field_context, class_function_signature_context,
                                            context_length)
            if becut:
                code_context = "\n".join(code_context.split("\n")[1:])
            
            if context["few_shot"] == -2:
                prompt_chat = PROMPT_CHAT_DEFAULT_FS.format(
                        code_context=code_context,
                        comment=comment, code=code
                    )
                prompt_complete = PROMPT_COMPLETE_DEFAULT_FS.format(
                        code_context=code_context,
                        comment=comment, code=code,
                        function_name=example["function_name"],
                    )
            else:
                prompt_chat = PROMPT_CHAT.format(
                        code_context=code_context,
                        comment=comment, code=code
                    )
                prompt_complete = PROMPT_COMPLETE.format(
                        code_context=code_context,
                        comment=comment, code=code,
                        function_name=example["function_name"],
                    )
        elif context["few_shot"] in [0,1,2,3,4]:
            prompt_complete = PROMPT_COMPLETE_FS[context["few_shot"]].format(comment=comment, code=code)
            prompt_chat = PROMPT_CHAT_FS[context["few_shot"]].format(
                code_context="", comment=comment, code=code)
            assert prompt_chat[0] == "I"
        else:
            raise NotImplementedError()
        prompted_example = {"task_id": hash_string(code), "project":project, "bug_id":bug_id, 
                "code":code, "function_signature":example["function_signature"],
                "prompt_complete":prompt_complete, "prompt_chat":prompt_chat, "defective":defective}
        return prompted_example
    
    if context is None:
        context = {
            "few_shot": True,
            "import_context": True,
            "class_signature": True,
            "class_field_context": True,
            "class_function_signature_context": True,
        }
        
        
    function_data = json.load(open(os.path.join(FILE_DIR, 'data', 'task_defectdetection_single_function.json')))
    print("Total number of tasks: ", len(function_data))
    
    prompted_data = []
    
    for example in function_data:
        project = example["project_id"]
        bug_id = example["bug_id"]
        
        buggy_example = list(example["buggy_function"].values())[0]
        fixed_example = list(example["fixed_function"].values())[0]
        buggy = buggy_example["function_body"]
        fix = fixed_example["function_body"]
        buggy_comment = buggy_example["function_comment"]
        fix_comment = fixed_example["function_comment"]
        
        prompted_example = build_example(project, bug_id, buggy_example, buggy_comment, buggy, True, context, context_length)
        prompted_data.append(prompted_example)
        prompted_example = build_example(project, bug_id, fixed_example, fix_comment, fix, False, context, context_length)
        prompted_data.append(prompted_example)
        
    prompted_data.sort(key=lambda x: x["task_id"])
    dataset = {"code_ucb_defectdetection": prompted_data}
    json.dump(dataset, open(os.path.join(FILE_DIR, 'data', f'task_defectdetection_bench_{save_suffix}.json'), 'w'), indent=4, ensure_ascii=False)
    

def generate_dataset(FILE_DIR):
    # context = {
    #         "few_shot": True,
    #     }
    # process_get_prompt(FILE_DIR, context, context_length=3072, save_suffix="fs|3072")
        
    # save_suffix_list = ["0100|3072", "1100|3072", "0110|3072",
    #                     "0101|3072", "1111|3072", "1111|2048",
    #                     "1111|1024", "1111|512",]
    save_suffix_list = ["1111|2048"]
    for save_suffix in save_suffix_list:
        context_str, context_length = save_suffix.split("|")
        context = {
            "few_shot": -1,
            "import_context": int(context_str[0]),
            "class_signature": int(context_str[1]),
            "class_field_context": int(context_str[2]),
            "class_function_signature_context": int(context_str[3]),
        }
        process_get_prompt(FILE_DIR, context, context_length=int(context_length), save_suffix=save_suffix)

def generate_default_dataset(FILE_DIR):
    context = {
        "few_shot": 4,
    }
    process_get_prompt(FILE_DIR, context, context_length=2048, save_suffix="default_table3|2048")

def generate_fs_dataset(FILE_DIR):
    context = {
        "few_shot": 0,
    }
    process_get_prompt(FILE_DIR, context, context_length=2048, save_suffix="fs0|2048")
    context = {
        "few_shot": 1,
    }
    process_get_prompt(FILE_DIR, context, context_length=2048, save_suffix="fs1|2048")
    context = {
        "few_shot": 2,
    }
    process_get_prompt(FILE_DIR, context, context_length=2048, save_suffix="fs2|2048")
    context = {
        "few_shot": 3,
    }
    process_get_prompt(FILE_DIR, context, context_length=2048, save_suffix="fs3|2048")
    context = {
        "few_shot": 4,
    }
    process_get_prompt(FILE_DIR, context, context_length=2048, save_suffix="fs4|2048")

def generate_default_with_fs_dataset(FILE_DIR):
    context_str, context_length = "1111|2048".split("|")
    context = {
        "few_shot": -2,
        "import_context": int(context_str[0]),
        "class_signature": int(context_str[1]),
        "class_field_context": int(context_str[2]),
        "class_function_signature_context": int(context_str[3]),
    }
    process_get_prompt(FILE_DIR, context, context_length=2048, save_suffix="default|2048")


def main():
    FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    process_extract_info(FILE_DIR)
    process_extract_function(FILE_DIR)
    
    generate_default_dataset(FILE_DIR)
    generate_fs_dataset(FILE_DIR)
    generate_default_with_fs_dataset(FILE_DIR)
    
if __name__ == '__main__':
    main()    