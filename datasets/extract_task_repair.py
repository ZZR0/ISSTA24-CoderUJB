from concurrent.futures import ProcessPoolExecutor, as_completed
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

def process_worker(item):
    FILE_DIR, example = item
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
    
    try:
        import_context = code_ast_buggy.get_import_context_source()
        class_signature = code_ast_buggy.get_class_signature_context_source()
        class_field_context = code_ast_buggy.get_class_field_context_source()
        class_function_signature_context = code_ast_buggy.get_class_functions_signature_context_source()
    except:
        return None
    
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
                "import_context": import_context,
                "class_signature": class_signature,
                "class_field_context": class_field_context,
                "class_function_signature_context": class_function_signature_context
            }
        if change.action == 'add':
            example_processed["fixed_function"][f"{function_name}|{function_star_line}|{function_end_line}"] = func_info
        elif change.action == 'sub':
            example_processed["buggy_function"][f"{function_name}|{function_star_line}|{function_end_line}"] = func_info

    return example_processed

def process_extract_info(FILE_DIR):
    json_data = json.load(open(os.path.join(FILE_DIR, 'data', 'task_info.json')))
    processed_data = []
    tasks = [(FILE_DIR, example) for example in json_data]
    # tasks = [task for task in tasks if task[1]["project_id"] == "Math" and task[1]["bug_id"] == "90"]
    # print(tasks)
    
    # for task in tasks:
    #     processed_data.append(process_worker(*task))
        
    processed_data = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()//4) as executor:
        future_tasks = {executor.submit(process_worker, task) for task in tasks}
        with tqdm(as_completed(future_tasks), total=len(tasks), desc="Processing...") as progress_bar:
            for future in progress_bar:
                processed_data.append(future.result())
                    
    processed_data = [example for example in processed_data if example is not None]
    json.dump(processed_data, open(os.path.join(FILE_DIR, 'data', 'task_repair.json'), 'w'), indent=4, ensure_ascii=False)

def process_extract_function(FILE_DIR):
    
    task_repair = json.load(open(os.path.join(FILE_DIR, 'data', 'task_repair.json')))
    print("Total number of tasks: ", len(task_repair))
    
    single_function_tasks = []
    multi_function_tasks = []
    
    for task in task_repair:
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
    
    json.dump(single_function_tasks, open(os.path.join(FILE_DIR, 'data', 'task_repair_single_function.json'), 'w'), indent=4, ensure_ascii=False)
    json.dump(multi_function_tasks, open(os.path.join(FILE_DIR, 'data', 'task_repair_multi_function.json'), 'w'), indent=4, ensure_ascii=False)

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

def get_code_context_order(import_context, class_signature, 
                     class_field_context, class_function_signature_context, context_length, 
                     improve_list=["class_function_signature_context", "class_field_context", "import_context", "class_signature"]):
    context_improve_list = {}
    for improve_key in improve_list:
        if improve_key == "import_context":
            context_improve_list[improve_key] = import_context
        elif improve_key == "class_signature":
            context_improve_list[improve_key] = class_signature
        elif improve_key == "class_field_context":
            context_improve_list[improve_key] = class_field_context
        elif improve_key == "class_function_signature_context":
            context_improve_list[improve_key] = class_function_signature_context
    
    code_context_format = "{import_context}\n\n{class_signature} {{\n{class_field_context}\n\n{class_function_signature_context}\n}}"
    for remove_key in context_improve_list:
        code_context = code_context_format.format(
            import_context=context_improve_list["import_context"],
            class_signature=context_improve_list["class_signature"],
            class_field_context=context_improve_list["class_field_context"],
            class_function_signature_context=context_improve_list["class_function_signature_context"]
        )
        code_context = code_context.strip()
        cut_code_context = tokenizer.convert_tokens_to_string(tokenizer.tokenize(code_context)[-context_length:])
        cut = cut_code_context != code_context
        if cut:
            context_improve_list[remove_key] = ""
        else:
            break
    return cut, cut_code_context

def process_get_prompt(FILE_DIR, context=None, context_length=1024, save_suffix="", improve_list=[]):
    if context is None:
        context = {
            "few_shot": -1,
            "import_context": True,
            "class_signature": True,
            "class_field_context": True,
            "class_function_signature_context": True,
        }
    PROMPT_COMPLETE_FS0 = """// Provide a fix for the buggy function\n\n// Buggy Function\n{bug}\n\n// Fixed Function\n"""
    PROMPT_CHAT_FS0 = """You are a professional Java programmer, and I will provide you with a buggy function named `{function_name}`, without specifying the location of the bug. Your task is to identify and fix any issues within the given function based on the provided abstract Java class context information along with the following buggy function.\n\n```java\n{bug}\n```\n"""
    PROMPT_COMPLETE_FS1 = """// Provide a fix for the buggy function\n\n// Buggy Function\npublic static int bitcount(int n) {{\n    int count = 0;\n    while (n != 0) {{\n        n = (n ^ (n - 1));\n        count++;\n    }}\n    return count;\n}}\n\n// Fixed Function\npublic static int bitcount(int n) {{\n    int count = 0;\n    while (n != 0) {{\n        n = (n & (n - 1));\n        count++;\n    }}\n    return count;\n}}\n\n\n// Provide a fix for the buggy function\n\n// Buggy Function\n{bug}\n\n// Fixed Function\n"""
    PROMPT_CHAT_FS1 = """You are a professional Java programmer, and I will provide you with a buggy function named `{function_name}`, without specifying the location of the bug. Your task is to identify and fix any issues within the given function based on the provided abstract Java class context information along with the following buggy function.\n\n```java\n{bug}\n```\n"""
    PROMPT_COMPLETE_FS2 = """// Provide a fix for the buggy function\n\n// Buggy Function\npublic static int bitcount(int n) {{\n    int count = 0;\n    while (n != 0) {{\n        n = (n ^ (n - 1));\n        count++;\n    }}\n    return count;\n}}\n\n// Fixed Function\npublic static int bitcount(int n) {{\n    int count = 0;\n    while (n != 0) {{\n        n = (n & (n - 1));\n        count++;\n    }}\n    return count;\n}}\n\n\n// Provide a fix for the buggy function\n\n// Buggy Function\npublic static Node reverse_linked_list(Node node) {{\n    Node prevnode = null;\n    Node nextnode;\n    while (node != null) {{\n        nextnode = node.getSuccessor();\n        node.setSuccessor(prevnode);\n        node = nextnode;\n    }}\n    return prevnode;\n}}\n\n// Fixed Function\npublic static Node reverse_linked_list(Node node) {{\n    Node prevnode = null;\n    Node nextnode;\n    while (node != null) {{\n        nextnode = node.getSuccessor();\n        node.setSuccessor(prevnode);\n        prevnode = node;\n        node = nextnode;\n    }}\n    return prevnode;\n}}\n\n\n// Provide a fix for the buggy function\n\n// Buggy Function\n{bug}\n\n// Fixed Function\n"""
    PROMPT_CHAT_FS2 = """You are a professional Java programmer, and I will provide you with a buggy function named `{function_name}`, without specifying the location of the bug. Your task is to identify and fix any issues within the given function based on the provided abstract Java class context information along with the following buggy function.\n\n```java\n{bug}\n```\n"""
    PROMPT_COMPLETE_FS3 = """// Provide a fix for the buggy function\n\n// Buggy Function\npublic static int bitcount(int n) {{\n    int count = 0;\n    while (n != 0) {{\n        n = (n ^ (n - 1));\n        count++;\n    }}\n    return count;\n}}\n\n// Fixed Function\npublic static int bitcount(int n) {{\n    int count = 0;\n    while (n != 0) {{\n        n = (n & (n - 1));\n        count++;\n    }}\n    return count;\n}}\n\n\n// Provide a fix for the buggy function\n\n// Buggy Function\npublic static Node reverse_linked_list(Node node) {{\n    Node prevnode = null;\n    Node nextnode;\n    while (node != null) {{\n        nextnode = node.getSuccessor();\n        node.setSuccessor(prevnode);\n        node = nextnode;\n    }}\n    return prevnode;\n}}\n\n// Fixed Function\npublic static Node reverse_linked_list(Node node) {{\n    Node prevnode = null;\n    Node nextnode;\n    while (node != null) {{\n        nextnode = node.getSuccessor();\n        node.setSuccessor(prevnode);\n        prevnode = node;\n        node = nextnode;\n    }}\n    return prevnode;\n}}\n\n\n// Provide a fix for the buggy function\n\n// Buggy Function\npublic static int max_sublist_sum(int[] arr) {{\n    int max_ending_here = 0;\n    int max_so_far = 0;\n\n    for (int x : arr) {{\n        max_ending_here = max_ending_here + x;\n        max_so_far = Math.max(max_so_far, max_ending_here);\n    }}\n\n    return max_so_far;\n}}\n\n// Fixed Function\npublic static int max_sublist_sum(int[] arr) {{\n    int max_ending_here = 0;\n    int max_so_far = 0;\n\n    for (int x : arr) {{\n        max_ending_here = Math.max(0,max_ending_here + x);\n        max_so_far = Math.max(max_so_far, max_ending_here);\n    }}\n\n    return max_so_far;\n}}\n\n\n// Provide a fix for the buggy function\n\n// Buggy Function\n{bug}\n\n// Fixed Function\n"""
    PROMPT_CHAT_FS3 = """You are a professional Java programmer, and I will provide you with a buggy function named `{function_name}`, without specifying the location of the bug. Your task is to identify and fix any issues within the given function based on the provided abstract Java class context information along with the following buggy function.\n\n```java\n{bug}\n```\n"""
    PROMPT_COMPLETE_FS4 = """// Provide a fix for the buggy function\n\n// Buggy Function\npublic static int bitcount(int n) {{\n    int count = 0;\n    while (n != 0) {{\n        n = (n ^ (n - 1));\n        count++;\n    }}\n    return count;\n}}\n\n// Fixed Function\npublic static int bitcount(int n) {{\n    int count = 0;\n    while (n != 0) {{\n        n = (n & (n - 1));\n        count++;\n    }}\n    return count;\n}}\n\n\n// Provide a fix for the buggy function\n\n// Buggy Function\npublic static Node reverse_linked_list(Node node) {{\n    Node prevnode = null;\n    Node nextnode;\n    while (node != null) {{\n        nextnode = node.getSuccessor();\n        node.setSuccessor(prevnode);\n        node = nextnode;\n    }}\n    return prevnode;\n}}\n\n// Fixed Function\npublic static Node reverse_linked_list(Node node) {{\n    Node prevnode = null;\n    Node nextnode;\n    while (node != null) {{\n        nextnode = node.getSuccessor();\n        node.setSuccessor(prevnode);\n        prevnode = node;\n        node = nextnode;\n    }}\n    return prevnode;\n}}\n\n\n// Provide a fix for the buggy function\n\n// Buggy Function\npublic static int max_sublist_sum(int[] arr) {{\n    int max_ending_here = 0;\n    int max_so_far = 0;\n\n    for (int x : arr) {{\n        max_ending_here = max_ending_here + x;\n        max_so_far = Math.max(max_so_far, max_ending_here);\n    }}\n\n    return max_so_far;\n}}\n\n// Fixed Function\npublic static int max_sublist_sum(int[] arr) {{\n    int max_ending_here = 0;\n    int max_so_far = 0;\n\n    for (int x : arr) {{\n        max_ending_here = Math.max(0,max_ending_here + x);\n        max_so_far = Math.max(max_so_far, max_ending_here);\n    }}\n\n    return max_so_far;\n}}\n\n\n// Provide a fix for the buggy function\n\n// Buggy Function\npublic static double sqrt(double x, double epsilon) {{\n    double approx = x / 2f;\n    while (Math.abs(x-approx) > epsilon) {{\n        approx = 0.5f * (approx + x / approx);\n    }}\n    return approx;\n}}\n\n// Fixed Function\npublic static double sqrt(double x, double epsilon) {{\n    double approx = x / 2d;\n    while (Math.abs(x-approx*approx) > epsilon) {{\n        approx = 0.5d * (approx + x / approx);\n    }}\n    return approx;\n}}\n\n\n// Provide a fix for the buggy function\n\n// Buggy Function\n{bug}\n\n// Fixed Function\n"""
    PROMPT_CHAT_FS4 = """You are a professional Java programmer, and I will provide you with a buggy function named `{function_name}`, without specifying the location of the bug. Your task is to identify and fix any issues within the given function based on the provided abstract Java class context information along with the following buggy function.\n\n```java\n{bug}\n```\n"""
    
    PROMPT_COMPLETE_FS = [
        PROMPT_COMPLETE_FS0, PROMPT_COMPLETE_FS1, PROMPT_COMPLETE_FS2, PROMPT_COMPLETE_FS3, PROMPT_COMPLETE_FS4
    ]
    PROMPT_CHAT_FS = [
        PROMPT_CHAT_FS0, PROMPT_CHAT_FS1, PROMPT_CHAT_FS2, PROMPT_CHAT_FS3, PROMPT_CHAT_FS4
    ]
    
    PROMPT_COMPLETE = """{code_context}\n\n// You are a professional Java programmer, please fix the bug in the function named {function_name} based on the provided abstract Java class context information and the following buggy function.\n// Buggy Function\n{bug}\n\n// Fixed Function\n"""
    PROMPT_CHAT = """```java\n{code_context}\n```\n\nYou are a professional Java programmer, and I will provide you with a buggy function named `{function_name}`, without specifying the location of the bug. Your task is to identify and fix any issues within the given function based on the provided abstract Java class context information along with the following buggy function.\n\n```java\n{bug}\n```\n"""
    
    function_data = json.load(open(os.path.join(FILE_DIR, 'data', 'task_repair_single_function.json')))
    print("Total number of tasks: ", len(function_data))
    
    prompted_data = []
    
    for example in function_data:
        project = example["project_id"]
        bug_id = example["bug_id"]
        
        testmethods = example["tests_trigger"]
        source_dir = example["src_classes"]
        tests_dir = example["src_tests"]
        location = list(example["fixed_source"].keys())[0]
        
        buggy_example = list(example["buggy_function"].values())[0]
        fixed_example = list(example["fixed_function"].values())[0]
        buggy = buggy_example["function_body"]
        fix = fixed_example["function_body"]
        function_signature = fixed_example["function_signature"]
        start = fixed_example["function_star_line"]
        end = fixed_example["function_end_line"]
        
        comment = buggy_example["function_comment"]
        code_context = ""
        
        if context["few_shot"] == -1:
            import_context = buggy_example["import_context"] if context["import_context"] else ""
            class_signature = buggy_example["class_signature"] if context["class_signature"] else ""
            class_field_context = buggy_example["class_field_context"] if context["class_field_context"] else ""
            class_function_signature_context = buggy_example["class_function_signature_context"] if context["class_function_signature_context"] else ""
            
            if len(improve_list) > 0:
                becut, code_context = get_code_context_order(import_context, class_signature, 
                                                class_field_context, class_function_signature_context,
                                                context_length, improve_list=improve_list)
            else:
                becut, code_context = get_code_context(import_context, class_signature, 
                                                    class_field_context, class_function_signature_context,
                                                    context_length)
            if becut:
                code_context = "\n".join(code_context.split("\n")[1:])
                
            prompt_chat = PROMPT_CHAT.format(
                    code_context=code_context,
                    function_name=buggy_example["function_name"],
                    bug=f'{comment}\n{buggy}'
                )
            code_context = "// "+ "\n// ".join(code_context.split("\n"))
            prompt_complete = PROMPT_COMPLETE.format(
                    code_context=code_context,
                    function_name=buggy_example["function_name"],
                    bug=f'{comment}\n{buggy}'
                )
        elif context["few_shot"] in [0,1,2,3,4]:
            prompt_complete = PROMPT_COMPLETE_FS[context["few_shot"]].format(bug=f'{comment}\n{buggy}')
            prompt_chat = PROMPT_CHAT_FS[context["few_shot"]].format(
                    function_name=buggy_example["function_name"],
                    bug=f'{comment}\n{buggy}'
                )
            assert prompt_chat[0] == "Y"
        
        source = list(example["fixed_source"].values())[0]
        
        test_sources = get_test_source(os.path.join(FILE_DIR, "projects", project, bug_id+"f"), tests_dir, testmethods)
        
        indent = fix[:len(fix)-len(fix.strip())]
        prompted_example = {"task_id": hash_string(project+str(bug_id)+buggy), "project":project, "bug_id":bug_id, 
            "testmethods": testmethods, "source_dir":source_dir, "location":location,
            "start": start, "end": end, "buggy":buggy, "fix":fix, "function_signature":function_signature,
            "prompt_complete":prompt_complete, "prompt_chat":prompt_chat, 
            "import_context": buggy_example["import_context"], "class_signature": buggy_example["class_signature"],
            "class_field_context": buggy_example["class_field_context"], "class_function_signature_context": buggy_example["class_function_signature_context"],
            "code_context": code_context,
            "test_sources":test_sources, "source":source, "indent":indent}
        
        prompted_data.append(prompted_example)
    prompted_data.sort(key=lambda x: x["task_id"])
    dataset = {"code_ucb_repair": prompted_data}
    json.dump(dataset, open(os.path.join(FILE_DIR, 'data', f'task_repair_bench_{save_suffix}.json'), 'w'), indent=4, ensure_ascii=False)
    
def process_get_correct_result(FILE_DIR, save_suffix=""):
    data = json.load(open(os.path.join(FILE_DIR, 'data', 'task_repair_bench.json')))
    correct_result = []
    for idx, example in enumerate(data["code_ucb_repair"]):
        result = {
            "task_id": idx,
            "outputs": [example["prompt_complete"] + "\n" + example["fix"]]*10
        }
        correct_result.append(result)
        
    json.dump(correct_result, open(os.path.join(FILE_DIR, 'data', f'task_repair_correct_result_{save_suffix}.json'), 'w'), indent=4, ensure_ascii=False)


def generate_dataset(FILE_DIR):
    context = {
            "few_shot": True,
        }
    process_get_prompt(FILE_DIR, context, context_length=3072, save_suffix="fs|3072")
    process_get_correct_result(FILE_DIR, save_suffix="fs|3072")
        
    save_suffix_list = ["0100|3072", "1100|3072", "0110|3072",
                        "0101|3072", "1111|3072", "1111|2048",
                        "1111|1024", "1111|512",]
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
        process_get_correct_result(FILE_DIR, save_suffix=save_suffix)

def generate_order_dataset(FILE_DIR):
    save_suffix_list = ["1111|3072|order", "1111|2048|order",
                        "1111|1024|order", "1111|512|order",]
    for save_suffix in save_suffix_list:
        context_str = save_suffix.split("|")[0]
        context_length = save_suffix.split("|")[1]
        context = {
            "few_shot": -1,
            "import_context": int(context_str[0]),
            "class_signature": int(context_str[1]),
            "class_field_context": int(context_str[2]),
            "class_function_signature_context": int(context_str[3]),
        }
        process_get_prompt(FILE_DIR, context, context_length=int(context_length), save_suffix=save_suffix,
                           improve_list=["import_context", "class_field_context", "class_function_signature_context", "class_signature"])
        process_get_correct_result(FILE_DIR, save_suffix=save_suffix)

def generate_default_dataset(FILE_DIR):
    save_suffix_list = ["default|2048"]
    for save_suffix in save_suffix_list:
        context_str, context_length = save_suffix.split("|")
        context = {
            "few_shot": -1,
            "import_context": 1,
            "class_signature": 1,
            "class_field_context": 1,
            "class_function_signature_context": 1,
        }
        process_get_prompt(FILE_DIR, context, context_length=int(context_length), save_suffix=save_suffix)
        process_get_correct_result(FILE_DIR, save_suffix=save_suffix)

def generate_fs_dataset(FILE_DIR):
    save_suffix_list = ["fs0|2048", "fs1|2048", "fs2|2048", "fs3|2048", "fs4|2048"]
    for save_suffix in save_suffix_list:
        context_str, context_length = save_suffix.split("|")
        context = {
            "few_shot": int(context_str[-1]),
            "import_context": 0,
            "class_signature": 0,
            "class_field_context": 0,
            "class_function_signature_context": 0,
        }
        process_get_prompt(FILE_DIR, context, context_length=int(context_length), save_suffix=save_suffix)
        process_get_correct_result(FILE_DIR, save_suffix=save_suffix)

def main():
    FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    process_extract_info(FILE_DIR)
    process_extract_function(FILE_DIR)
    
    generate_default_dataset(FILE_DIR)
    generate_fs_dataset(FILE_DIR)
    

if __name__ == '__main__':
    main()    