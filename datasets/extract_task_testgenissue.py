import ast
from concurrent.futures import ProcessPoolExecutor
import os
import json
import random
import re
import subprocess
from code_parser import Code_AST
from tqdm import tqdm
import chardet
from transformers import AutoTokenizer
import pandas as pd
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from utils import gen_worker, read_file, check_is_complete_function, \
                    get_code_prefix, hash_string

random.seed(42)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def get_formatted_report_libro(report):
    rep_title, rep_content = report["title"], report["description"]
    rep_title = BeautifulSoup(rep_title.strip(), 'html.parser').get_text()
    rep_content = md(rep_content.strip())
    # rep_content = BeautifulSoup(
    #         rep_content.strip(), 'html.parser').get_text()
    formatted_report = {}
    formatted_report["issue_id"] = report["issue_id"]
    formatted_report["title"] = rep_title
    formatted_report["discussion"] = rep_content
    # print(formatted_report["discussion"])
    # input("Continue?")
    return formatted_report

def get_url(FILE_DIR, project, bug_id):
    project_info = pd.read_csv(os.path.join(FILE_DIR, "projects", project, "query_info.csv"))
    return project_info.loc[project_info["bug.id"] == int(bug_id)]["report.url"].values[0]

def get_test_dir(FILE_DIR, project, bug_id):
    data = json.load(open(os.path.join(FILE_DIR, "data", "task_info.json")))
    for item in data:
        if item["project_id"] == project and item["bug_id"] == bug_id:
            return item["src_tests"]
    return None

def get_src_dir(FILE_DIR, project, bug_id):
    data = json.load(open(os.path.join(FILE_DIR, "data", "task_info.json")))
    for item in data:
        if item["project_id"] == project and item["bug_id"] == bug_id:
            return item["src_classes"]
    return None

def get_classes_modified(FILE_DIR, project, bug_id):
    data = json.load(open(os.path.join(FILE_DIR, "data", "task_info.json")))
    for item in data:
        if item["project_id"] == project and item["bug_id"] == bug_id:
            return item["classes_modified"]
    return None


def process_issues_info(FILE_DIR):
    all_issue_tests = []
    
    data = json.load(open(os.path.join(FILE_DIR, "data", "task_repair_bench_default|2048.json")))
    for example in tqdm(data["code_ujb_repair"]):
        example_id = f"{example['project']}-{example['bug_id']}"
        
        testmethods = example["testmethods"]
        # testmethods = [tm for tm in testmethods if "issue" in tm.split("::")[-1].lower() or "bug" in tm.split("::")[-1].lower()]
        # if len(testmethods) == 0: continue
        
        # issue_url = get_url(FILE_DIR, example["project"], example["bug_id"])
        # if issue_url == "UNKNOWN": continue
        
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
    json.dump(all_issue_tests, open(os.path.join(FILE_DIR, "data", "task_testgenissue_info.json"), "w"), indent=4)

def get_raw_report(FILE_DIR, project_id, bug_id, report_url):
    file_type = "json"
    if not report_url.endswith(".json"):
        report_path = os.path.join(FILE_DIR, "issue_report", f"{project_id}_{str(bug_id)}" + '.html')
        cmd = ['wget', '-O', report_path, report_url]
        if "github" in report_url:
            file_type = "github_html"
        elif "sourceforge" in report_url:
            file_type = "sourceforge_html"
        elif "apache" in report_url:
            file_type = "apache_html"
        else:
            raise Exception("Unknown report type")
    else:
        file_type = "json"
        report_path = os.path.join(FILE_DIR, "issue_report", f"{project_id}_{str(bug_id)}" + '.json')
        cmd = ['wget', '-O', report_path, report_url]
    
    if not os.path.exists(report_path):
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    return report_path, file_type

def get_formatted_report(report_path, file_type):
    if file_type == "json":
        report = json.loads(read_file(report_path))
    elif "html" in file_type:
        report = json.loads(read_file(report_path.replace(".html", ".json")))
    else:
        raise Exception("Unknown report type")
    
    # print(report.keys())
    formatted_report = {}
    formatted_report["issue_id"] = report["id"]
    formatted_report["title"] = report["summary"]
    formatted_report["discussion"] = report["comments"][0]["content"]
    return formatted_report

def get_function(class_file_path, function_name):
    
    class_file = read_file(class_file_path)
    try:
        class_ast = Code_AST(code=class_file, lang="java").ast
    except:
        return None, None, None, None
    functions = [function for function in class_ast.get_functions() if function.get_function_name() == function_name]
    assert len(functions) == 1, f"Cannot find function {function_name} in class {class_file_path}"
    function = functions[0]
    
    comment = function.get_function_comment() if function is not None else None
    
    return function, comment, class_file, class_ast

def get_insert_place(class_file_path):
    class_file = read_file(class_file_path)
    try:
        class_ast = Code_AST(code=class_file, lang="java").ast
    except:
        return None, None
    
    functions = [function for function in class_ast.get_functions() if check_is_complete_function(function)]
    assert len(functions) > 1, f"Cannot find function in class {class_file_path}"
    insert_place = []
    prev_end = functions[0].end_line
    for function in functions[1:]:
        comments = function.get_function_comment_nodes()
        if len(comments) == 0:
            start_line = function.start_line
        else:
            start_line = [comment.start_line for comment in comments]
            start_line = min(start_line)
        insert_place.extend([i for i in range(prev_end+1, start_line)])
        prev_end = function.end_line
    
    assert len(insert_place) > 0, f"Cannot find function in class {class_file_path}"
    
    return insert_place[-1], class_file

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
    data = json.load(open(os.path.join(FILE_DIR, "data", "task_testgenissue_info.json")))
    all_report = json.load(open(os.path.join(FILE_DIR, "bug_report.json")))
    all_report = {f"{item['project']}-{item['bug_id']}": item for item in all_report}
    for issue in tqdm(data):
        # report_path, file_type = get_raw_report(FILE_DIR, issue["project_id"], issue["bug_id"], issue["issue_url"])
        # report = get_formatted_report(report_path, file_type)
        report_key = f'{issue["project_id"]}-{issue["bug_id"]}'
        if not report_key in all_report:
            print(report_key)
            continue
        report = get_formatted_report_libro(all_report[report_key])
        
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
    
    json.dump(processed_tasks, open(os.path.join(FILE_DIR, "data", "task_testgenissue.json"), "w"), indent=4)

def process_filter_prompt(FILE_DIR):
    data = json.load(open(os.path.join(FILE_DIR, 'data', 'task_testgenissue.json')))
    function_name_set = set()
    filtered_data_dict = {}
    for example in data:
        if example["function_name"] in function_name_set:
            continue
        function_name_set.add(example["function_name"])
        
        if example["function_start_line_fixed"]-example["function_end_line_fixed"] > 30: continue
        # if len(example["comment"]) < 128 or (not "/*" in example["comment"]) or (not "*/" in example["comment"]): continue
        
        if example["project_id"] not in filtered_data_dict:
            filtered_data_dict[example["project_id"]] = []
        filtered_data_dict[example["project_id"]].append(example)
    
    filtered_data = []
    for project in filtered_data_dict:
        filtered_data.extend(filtered_data_dict[project])
    
    # filtered_data = filtered_data[:10]
    print("Total number of tasks after filtering: ", len(filtered_data))
    json.dump(filtered_data, open(os.path.join(FILE_DIR, 'data', 'task_testgenissue_filtered.json'), 'w'), indent=4, ensure_ascii=False)

def process_get_better_comment(FILE_DIR, load=False, model_id="gpt-3.5-turbo"):
    data = json.load(open(os.path.join(FILE_DIR, 'data', 'task_testgenissue_filtered.json')))
    load_path = os.path.join(FILE_DIR, 'data', 'task_testgenissue_better_comment_dict.json')
    processed_tasks = []
    unprocessed_tasks = []
    if load and os.path.exists(load_path):
        task_to_data = json.load(open(load_path))
        for example in data:
            task_id = example["task_id"]
            if task_id in task_to_data:
                example["better_comment"] = task_to_data[task_id]["better_comment"]
                example["content_comment"] = task_to_data[task_id]["content_comment"]
                processed_tasks.append(example)
            else:
                unprocessed_tasks.append(example)
    else:
        unprocessed_tasks = data
    
    if len(unprocessed_tasks) != 0:
        GEN_PROMPT = """I am developing a coding exercise for students majoring in computer science, where they will be responsible for generating Java test cases using context derived from an issued report, along with the matching function signature connoting the test cases.\n\nYour task is to recognize the intent of the issue report, and generate a function comments for the test case, ensuring they accurately portray the intended test cases. This heightened clarity should aid the students in comprehending the assignment and assist them in creating suitable Java test cases.\n\n[Issue Report]\n```markdown\n{issue_report}\n```\n\n[Test Case To Generate]\n```java\n{input}\n```"""
        tasks = []
        for example in data:
            function = example["function"]
            comment = example["function_comment"]
            signature = example["function_signature"]
            tasks.append((example, model_id, GEN_PROMPT.format(
                                                    issue_report=example["issue_report"],
                                                    input=f'{comment}\n{example["indent"]}{signature}',
                                                )))

        # processed_tasks = []
        # for task in tqdm(tasks, desc="Get Better Comment"):
        #     processed_tasks.append(gen_worker(task))
            
        with ProcessPoolExecutor(max_workers=4) as executor:
            processed_tasks = list(tqdm(executor.map(gen_worker, tasks), total=len(tasks), desc="Get Better Comment"))
        
    json.dump(processed_tasks, open(os.path.join(FILE_DIR, 'data', 'task_testgenissue_filtered_with_better_comment.json'), 'w'), indent=4, ensure_ascii=False)

    task_to_data = {}
    for task in processed_tasks:
        task_id = task["task_id"]
        task_to_data[task_id] = {"better_comment":task["better_comment"], "content_comment":task["content_comment"]}
    json.dump(task_to_data, open(os.path.join(FILE_DIR, 'data', 'task_testgenissue_better_comment_dict.json'), 'w'), indent=4, ensure_ascii=False)


def process_final_bench(FILE_DIR):
    data = json.load(open(os.path.join(FILE_DIR, 'data', 'task_testgenissue_filtered_with_better_comment_and_score.json')))
    processed_data = []
    for prompted_example in data:
        if prompted_example["score"] >= 7:
            processed_data.append(prompted_example)
    print("Total Task Case after filtering:", len(processed_data))
    json.dump(processed_data, open(os.path.join(FILE_DIR, 'data', 'task_testgenissue_filtered_final.json'), 'w'), indent=4, ensure_ascii=False)

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
            "class_function_signature_context": True
        }
    # PROMPT_COMPLETE = """// Issue Report:\n{issue_report}\n\n// Abstract Java Class:\n{code_context}\n// You are a professional Java test case writer, please create a test case named `{function_name}` for the issue `{issue_id}`, utilizing the provided issue report information, abstract Java class context information and the following function signature.\n\n{nl_context}"""
    # PROMPT_CHAT = """```markdown\n{issue_report}\n```\n\n```java\n{code_context}\n```\n\nYou are a professional Java test case writer, please create a test case named `{function_name}` for the issue `{issue_id}`, utilizing the provided issue report information, abstract Java class context information and the following function signature.\n\n```java\n{nl_context}\n```\n"""
    
    PROMPT_COMPLETE_FS0 = """// You are a professional Java test case writer, please create a test case named `{function_name}` for the issue `{issue_id}`, utilizing the provided issue report information and the following function signature.\n\n{issue_report}\n\n{nl_context}"""
    PROMPT_CHAT_FS0 = """```markdown\n{issue_report}\n```\n\nYou are a professional Java test case writer, please create a test case named `{function_name}` for the issue `{issue_id}`, utilizing the provided issue report information and the following function signature.\n\n```java\n{nl_context}\n```\n"""
    PROMPT_COMPLETE_FS1 = """// You are a professional Java test case writer, please create a test case named `testIssue504` for the issue `Closure-504`, utilizing the provided issue report information and the following function signature.\n\n// ## Issue-ID: Closure-504\n\n// ## Issue-Title: \n// void function () {{}}(); wrongly identified as having no side effects\n\n// ## Issue-Description: \n// This code results in the execution of the function and should not be identified as having no side effects.\n\npublic void testIssue504() {{\n    args.add("--compilation_level=ADVANCED_OPTIMIZATIONS");\n    test("void function() {{ alert('hi'); }}();",\n            "alert('hi');", CheckSideEffects.USELESS_CODE_ERROR);\n}}\n\n// You are a professional Java test case writer, please create a test case named `{function_name}` for the issue `{issue_id}`, utilizing the provided issue report information and the following function signature.\n\n{issue_report}\n\n{nl_context}"""
    PROMPT_CHAT_FS1 = """```markdown\n{issue_report}\n```\n\nYou are a professional Java test case writer, please create a test case named `{function_name}` for the issue `{issue_id}`, utilizing the provided issue report information and the following function signature.\n\n```java\n{nl_context}\n```\n"""
    PROMPT_COMPLETE_FS2 = """// You are a professional Java test case writer, please create a test case named `testIssue504` for the issue `Closure-504`, utilizing the provided issue report information and the following function signature.\n\n// ## Issue-ID: Closure-504\n\n// ## Issue-Title: \n// void function () {{}}(); wrongly identified as having no side effects\n\n// ## Issue-Description: \n// This code results in the execution of the function and should not be identified as having no side effects.\n\npublic void testIssue504() {{\n    args.add("--compilation_level=ADVANCED_OPTIMIZATIONS");\n    test("void function() {{ alert('hi'); }}();",\n            "alert('hi');", CheckSideEffects.USELESS_CODE_ERROR);\n}}\n\n// You are a professional Java test case writer, please create a test case named `testAtanI` for the issue `Math-MATH-657`, utilizing the provided issue report information and the following function signature.\n\n// ## Issue-ID: Math-MATH-657\n\n// ## Issue-Title: \n// Division by zero\n\n// ## Issue-Description: \n\nIn class Complex, division by zero always returns NaN. I think that it should return NaN only when the numerator is also ZERO, otherwise the result should be INF. See [here](http://en.wikipedia.org/wiki/Riemann_sphere#Arithmetic_operations).\n\n@Test\npublic void testAtanI() {{\n    Assert.assertTrue(Complex.I.atan().isNaN());\n}}\n\n// You are a professional Java test case writer, please create a test case named `{function_name}` for the issue `{issue_id}`, utilizing the provided issue report information and the following function signature.\n\n{issue_report}\n\n{nl_context}"""
    PROMPT_CHAT_FS2 = """```markdown\n{issue_report}\n```\n\nYou are a professional Java test case writer, please create a test case named `{function_name}` for the issue `{issue_id}`, utilizing the provided issue report information and the following function signature.\n\n```java\n{nl_context}\n```\n"""
    PROMPT_COMPLETE_FS3 = """// You are a professional Java test case writer, please create a test case named `testIssue504` for the issue `Closure-504`, utilizing the provided issue report information and the following function signature.\n\n// ## Issue-ID: Closure-504\n\n// ## Issue-Title: \n// void function () {{}}(); wrongly identified as having no side effects\n\n// ## Issue-Description: \n// This code results in the execution of the function and should not be identified as having no side effects.\n\npublic void testIssue504() {{\n    args.add("--compilation_level=ADVANCED_OPTIMIZATIONS");\n    test("void function() {{ alert('hi'); }}();",\n            "alert('hi');", CheckSideEffects.USELESS_CODE_ERROR);\n}}\n\n// You are a professional Java test case writer, please create a test case named `testAtanI` for the issue `Math-MATH-657`, utilizing the provided issue report information and the following function signature.\n\n// ## Issue-ID: Math-MATH-657\n\n// ## Issue-Title: \n// Division by zero\n\n// ## Issue-Description: \n\nIn class Complex, division by zero always returns NaN. I think that it should return NaN only when the numerator is also ZERO, otherwise the result should be INF. See [here](http://en.wikipedia.org/wiki/Riemann_sphere#Arithmetic_operations).\n\n@Test\npublic void testAtanI() {{\n    Assert.assertTrue(Complex.I.atan().isNaN());\n}}\n\n// You are a professional Java test case writer, please create a test case named `testIssue582` for the issue `Closure-582`, utilizing the provided issue report information and the following function signature.\n\n// ## Issue-ID: Closure-582\n\n// ## Issue-Title: \n// -0.0 becomes 0 even in whitespace mode\n\n// ## Issue-Description: \n// Affects dart: http://code.google.com/p/dart/issues/detail?id=146\npublic void testIssue582() {{\n    assertPrint("var x = -0.0;", "var x=-0.0");\n}}\n\n// You are a professional Java test case writer, please create a test case named `{function_name}` for the issue `{issue_id}`, utilizing the provided issue report information and the following function signature.\n\n{issue_report}\n\n{nl_context}"""
    PROMPT_CHAT_FS3 = """```markdown\n{issue_report}\n```\n\nYou are a professional Java test case writer, please create a test case named `{function_name}` for the issue `{issue_id}`, utilizing the provided issue report information and the following function signature.\n\n```java\n{nl_context}\n```\n"""
    PROMPT_COMPLETE_FS4 = """// You are a professional Java test case writer, please create a test case named `testIssue504` for the issue `Closure-504`, utilizing the provided issue report information and the following function signature.\n\n// ## Issue-ID: Closure-504\n\n// ## Issue-Title: \n// void function () {{}}(); wrongly identified as having no side effects\n\n// ## Issue-Description: \n// This code results in the execution of the function and should not be identified as having no side effects.\n\npublic void testIssue504() {{\n    args.add("--compilation_level=ADVANCED_OPTIMIZATIONS");\n    test("void function() {{ alert('hi'); }}();",\n            "alert('hi');", CheckSideEffects.USELESS_CODE_ERROR);\n}}\n\n// You are a professional Java test case writer, please create a test case named `testAtanI` for the issue `Math-MATH-657`, utilizing the provided issue report information and the following function signature.\n\n// ## Issue-ID: Math-MATH-657\n\n// ## Issue-Title: \n// Division by zero\n\n// ## Issue-Description: \n\nIn class Complex, division by zero always returns NaN. I think that it should return NaN only when the numerator is also ZERO, otherwise the result should be INF. See [here](http://en.wikipedia.org/wiki/Riemann_sphere#Arithmetic_operations).\n\n@Test\npublic void testAtanI() {{\n    Assert.assertTrue(Complex.I.atan().isNaN());\n}}\n\n// You are a professional Java test case writer, please create a test case named `testIssue582` for the issue `Closure-582`, utilizing the provided issue report information and the following function signature.\n\n// ## Issue-ID: Closure-582\n\n// ## Issue-Title: \n// -0.0 becomes 0 even in whitespace mode\n\n// ## Issue-Description: \n// Affects dart: http://code.google.com/p/dart/issues/detail?id=146\npublic void testIssue582() {{\n    assertPrint("var x = -0.0;", "var x=-0.0");\n}}\n\n// You are a professional Java test case writer, please create a test case named `testTopLevelValueTypeWithSkipValue` for the issue `Gson-773`, utilizing the provided issue report information and the following function signature.\n\n// ## Issue-ID: Gson-773\n\n// ## Issue-Title: \n// Update reader and writer for RFC 7159.\n\n// ## Issue-Description: \n// This allows for top-level value types without the requirement of leniency.\n\npublic void testTopLevelValueTypeWithSkipValue() throws IOException {{\n    JsonReader reader = new JsonReader(reader("true"));\n    reader.skipValue();\n    assertEquals(JsonToken.END_DOCUMENT, reader.peek());\n}}\n\n// You are a professional Java test case writer, please create a test case named `{function_name}` for the issue `{issue_id}`, utilizing the provided issue report information and the following function signature.\n\n{issue_report}\n\n{nl_context}"""
    PROMPT_CHAT_FS4 = """```markdown\n{issue_report}\n```\n\nYou are a professional Java test case writer, please create a test case named `{function_name}` for the issue `{issue_id}`, utilizing the provided issue report information and the following function signature.\n\n```java\n{nl_context}\n```\n"""
    
    PROMPT_COMPLETE_FS = [
        PROMPT_COMPLETE_FS0, PROMPT_COMPLETE_FS1, PROMPT_COMPLETE_FS2, PROMPT_COMPLETE_FS3, PROMPT_COMPLETE_FS4
    ]
    PROMPT_CHAT_FS = [
        PROMPT_CHAT_FS0, PROMPT_CHAT_FS1, PROMPT_CHAT_FS2, PROMPT_CHAT_FS3, PROMPT_CHAT_FS4
    ]
    PROMPT_COMPLETE = """// You are a professional Java test case writer, please create a test case named `{function_name}` for the issue `{issue_id}`, utilizing the provided issue report information and the following function signature.\n\n{issue_report}\n\n{nl_context}"""
    PROMPT_CHAT = """```markdown\n{issue_report}\n```\n\nYou are a professional Java test case writer, please create a test case named `{function_name}` for the issue `{issue_id}`, utilizing the provided issue report information and the following function signature.\n\n```java\n{nl_context}\n```\n"""
    
    function_data = json.load(open(os.path.join(FILE_DIR, 'data', 'task_testgenissue.json')))
    print("Total number of tasks: ", len(function_data))
    
    prompted_data = []
    
    for example in tqdm(function_data):
        issue_report = tokenizer.convert_tokens_to_string(tokenizer.tokenize(example["issue_report"])[:1024])
        issue_report = "\n".join(issue_report.split("\n"))
        
        import_context = example["import_context"] if context["import_context"] else ""
        class_signature = example["class_signature"] if context["class_signature"] else ""
        class_field_context = example["class_field_context"] if context["class_field_context"] else ""
        class_function_signature_context = example["class_function_signature_context"] if context["class_function_signature_context"] else ""
        
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
            
        # if len(example["better_comment"].strip()) <= 1:
        #     continue
        # comment = example["better_comment"]
        # comment = pretty_comment(comment, example["indent"])
        if context["few_shot"] == -1:
            prompt_chat = PROMPT_CHAT.format(
                issue_report=issue_report,
                function_name=example["function_name"],
                issue_id=example["issue_id"],
                nl_context=f'{example["indent"]}{example["function_signature"].strip()}'+" {\n"
            )
            issue_report = "// "+ "\n// ".join(issue_report.split("\n"))
            code_context = "// "+ "\n// ".join(code_context.split("\n"))
            prompt_complete = PROMPT_COMPLETE.format(
                issue_report=issue_report,
                function_name=example["function_name"],
                issue_id=example["issue_id"],
                nl_context=f'{example["indent"]}{example["function_signature"].strip()}'+" {\n"
            )
            
            prompt_complete_without_signature = PROMPT_COMPLETE.format(
                                                    issue_report=issue_report,
                                                    function_name=example["function_name"],
                                                    issue_id=example["issue_id"],
                                                    nl_context=f''
                                                )
        elif context["few_shot"] in [0,1,2,3,4]:
            prompt_chat = PROMPT_CHAT_FS[context["few_shot"]].format(
                issue_report=issue_report,
                function_name=example["function_name"],
                issue_id=example["issue_id"],
                nl_context=f'{example["indent"]}{example["function_signature"].strip()}'+" {\n"
            )
            issue_report = "// "+ "\n// ".join(issue_report.split("\n"))
            prompt_complete = PROMPT_COMPLETE_FS[context["few_shot"]].format(
                issue_report=issue_report,
                function_name=example["function_name"],
                issue_id=example["issue_id"],
                nl_context=f'{example["indent"]}{example["function_signature"].strip()}'+" {\n"
            )
            
            prompt_complete_without_signature = PROMPT_COMPLETE_FS[context["few_shot"]].format(
                                                    issue_report=issue_report,
                                                    function_name=example["function_name"],
                                                    issue_id=example["issue_id"],
                                                    nl_context=f''
                                                )
            
        
        prompted_example = {
            "task_id": hash_string(example["task_id"]),
            "project":example["project_id"], "bug_id":example["bug_id"], 
            "testmethod": example["testmethod"], "source_dir":example["test_classes"], 
            "classes_modified": example["classes_modified"],
            "location":example["location"],
            "location_fixed":example["location"],
            "location_buggy":example["location"],
            "start_buggy": example["function_start_line_buggy"], "end_buggy": example["function_end_line_buggy"], 
            "start_fixed": example["function_start_line_fixed"], "end_fixed": example["function_end_line_fixed"],
            "function":example["function"], "comment":example["function_comment"], 
            "function_name": example["function_name"],
            "prompt_chat":prompt_chat, 
            "prompt_complete":prompt_complete, 
            "prompt_complete_without_signature":prompt_complete_without_signature,
            "function_signature":example["function_signature"], 
            "source_buggy":example["test_source_buggy"], 
            "source_fixed":example["test_source_fixed"], 
            "indent": example["indent"],
        }
        
        prompted_data.append(prompted_example)
    prompted_data.sort(key=lambda x: x["task_id"])
    dataset = {"code_ujb_testgenissue": prompted_data}
    json.dump(dataset, open(os.path.join(FILE_DIR, 'data', f'task_testgenissue_bench_{save_suffix}.json'), 'w'), indent=4, ensure_ascii=False)

def process_get_correct_result(FILE_DIR, save_suffix=""):
    data = json.load(open(os.path.join(FILE_DIR, 'data', f'task_testgenissue_bench_{save_suffix}.json')))
    correct_result = []
    for idx, example in enumerate(data["code_ujb_testgenissue"]):
        result = {
            "task_id": idx,
            "outputs": [example["prompt_complete_without_signature"] + example["function"]]*10
        }
        correct_result.append(result)
        
    json.dump(correct_result, open(os.path.join(FILE_DIR, 'data', f'task_testgenissue_correct_result_{save_suffix}.json'), 'w'), indent=4, ensure_ascii=False)

def generate_dataset(FILE_DIR):
    save_suffix_list = ["1111|3072", "0100|3072", "1100|3072", "0110|3072",
                        "0101|3072", "1111|3072", "1111|2048",
                        "1111|1024", "1111|512",]
    for save_suffix in save_suffix_list:
        context_str, context_length = save_suffix.split("|")
        context = {
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
            "default": 0,
            "import_context": int(context_str[0]),
            "class_signature": int(context_str[1]),
            "class_field_context": int(context_str[2]),
            "class_function_signature_context": int(context_str[3]),
        }
        process_get_prompt(FILE_DIR, context, context_length=int(context_length), save_suffix=save_suffix,
                           improve_list=["class_function_signature_context", "class_field_context", "import_context", "class_signature"])
        process_get_correct_result(FILE_DIR, save_suffix=save_suffix)

def generate_default_dataset(FILE_DIR):
    save_suffix_list = ["default|2048",]
    for save_suffix in save_suffix_list:
        context_str, context_length = save_suffix.split("|")
        context = {
            "few_shot": -1,
            "import_context": 0,
            "class_signature": 0,
            "class_field_context": 0,
            "class_function_signature_context": 0,
        }
        process_get_prompt(FILE_DIR, context, context_length=int(context_length), save_suffix=save_suffix)
        process_get_correct_result(FILE_DIR, save_suffix=save_suffix)

def generate_fs_dataset(FILE_DIR):
    save_suffix_list = ["fs0|2048","fs1|2048","fs2|2048","fs3|2048","fs4|2048",]
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
    process_issues_info(FILE_DIR)
    process_issues_task(FILE_DIR)
    
    generate_default_dataset(FILE_DIR)
    generate_fs_dataset(FILE_DIR)
    
    print("Done")
    

if __name__ == '__main__':
    main()    