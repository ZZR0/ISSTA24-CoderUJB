import ast
import copy
import json
import os
import random
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

from code_parser import Code_AST
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import (check_is_complete_function,
                   get_prompt_with_comment, 
                   read_file, hash_string, get_indent,
                   fast_multiprocessing)

random.seed(42)
tokenizer = AutoTokenizer.from_pretrained("gpt2")


def get_project_ids():
    cmd = ['defects4j', 'pids']
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    project_ids = result.stdout.splitlines()
    return project_ids

def get_location_function(functions, line):
    modified_functions = []
    for function in functions:
        if line >= function.start_line and line <= function.end_line:
            modified_functions.append(function)
    
    if len(modified_functions) == 0:
        return None
    
    modified_function = modified_functions[0]
    for function in modified_functions:
        if modified_function.source in function.source:
            modified_function = function
    
    return modified_function

def process_worker(FILE_DIR, project_id, bug_id, src_classes, src_tests, mappings):
    if len(mappings) < 2: return None
    
    mapping = mappings[0]
    class_file_path = os.path.join(FILE_DIR, "projects", project_id, str(bug_id)+"f", 
                                    src_classes, mapping["be_test_class_file"])
    class_file = read_file(class_file_path)
    try:
        class_ast = Code_AST(code=class_file, lang="java").ast
        class_context = class_ast.get_class_context_source()
        import_context = class_ast.get_import_context_source()
        class_signature = class_ast.get_class_signature_context_source()
        class_field_context = class_ast.get_class_field_context_source()
        class_function_signature_context = class_ast.get_class_functions_signature_context_source()
    except:
        return None
    functions = [function for function in class_ast.get_functions() if function.get_function_name() == mapping["be_test_function_name"]]
    if len(functions) == 0: return None
    function = get_location_function(functions, line=int(mapping["line_numbers"][-1]))
    if function is None: return None
    if not check_is_complete_function(function): return None
    if len(function.source) < 256: return None
    
    comment = function.get_function_comment()
    if len(comment) < 128 or (not "/*" in comment) or (not "*/" in comment): return None
    
    location = os.path.join(src_classes, mapping["be_test_class_file"])
    function_name = mapping["be_test_function_name"]
    testmethods = [f"{mapping['test_file']}::{mapping['test_function']}::{mapping['method_line_rate']}" for mapping in mappings if "test" in mapping['test_function'].lower()]
    tested_lines = []
    for mapping in mappings:
        if not "test" in mapping['test_function'].lower(): continue
        tested_lines.extend(mapping["line_numbers"])
    tested_lines = sorted(list(set(tested_lines)))
    if len(testmethods) < 2: return None
    
    function_prefix_with_signature, _, function_signature = get_code_prefix(class_ast, function.source_line)
    
    indent = get_indent(function.source_line)
    function_example = {
        "task_id": f"complete|{project_id}|{bug_id}|{location}|{function_name}|{function.start_line}|{function.end_line}",
        "project_id": project_id,
        "bug_id": bug_id,
        "testmethods": testmethods,
        "source_dir": src_classes,
        "location": location,
        "function_star_line": function.start_line,
        "function_end_line": function.end_line,
        "function": function.source_line,
        "function_name": function_name,
        "function_comment": comment,
        "function_prefix_with_signature": function_prefix_with_signature,
        "function_signature": function_signature, 
        "source": class_file,
        "class_context": class_context,
        "import_context": import_context,
        "class_signature": class_signature,
        "class_field_context": class_field_context,
        "class_function_signature_context": class_function_signature_context,
        "indent": indent,
        "function_tested_rate": len(tested_lines)/(function.end_line-function.start_line),
    }
    return function_example

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
        
        # print(len(merged_func_test_map["test_relevant_methods"]))
        function_to_test = {}
        for mapping in tqdm(merged_func_test_map["test_relevant_methods"]):
            if "<" in mapping["be_test_function_name"]: continue
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
        
        processed_data.append(merged_func_test_map)
        # break
    return processed_data

def process_extract_function(FILE_DIR, mapping_data):
    processed_data = []
    tasks = []
    for idx, item in enumerate(mapping_data):
        for function_key in item["function_to_test"]:
            tasks.append((FILE_DIR, item["project_id"], item["bug_id"], item["src_classes"], item["src_tests"], item["function_to_test"][function_key]))
        # break
    random.shuffle(tasks)
    # tasks = tasks[:500000]
    # print(tasks)
    
    # for task in tqdm(tasks):
    #     processed_data.append(process_worker(*task))
    
    # processed_data = robust_multiprocessing(process_worker, tasks)
    processed_data = fast_multiprocessing(process_worker, tasks)
    
    processed_data = [example for example in processed_data if example is not None]
    json.dump(processed_data, open(os.path.join(FILE_DIR, 'data', 'task_complete.json'), 'w'), indent=4, ensure_ascii=False)

def get_code_prefix(code_ast, function_code):
    # return "", "", ""
    functions = code_ast.get_functions()
    function = None
    for func in functions:
        if func.source_line == function_code:
            function = func
            break
    
    if function is None: raise Exception("Cannot find function")
    
    functino_body = function.get_function_body()
    function_prefix_with_signature = str(bytes(code_ast.source, 'utf8')[:functino_body.start_byte], encoding="utf-8")
    function_prefix_with_comment = "\n".join(code_ast.source.splitlines()[:function.start_line])
    function_signature = str(bytes(code_ast.source, 'utf8')[function.start_byte:functino_body.start_byte], encoding="utf-8")
    return function_prefix_with_signature, function_prefix_with_comment, function_signature

def process_filter_prompt(FILE_DIR):
    data = json.load(open(os.path.join(FILE_DIR, 'data', 'task_complete.json')))
    function_name_set = set()
    filtered_data_dict = {}
    for example in data:
        if example["function_name"] in function_name_set:
            continue
        function_name_set.add(example["function_name"])
        
        if example["function_star_line"]-example["function_end_line"] > 30: continue
        
        test_class = [testmethod.split("::")[0].split(".")[-1] for testmethod in example["testmethods"]]
        test_class = [tc.replace("Tests", "").replace("Test", "").replace("tests", "").replace("test", "") for tc in test_class]
        file_class = example["location"].replace(".java","").split("/")[-1]
        if not file_class in test_class: continue
        
        if example["project_id"] not in filtered_data_dict:
            filtered_data_dict[example["project_id"]] = []
        filtered_data_dict[example["project_id"]].append(example)
    
    filtered_data = []
    for project in filtered_data_dict:
        filtered_data.extend(filtered_data_dict[project])
    
    # filtered_data = filtered_data[:10]
    print("Total number of tasks after filtering: ", len(filtered_data))
    json.dump(filtered_data, open(os.path.join(FILE_DIR, 'data', 'task_complete_filtered.json'), 'w'), indent=4, ensure_ascii=False)


def process_final_bench(FILE_DIR):
    data = json.load(open(os.path.join(FILE_DIR, 'data', 'task_complete_filtered.json')))
    processed_data = []
    for example in data:
        test_class = [testmethod.split("::")[0].split(".")[-1] for testmethod in example["testmethods"]]
        test_class = [tc.replace("Tests", "").replace("Test", "").replace("tests", "").replace("test", "") for tc in test_class]
        file_class = example["location"].replace(".java","").split("/")[-1]
        if file_class in test_class:
            if example["function_tested_rate"] >= 0.6:
                processed_data.append(example)
            elif example["function_tested_rate"] >= 0.8:
                processed_data.append(example)
    
    print("Total Task Case after filtering:", len(processed_data))
    json.dump(processed_data, open(os.path.join(FILE_DIR, 'data', 'task_complete_filtered_final.json'), 'w'), indent=4, ensure_ascii=False)

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
    # print("import_context", import_context)
    # print(class_signature)
    # print(class_field_context)
    # print(class_function_signature_context)
    
    # exit()
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
    PROMPT_COMPLETE = """{code_context}\n\n// You are a professional Java programmer, please create a function named `{function_name}` based on the provided abstract Java class context information and the following natural language annotations.\n\n{nl_context}"""
    PROMPT_CHAT = """```java\n{code_context}\n```\n\nYou are a professional Java programmer, please create a function named `{function_name}` based on the provided abstract Java class context information and the following natural language annotations.\n\n```java\n{nl_context}\n```\n"""
    PROMPT_COMPLETE_FS0 = """// You are a professional Java programmer, please create a function named `{function_name}` based on the provided abstract Java class context information and the following natural language annotations.\n\n{nl_context}"""
    PROMPT_CHAT_FS0 = """You are a professional Java programmer, please create a function named `{function_name}` based on the provided abstract Java class context information and the following natural language annotations.\n\n```java\n{nl_context}\n```\n"""
    PROMPT_COMPLETE_FS1 = """// You are a professional Java programmer, please create a function named `getCache` based on the provided abstract Java class context information and the following natural language annotations.\n\n/**\n * Get a cache of Strategies for a particular field\n * @param field The Calendar field\n * @return a cache of Locale to Strategy\n */\nprivate static ConcurrentMap<Locale, Strategy> getCache(final int field) {{\n    synchronized(caches) {{\n        if(caches[field]==null) {{\n            caches[field]= new ConcurrentHashMap<Locale,Strategy>(3);\n        }}\n        return caches[field];\n    }}\n}}\n\n// You are a professional Java programmer, please create a function named `{function_name}` based on the provided abstract Java class context information and the following natural language annotations.\n\n{nl_context}"""
    PROMPT_CHAT_FS1 = """You are a professional Java programmer, please create a function named `{function_name}` based on the provided abstract Java class context information and the following natural language annotations.\n\n```java\n{nl_context}\n```\n"""
    PROMPT_COMPLETE_FS2 = """// You are a professional Java programmer, please create a function named `getCache` based on the provided abstract Java class context information and the following natural language annotations.\n\n/**\n * Get a cache of Strategies for a particular field\n * @param field The Calendar field\n * @return a cache of Locale to Strategy\n */\nprivate static ConcurrentMap<Locale, Strategy> getCache(final int field) {{\n    synchronized(caches) {{\n        if(caches[field]==null) {{\n            caches[field]= new ConcurrentHashMap<Locale,Strategy>(3);\n        }}\n        return caches[field];\n    }}\n}}\n\n// You are a professional Java programmer, please create a function named `title` based on the provided abstract Java class context information and the following natural language annotations.\n\n/**\n Get the string contents of the document's {{@code title}} element.\n @return Trimmed title, or empty string if none set.\n */\npublic String title() {{\n    // title is a preserve whitespace tag (for document output), but normalised here\n    Element titleEl = getElementsByTag("title").first();\n    return titleEl != null ? StringUtil.normaliseWhitespace(titleEl.text()).trim() : "";\n}}\n\n// You are a professional Java programmer, please create a function named `{function_name}` based on the provided abstract Java class context information and the following natural language annotations.\n\n{nl_context}"""
    PROMPT_CHAT_FS2 = """You are a professional Java programmer, please create a function named `{function_name}` based on the provided abstract Java class context information and the following natural language annotations.\n\n```java\n{nl_context}\n```\n"""
    PROMPT_COMPLETE_FS3 = """// You are a professional Java programmer, please create a function named `getCache` based on the provided abstract Java class context information and the following natural language annotations.\n\n/**\n * Get a cache of Strategies for a particular field\n * @param field The Calendar field\n * @return a cache of Locale to Strategy\n */\nprivate static ConcurrentMap<Locale, Strategy> getCache(final int field) {{\n    synchronized(caches) {{\n        if(caches[field]==null) {{\n            caches[field]= new ConcurrentHashMap<Locale,Strategy>(3);\n        }}\n        return caches[field];\n    }}\n}}\n\n// You are a professional Java programmer, please create a function named `title` based on the provided abstract Java class context information and the following natural language annotations.\n\n/**\n Get the string contents of the document's {{@code title}} element.\n @return Trimmed title, or empty string if none set.\n */\npublic String title() {{\n    // title is a preserve whitespace tag (for document output), but normalised here\n    Element titleEl = getElementsByTag("title").first();\n    return titleEl != null ? StringUtil.normaliseWhitespace(titleEl.text()).trim() : "";\n}}\n\n// You are a professional Java programmer, please create a function named `getOwnPropertyNames` based on the provided abstract Java class context information and the following natural language annotations.\n\n/**\n * Includes the prototype iff someone has created it. We do not want\n * to expose the prototype for ordinary functions.\n */\n@Override\npublic Set<String> getOwnPropertyNames() {{\n  if (prototypeSlot == null) {{\n    return super.getOwnPropertyNames();\n  }} else {{\n    Set<String> names = Sets.newHashSet("prototype");\n    names.addAll(super.getOwnPropertyNames());\n    return names;\n  }}\n}}\n\n// You are a professional Java programmer, please create a function named `{function_name}` based on the provided abstract Java class context information and the following natural language annotations.\n\n{nl_context}"""
    PROMPT_CHAT_FS3 = """You are a professional Java programmer, please create a function named `{function_name}` based on the provided abstract Java class context information and the following natural language annotations.\n\n```java\n{nl_context}\n```\n"""
    PROMPT_COMPLETE_FS4 = """// You are a professional Java programmer, please create a function named `getCache` based on the provided abstract Java class context information and the following natural language annotations.\n\n/**\n * Get a cache of Strategies for a particular field\n * @param field The Calendar field\n * @return a cache of Locale to Strategy\n */\nprivate static ConcurrentMap<Locale, Strategy> getCache(final int field) {{\n    synchronized(caches) {{\n        if(caches[field]==null) {{\n            caches[field]= new ConcurrentHashMap<Locale,Strategy>(3);\n        }}\n        return caches[field];\n    }}\n}}\n\n// You are a professional Java programmer, please create a function named `title` based on the provided abstract Java class context information and the following natural language annotations.\n\n/**\n Get the string contents of the document's {{@code title}} element.\n @return Trimmed title, or empty string if none set.\n */\npublic String title() {{\n    // title is a preserve whitespace tag (for document output), but normalised here\n    Element titleEl = getElementsByTag("title").first();\n    return titleEl != null ? StringUtil.normaliseWhitespace(titleEl.text()).trim() : "";\n}}\n\n// You are a professional Java programmer, please create a function named `getOwnPropertyNames` based on the provided abstract Java class context information and the following natural language annotations.\n\n/**\n * Includes the prototype iff someone has created it. We do not want\n * to expose the prototype for ordinary functions.\n */\n@Override\npublic Set<String> getOwnPropertyNames() {{\n  if (prototypeSlot == null) {{\n    return super.getOwnPropertyNames();\n  }} else {{\n    Set<String> names = Sets.newHashSet("prototype");\n    names.addAll(super.getOwnPropertyNames());\n    return names;\n  }}\n}}\n\n// You are a professional Java programmer, please create a function named `eye` based on the provided abstract Java class context information and the following natural language annotations.\n\n/**\n * @param n Number of rows.\n * @param m Number of columns.\n * @return n-by-m matrix of 0 values out of diagonal, and 1 values on\n * the diagonal.\n */\nprivate static RealMatrix eye(int n, int m) {{\n    final double[][] d = new double[n][m];\n    for (int r = 0; r < n; r++) {{\n        if (r < m) {{\n            d[r][r] = 1;\n        }}\n    }}\n    return new Array2DRowRealMatrix(d, false);\n}}\n\n// You are a professional Java programmer, please create a function named `{function_name}` based on the provided abstract Java class context information and the following natural language annotations.\n\n{nl_context}"""
    PROMPT_CHAT_FS4 = """You are a professional Java programmer, please create a function named `{function_name}` based on the provided abstract Java class context information and the following natural language annotations.\n\n```java\n{nl_context}\n```\n"""
    
    PROMPT_COMPLETE_FS = [
        PROMPT_COMPLETE_FS0, PROMPT_COMPLETE_FS1, PROMPT_COMPLETE_FS2, PROMPT_COMPLETE_FS3, PROMPT_COMPLETE_FS4
    ]
    PROMPT_CHAT_FS = [
        PROMPT_CHAT_FS0, PROMPT_CHAT_FS1, PROMPT_CHAT_FS2, PROMPT_CHAT_FS3, PROMPT_CHAT_FS4
    ]
    function_data = json.load(open(os.path.join(FILE_DIR, 'data', 'task_complete_filtered_final.json')))
    print("Total number of tasks: ", len(function_data))
    
    prompted_data = []
    
    for example in tqdm(function_data):
        file_class = example["location"].replace(".java","").split("/")[-1]
        testmethods = [testmethod for testmethod in example["testmethods"] if file_class in testmethod.split("::")[0].split(".")[-1]]
        testmethods = [testmethod for testmethod in testmethods if file_class == testmethod\
            .split("::")[0].split(".")[-1].replace("Tests", "").replace("Test", "").replace("tests", "").replace("test", "")]
        testmethods.sort(key=lambda x: float(x.split("::")[-1]))
        testmethods = ["::".join(testmethod.split("::")[:2]) for testmethod in testmethods]
        testmethods = testmethods[:3]
        if len(testmethods) == 0: continue
        
        import_context = example["import_context"] if context["import_context"] else ""
        class_signature = example["class_signature"] if context["class_signature"] else ""
        class_field_context = example["class_field_context"] if context["class_field_context"] else ""
        class_function_signature_context = example["class_function_signature_context"] if context["class_function_signature_context"] else ""
        
        code_context = ""
        if context["few_shot"] < 0:
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
            comment = example["function_comment"]
            prompt_chat = PROMPT_CHAT.format(
                    code_context=code_context,
                    function_name=example["function_name"],
                    nl_context=f'{comment}\n{example["indent"]}{example["function_signature"]}'+"{\n"
                )
            code_context = "// "+ "\n// ".join(code_context.split("\n"))
            prompt_complete = PROMPT_COMPLETE.format(
                    code_context=code_context,
                    function_name=example["function_name"],
                    nl_context=f'{comment}\n{example["indent"]}{example["function_signature"]}'+"{\n"
                )
        elif context["few_shot"] in [0,1,2,3,4]:
            comment = example["function_comment"]
            prompt_chat = PROMPT_CHAT_FS[context["few_shot"]].format(
                function_name=example["function_name"],
                nl_context=f'{comment}\n{example["indent"]}{example["function_signature"]}'+"{\n"
            )
            prompt_complete = PROMPT_COMPLETE_FS[context["few_shot"]].format(
                function_name=example["function_name"],
                nl_context=f'{comment}\n{example["indent"]}{example["function_signature"]}'+"{\n"
            )
        else:
            raise NotImplementedError()
        
            
        prompt_chat_with_comment = get_prompt_with_comment(prompt_chat)
        prompt_complete_with_comment = get_prompt_with_comment(prompt_complete)
        
        prompted_example = {
                "task_id": hash_string(example["task_id"]),
                "project":example["project_id"], "bug_id":example["bug_id"], 
                "testmethods": testmethods, "source_dir":example["source_dir"], 
                "location":example["location"],
                "start": example["function_star_line"], "end": example["function_end_line"], 
                "function":example["function"], "comment":example["function_comment"], 
                "function_name": example["function_name"],
                "prompt_chat":prompt_chat, 
                "prompt_chat_with_comment":prompt_chat_with_comment, 
                "prompt_complete":prompt_complete, 
                "prompt_complete_with_comment":prompt_complete_with_comment,
                "function_signature":example["function_signature"], 
                "import_context": example["import_context"],
                "class_signature": example["class_signature"],
                "class_field_context": example["class_field_context"],
                "class_function_signature_context": example["class_function_signature_context"],
                "code_context": code_context,
                "source": example["source"], "indent": example["indent"],
                "function_tested_rate": example["function_tested_rate"],
            }

        prompted_data.append(prompted_example)
    prompted_data.sort(key=lambda x: x["task_id"])
    dataset = {"code_ujb_complete": prompted_data}
    json.dump(dataset, open(os.path.join(FILE_DIR, 'data', f'task_complete_bench_{save_suffix}.json'), 'w'), indent=4, ensure_ascii=False)
 
def process_get_correct_result(FILE_DIR, save_suffix=""):
    data = json.load(open(os.path.join(FILE_DIR, 'data', f'task_complete_bench_{save_suffix}.json')))
    correct_result = []
    for idx, example in enumerate(data["code_ujb_complete"]):
        result = {
            "task_id": idx,
            "outputs": [example["prompt_complete_with_comment"] + "\n" + example["function"]]*10
        }
        correct_result.append(result)
        
    json.dump(correct_result, open(os.path.join(FILE_DIR, 'data', 'task_complete_correct_result.json'), 'w'), indent=4, ensure_ascii=False)


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
    mapping_data = process_extract_mapping(FILE_DIR)
    process_extract_function(FILE_DIR, mapping_data)
    
    process_filter_prompt(FILE_DIR)
    process_final_bench(FILE_DIR)
    
    generate_default_dataset(FILE_DIR)
    generate_fs_dataset(FILE_DIR)
    
if __name__ == '__main__':
    main()    