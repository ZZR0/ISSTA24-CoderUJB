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

from utils import (get_prompt_with_comment, 
                   read_file, hash_string,
                   get_indent, fast_multiprocessing)

random.seed(42)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def check_is_test_function(node):
    if node.path.endswith("class_body|method_declaration"):
        return True
    if node.path.endswith("enum_body_declarations|method_declaration"):
        return True
    if node.path.endswith("class_body|constructor_declaration"):
        return True
    return False

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

def get_function(class_file_path, function_name, lines=None, debug=False):
    
    class_file = read_file(class_file_path)
    try:
        class_ast = Code_AST(code=class_file, lang="java").ast
    except:
        return None, None, None
    functions = [function for function in class_ast.get_functions() if function.get_function_name() == function_name]
    
    if lines is not None:
        function = get_location_function(functions, line=int(lines[-1]))
    else:
        function = functions[0]
    
    comment = function.get_function_comment() if function is not None else None
    
    if debug:
        print("class_file_path", class_file_path)
        print("function_name", function_name)
        print("lines", lines)
        print("functions", functions)
        print("functions lines", [(function.start_line, function.end_line) for function in functions])
        print("function", function)
    
    return function, comment, class_file, class_ast

def process_worker(FILE_DIR, project_id, bug_id, src_classes, src_tests, mappings):
    if len(mappings) < 1: return None
    
    mapping = mappings[0]
    class_file_path = os.path.join(FILE_DIR, "projects", project_id, str(bug_id)+"f", 
                                    src_tests, mapping['test_file'].replace(".", "/")+".java")
    
    function, comment, class_file, class_ast = get_function(class_file_path, mapping["test_function"], lines=None)
    
    if function is None: return None
    if function.get_function_body() is None: return None
    if not check_is_test_function(function): return None
    if len(function.source) < 256: return None
    if len(comment) < 128 or (not "/*" in comment) or (not "*/" in comment): return None
    
    function_prefix_with_signature, _, function_signature = get_code_prefix(class_ast, function.source_line)
    
    function_name = mapping["test_function"]
    location = os.path.join(src_tests, mapping['test_file'].replace(".", "/")+".java")
    testmethods = [f"{mapping['test_file']}::{mapping['test_function']}" for mapping in mappings if "test" in mapping['test_function'].lower()]
    testmethods = list(set(testmethods))
    if len(testmethods) == 0: return None
    
    classmethods = []
    for mapping in mappings:
        if "<" in mapping['be_test_function_name']: continue
        class_name = mapping['test_file'].split(".")[-1].replace("Tests", "").replace("Test", "").replace("tests", "").replace("test", "")
        if class_name != mapping["be_test_class_name"].split(".")[-1]: continue
        classmethods.append(
            {
                "be_test_class_file": mapping["be_test_class_file"],
                "be_test_class_name": mapping["be_test_class_name"],
                "be_test_function_name": mapping["be_test_function_name"],
                "be_test_function_signature": mapping["be_test_function_signature"],
                "line_numbers": mapping["line_numbers"],
                "method_line_rate": mapping["method_line_rate"]
            }
        )
    if len(set([method["be_test_class_name"] for method in classmethods])) != 1: return None
    # if len(classmethods) == 0: return None
    
    be_test_class_path = os.path.join(FILE_DIR, "projects", project_id, str(bug_id)+"f", 
                                    src_classes, classmethods[0]["be_test_class_file"])
    be_test_class_file = read_file(be_test_class_path)
    be_test_class_ast = Code_AST(code=be_test_class_file, lang="java").ast
    be_test_class_context = be_test_class_ast.get_class_context_source()
    be_test_import_context = be_test_class_ast.get_import_context_source()
    be_test_class_signature = be_test_class_ast.get_class_signature_context_source()
    be_test_class_field_context = be_test_class_ast.get_class_field_context_source()
    be_test_class_function_signature_context = be_test_class_ast.get_class_functions_signature_context_source()
    
    test_class_context = class_ast.get_class_context_source()
    test_import_context = class_ast.get_import_context_source()
    test_class_signature = class_ast.get_class_signature_context_source()
    test_class_field_context = class_ast.get_class_field_context_source()
    test_class_function_signature_context = class_ast.get_class_functions_signature_context_source()
    indent = get_indent(function.source_line)
    
    # be_test_function, be_test_comment, be_test_class_file, be_test_class_ast = get_function(be_test_class_path, 
    #                                                                     classmethods[0]["be_test_function_name"], 
    #                                                                     classmethods[0]["line_numbers"],
    #                                                                     debug=False)
    # if be_test_function is None: return None
    
    function_example = {
        "task_id": f"testgen|{project_id}|{bug_id}|{location}|{function_name}|{function.start_line}|{function.end_line}",
        "project_id": project_id,
        "bug_id": bug_id,
        "testmethods": testmethods,
        "source_dir": src_tests,
        "location": location,
        "function_star_line": function.start_line,
        "function_end_line": function.end_line,
        "function": function.source_line,
        "function_name": function_name,
        "function_comment": comment,
        "function_prefix_with_signature": function_prefix_with_signature,
        "function_signature": function_signature, 
        "source": class_file,
        "classmethods": classmethods,
        "be_test_class_context": be_test_class_context,
        "be_test_import_context": be_test_import_context,
        "be_test_class_signature": be_test_class_signature,
        "be_test_class_field_context": be_test_class_field_context,
        "be_test_class_function_signature_context": be_test_class_function_signature_context,
        "be_test_class_name": classmethods[0]["be_test_class_name"].split(".")[-1],
        "be_test_class_long_name": classmethods[0]["be_test_class_name"],
        "test_class_context": test_class_context,
        "test_import_context": test_import_context,
        "test_class_signature": test_class_signature,
        "test_class_field_context": test_class_field_context,
        "test_class_function_signature_context": test_class_function_signature_context,
        "indent": indent
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
        
        # function_to_test = {}
        # for mapping in merged_func_test_map["test_relevant_methods"]:
        #     if "<" in mapping["be_test_function_name"]: continue
        #     key = "::".join([
        #         mapping["be_test_class_name"],
        #         mapping["be_test_function_name"],
        #         mapping["be_test_function_signature"]
        #     ])
        #     if key not in function_to_test:
        #         function_to_test[key] = []
        #     function_to_test[key].append(mapping)
        #     # function_to_test[key].append(f"{mapping['test_file']}::{mapping['test_function']}")
        # merged_func_test_map["function_to_test"] = function_to_test
        
        test_to_function = {}
        for mapping in merged_func_test_map["test_relevant_methods"]:
            if "<" in mapping["be_test_function_name"]: continue
            key = "::".join([
                mapping["test_file"],
                mapping["test_function"],
            ])
            if key not in test_to_function:
                test_to_function[key] = []
            test_to_function[key].append(mapping)
        merged_func_test_map["test_to_function"] = test_to_function
        
        processed_data.append(merged_func_test_map)
    return processed_data

def process_extract_function(FILE_DIR, mapping_data):
    processed_data = []
    tasks = []
    for item in mapping_data:
        for function_key in item["test_to_function"]:
            tasks.append((FILE_DIR, item["project_id"], item["bug_id"], item["src_classes"], item["src_tests"], item["test_to_function"][function_key]))
    random.shuffle(tasks)
    # tasks = tasks[:3000]
    # print(tasks)
    print("Start processing...")
    # for task in tqdm(tasks):
    #     processed_data.append(process_worker(*task))
        
    processed_data = fast_multiprocessing(process_worker, tasks)
    
    processed_data = [example for example in processed_data if example is not None]
    json.dump(processed_data, open(os.path.join(FILE_DIR, 'data', 'task_testgen.json'), 'w'), indent=4, ensure_ascii=False)

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
    data = json.load(open(os.path.join(FILE_DIR, 'data', 'task_testgen.json')))
    function_name_set = set()
    filtered_data_dict = {}
    for example in data:
        if example["function_name"] in function_name_set:
            continue
        function_name_set.add(example["function_name"])
        
        if example["function_star_line"]-example["function_end_line"] > 30: continue
        
        if example["project_id"] not in filtered_data_dict:
            filtered_data_dict[example["project_id"]] = []
        filtered_data_dict[example["project_id"]].append(example)
    
    filtered_data = []
    for project in filtered_data_dict:
        filtered_data.extend(filtered_data_dict[project])
    
    # filtered_data = filtered_data[:10]
    print("Total number of tasks after filtering: ", len(filtered_data))
    json.dump(filtered_data, open(os.path.join(FILE_DIR, 'data', 'task_testgen_filtered.json'), 'w'), indent=4, ensure_ascii=False)


def process_final_bench(FILE_DIR):
    data = json.load(open(os.path.join(FILE_DIR, 'data', 'task_testgen_filtered.json')))
    processed_data = []
    for prompted_example in data:
        processed_data.append(prompted_example)
    print("Total Task Case after filtering:", len(processed_data))
    json.dump(processed_data, open(os.path.join(FILE_DIR, 'data', 'task_testgen_filtered_final.json'), 'w'), indent=4, ensure_ascii=False)

def get_code_context(be_test_import_context, be_test_class_signature, 
                     be_test_class_field_context, be_test_class_function_signature_context,
                     test_import_context, test_class_signature, 
                     test_class_field_context, test_class_function_signature_context,
                     context_length):
    code_context = "// Abstract Java Tested Class\n{be_test_import_context}\n\n{be_test_class_signature} {{\n{be_test_class_field_context}\n\n{be_test_class_function_signature_context}\n}}\n\n// Abstract Java Test Class\n{test_import_context}\n\n{test_class_signature} {{\n{test_class_field_context}\n\n{test_class_function_signature_context}\n}}"
    code_context = code_context.format(
        be_test_import_context=be_test_import_context,
        be_test_class_signature=be_test_class_signature,
        be_test_class_field_context=be_test_class_field_context,
        be_test_class_function_signature_context=be_test_class_function_signature_context,
        test_import_context=test_import_context,
        test_class_signature=test_class_signature,
        test_class_field_context=test_class_field_context,
        test_class_function_signature_context=test_class_function_signature_context
    )
    code_context = code_context.strip()
    cut_code_context = tokenizer.convert_tokens_to_string(tokenizer.tokenize(code_context)[-context_length:])
    cut = cut_code_context != code_context
    return cut, cut_code_context

def get_code_context_order(
                     be_test_import_context, be_test_class_signature, 
                     be_test_class_field_context, be_test_class_function_signature_context,
                     test_import_context, test_class_signature, 
                     test_class_field_context, test_class_function_signature_context,
                     context_length,
                     improve_list=["class_function_signature_context", "class_field_context", "import_context", "class_signature"]):
    context_improve_list = {}
    for improve_key in improve_list:
        if improve_key == "be_test_import_context":
            context_improve_list[improve_key] = be_test_import_context
        elif improve_key == "be_test_class_signature":
            context_improve_list[improve_key] = be_test_class_signature
        elif improve_key == "be_test_class_field_context":
            context_improve_list[improve_key] = be_test_class_field_context
        elif improve_key == "be_test_class_function_signature_context":
            context_improve_list[improve_key] = be_test_class_function_signature_context
        elif improve_key == "test_import_context":
            context_improve_list[improve_key] = test_import_context
        elif improve_key == "test_class_signature":
            context_improve_list[improve_key] = test_class_signature
        elif improve_key == "test_class_field_context":
            context_improve_list[improve_key] = test_class_field_context
        elif improve_key == "test_class_function_signature_context":
            context_improve_list[improve_key] = test_class_function_signature_context
    
    code_context_format = "// Abstract Java Tested Class\n{be_test_import_context}\n\n{be_test_class_signature} {{\n{be_test_class_field_context}\n\n{be_test_class_function_signature_context}\n}}\n\n// Abstract Java Test Class\n{test_import_context}\n\n{test_class_signature} {{\n{test_class_field_context}\n\n{test_class_function_signature_context}\n}}"
    for remove_key in context_improve_list:
        code_context = code_context_format.format(
            be_test_import_context=be_test_import_context,
            be_test_class_signature=be_test_class_signature,
            be_test_class_field_context=be_test_class_field_context,
            be_test_class_function_signature_context=be_test_class_function_signature_context,
            test_import_context=test_import_context,
            test_class_signature=test_class_signature,
            test_class_field_context=test_class_field_context,
            test_class_function_signature_context=test_class_function_signature_context
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
            "be_test_import_context": True,
            "be_test_class_signature": True,
            "be_test_class_field_context": True,
            "be_test_class_function_signature_context": True,
            "test_import_context": True,
            "test_class_signature": True,
            "test_class_field_context": True,
            "test_class_function_signature_context": True
        }
    PROMPT_COMPLETE = """{code_context}\n\n// You are a professional Java test case writer, please create a test case named `{function_name}` for the `{be_test_class_name}` class, utilizing the provided abstract Java class context information and the following natural language annotations.\n\n{nl_context}"""
    PROMPT_CHAT = """```java\n{code_context}\n```\n\nYou are a professional Java test case writer, please create a test case named `{function_name}` for the `{be_test_class_name}` class, utilizing the provided abstract Java class context information and the following natural language annotations.\n\n```java\n{nl_context}\n```\n"""
    PROMPT_COMPLETE_FS0 = """// You are a professional Java test case writer, please create a test case named `{function_name}` for the `{be_test_class_name}` class, utilizing the provided abstract Java class context information and the following natural language annotations.\n\n{nl_context}"""
    PROMPT_CHAT_FS0 = """You are a professional Java test case writer, please create a test case named `{function_name}` for the `{be_test_class_name}` class, utilizing the provided abstract Java class context information and the following natural language annotations.\n\n```java\n{nl_context}\n```\n"""
    PROMPT_COMPLETE_FS1 = """// You are a professional Java test case writer, please create a test case named `testResultGetResultObjectUnknown` for the `MultiBackgroundInitializer` class, utilizing the provided abstract Java class context information and the following natural language annotations.\n\n/**\n * Tries to query the results of an unknown child initializer from the\n * results object. This should cause an exception.\n */\n@Test(expected = NoSuchElementException.class)\npublic void testResultGetResultObjectUnknown() throws ConcurrentException {{\n    final MultiBackgroundInitializer.MultiBackgroundInitializerResults res = checkInitialize();\n    res.getResultObject("unknown");\n}}\n\n// You are a professional Java test case writer, please create a test case named `{function_name}` for the `{be_test_class_name}` class, utilizing the provided abstract Java class context information and the following natural language annotations.\n\n{nl_context}"""
    PROMPT_CHAT_FS1 = """You are a professional Java test case writer, please create a test case named `{function_name}` for the `{be_test_class_name}` class, utilizing the provided abstract Java class context information and the following natural language annotations.\n\n```java\n{nl_context}\n```\n"""
    PROMPT_COMPLETE_FS2 = """// You are a professional Java test case writer, please create a test case named `testResultGetResultObjectUnknown` for the `MultiBackgroundInitializer` class, utilizing the provided abstract Java class context information and the following natural language annotations.\n\n/**\n * Tries to query the results of an unknown child initializer from the\n * results object. This should cause an exception.\n */\n@Test(expected = NoSuchElementException.class)\npublic void testResultGetResultObjectUnknown() throws ConcurrentException {{\n    final MultiBackgroundInitializer.MultiBackgroundInitializerResults res = checkInitialize();\n    res.getResultObject("unknown");\n}}\n\n// You are a professional Java test case writer, please create a test case named `testOptimizationsRemoveParentAfterRemoveChild` for the `PeepholeOptimizationsPass` class, utilizing the provided abstract Java class context information and the following natural language annotations.\n\n/**\n * Test the case where the first peephole optimization removes a node and the\n * second wants to remove (the now nonexistent) parent of that node.\n */\npublic void testOptimizationsRemoveParentAfterRemoveChild() {{\n  currentPeepholePasses = ImmutableList.<AbstractPeepholeOptimization>of(\n        new RemoveNodesNamedXOptimization(),\n        new RemoveParentVarsForNodesNamedX());\n  test("var x,y; var z;", "var y; var z;");\n}}\n\n// You are a professional Java test case writer, please create a test case named `{function_name}` for the `{be_test_class_name}` class, utilizing the provided abstract Java class context information and the following natural language annotations.\n\n{nl_context}"""
    PROMPT_CHAT_FS2 = """You are a professional Java test case writer, please create a test case named `{function_name}` for the `{be_test_class_name}` class, utilizing the provided abstract Java class context information and the following natural language annotations.\n\n```java\n{nl_context}\n```\n"""
    PROMPT_COMPLETE_FS3 = """// You are a professional Java test case writer, please create a test case named `testResultGetResultObjectUnknown` for the `MultiBackgroundInitializer` class, utilizing the provided abstract Java class context information and the following natural language annotations.\n\n/**\n * Tries to query the results of an unknown child initializer from the\n * results object. This should cause an exception.\n */\n@Test(expected = NoSuchElementException.class)\npublic void testResultGetResultObjectUnknown() throws ConcurrentException {{\n    final MultiBackgroundInitializer.MultiBackgroundInitializerResults res = checkInitialize();\n    res.getResultObject("unknown");\n}}\n\n// You are a professional Java test case writer, please create a test case named `testOptimizationsRemoveParentAfterRemoveChild` for the `PeepholeOptimizationsPass` class, utilizing the provided abstract Java class context information and the following natural language annotations.\n\n/**\n * Test the case where the first peephole optimization removes a node and the\n * second wants to remove (the now nonexistent) parent of that node.\n */\npublic void testOptimizationsRemoveParentAfterRemoveChild() {{\n  currentPeepholePasses = ImmutableList.<AbstractPeepholeOptimization>of(\n        new RemoveNodesNamedXOptimization(),\n        new RemoveParentVarsForNodesNamedX());\n  test("var x,y; var z;", "var y; var z;");\n}}\n\n// You are a professional Java test case writer, please create a test case named `testNoRemovePrototypeDefinitionsOutsideGlobalScope1` for the `NameAnalyzer` class, utilizing the provided abstract Java class context information and the following natural language annotations.\n\n/**\n * Do not "prototype" property of variables that are not being\n * tracked (because they are local).\n * @bug 1809442\n */\npublic void testNoRemovePrototypeDefinitionsOutsideGlobalScope1() {{\n  testSame("function f(arg){{}}" +\n           "" +\n           "(function(){{" +\n           "  var O = {{}};" +\n           "  O.prototype = 'foo';" +\n           "  f(O);" +\n           "}})()");\n}}\n\n// You are a professional Java test case writer, please create a test case named `{function_name}` for the `{be_test_class_name}` class, utilizing the provided abstract Java class context information and the following natural language annotations.\n\n{nl_context}"""
    PROMPT_CHAT_FS3 = """You are a professional Java test case writer, please create a test case named `{function_name}` for the `{be_test_class_name}` class, utilizing the provided abstract Java class context information and the following natural language annotations.\n\n```java\n{nl_context}\n```\n"""
    PROMPT_COMPLETE_FS4 = """// You are a professional Java test case writer, please create a test case named `testResultGetResultObjectUnknown` for the `MultiBackgroundInitializer` class, utilizing the provided abstract Java class context information and the following natural language annotations.\n\n/**\n * Tries to query the results of an unknown child initializer from the\n * results object. This should cause an exception.\n */\n@Test(expected = NoSuchElementException.class)\npublic void testResultGetResultObjectUnknown() throws ConcurrentException {{\n    final MultiBackgroundInitializer.MultiBackgroundInitializerResults res = checkInitialize();\n    res.getResultObject("unknown");\n}}\n\n// You are a professional Java test case writer, please create a test case named `testOptimizationsRemoveParentAfterRemoveChild` for the `PeepholeOptimizationsPass` class, utilizing the provided abstract Java class context information and the following natural language annotations.\n\n/**\n * Test the case where the first peephole optimization removes a node and the\n * second wants to remove (the now nonexistent) parent of that node.\n */\npublic void testOptimizationsRemoveParentAfterRemoveChild() {{\n  currentPeepholePasses = ImmutableList.<AbstractPeepholeOptimization>of(\n        new RemoveNodesNamedXOptimization(),\n        new RemoveParentVarsForNodesNamedX());\n  test("var x,y; var z;", "var y; var z;");\n}}\n\n// You are a professional Java test case writer, please create a test case named `testNoRemovePrototypeDefinitionsOutsideGlobalScope1` for the `NameAnalyzer` class, utilizing the provided abstract Java class context information and the following natural language annotations.\n\n/**\n * Do not "prototype" property of variables that are not being\n * tracked (because they are local).\n * @bug 1809442\n */\npublic void testNoRemovePrototypeDefinitionsOutsideGlobalScope1() {{\n  testSame("function f(arg){{}}" +\n           "" +\n           "(function(){{" +\n           "  var O = {{}};" +\n           "  O.prototype = 'foo';" +\n           "  f(O);" +\n           "}})()");\n}}\n\n// You are a professional Java test case writer, please create a test case named `testResultGetInitializerUnknown` for the `MultiBackgroundInitializer` class, utilizing the provided abstract Java class context information and the following natural language annotations.\n\n/**\n * Tries to query an unknown child initializer from the results object. This\n * should cause an exception.\n */\n@Test(expected = NoSuchElementException.class)\npublic void testResultGetInitializerUnknown() throws ConcurrentException {{\n    final MultiBackgroundInitializer.MultiBackgroundInitializerResults res = checkInitialize();\n    res.getInitializer("unknown");\n}}\n\n// You are a professional Java test case writer, please create a test case named `{function_name}` for the `{be_test_class_name}` class, utilizing the provided abstract Java class context information and the following natural language annotations.\n\n{nl_context}"""
    PROMPT_CHAT_FS4 = """You are a professional Java test case writer, please create a test case named `{function_name}` for the `{be_test_class_name}` class, utilizing the provided abstract Java class context information and the following natural language annotations.\n\n```java\n{nl_context}\n```\n"""
    
    PROMPT_COMPLETE_FS = [
        PROMPT_COMPLETE_FS0, PROMPT_COMPLETE_FS1, PROMPT_COMPLETE_FS2, PROMPT_COMPLETE_FS3, PROMPT_COMPLETE_FS4
    ]
    PROMPT_CHAT_FS = [
        PROMPT_CHAT_FS0, PROMPT_CHAT_FS1, PROMPT_CHAT_FS2, PROMPT_CHAT_FS3, PROMPT_CHAT_FS4
    ]
    
    function_data = json.load(open(os.path.join(FILE_DIR, 'data', 'task_testgen_filtered_final.json')))
    print("Total number of tasks: ", len(function_data))
    
    prompted_data = []
    
    for example in tqdm(function_data):
        if len(example["testmethods"]) == 0: continue
        
        be_test_import_context = example["be_test_import_context"] if context["be_test_import_context"] else ""
        be_test_class_signature = example["be_test_class_signature"] if context["be_test_class_signature"] else ""
        be_test_class_field_context = example["be_test_class_field_context"] if context["be_test_class_field_context"] else ""
        be_test_class_function_signature_context = example["be_test_class_function_signature_context"] if context["be_test_class_function_signature_context"] else ""
        test_import_context = example["test_import_context"] if context["test_import_context"] else ""
        test_class_signature = example["test_class_signature"] if context["test_class_signature"] else ""
        test_class_field_context = example["test_class_field_context"] if context["test_class_field_context"] else ""
        test_class_function_signature_context = example["test_class_function_signature_context"] if context["test_class_function_signature_context"] else ""
        
        if len(improve_list) > 0:
            becut, code_context = get_code_context_order(be_test_import_context, be_test_class_signature,
                                                be_test_class_field_context, be_test_class_function_signature_context,
                                                test_import_context, test_class_signature,
                                                test_class_field_context, test_class_function_signature_context,
                                                context_length, improve_list=improve_list)
        else:
            becut, code_context = get_code_context(be_test_import_context, be_test_class_signature,
                                                be_test_class_field_context, be_test_class_function_signature_context,
                                                test_import_context, test_class_signature,
                                                test_class_field_context, test_class_function_signature_context,
                                                context_length)
        
        if becut:
            code_context = "\n".join(code_context.split("\n")[1:])
        # comment = pretty_comment(comment, example["indent"])
        # if comment[1] != " ":
        #     print(comment)
        #     print(len(example["indent"]))
        comment = example["function_comment"]
        
        if context["few_shot"] == -1:
            prompt_chat = PROMPT_CHAT.format(
                code_context=code_context,
                function_name=example["function_name"],
                be_test_class_name=example["be_test_class_name"],
                nl_context=f'{comment}\n{example["indent"]}{example["function_signature"].strip()}'+" {\n"
            )
            code_context = "// "+ "\n// ".join(code_context.split("\n"))
            prompt_complete = PROMPT_COMPLETE.format(
                code_context=code_context,
                function_name=example["function_name"],
                be_test_class_name=example["be_test_class_name"],
                nl_context=f'{comment}\n{example["indent"]}{example["function_signature"].strip()}'+" {\n"
            )
        elif context["few_shot"] in [0,1,2,3,4]:
            prompt_chat = PROMPT_CHAT_FS[context["few_shot"]].format(
                function_name=example["function_name"],
                be_test_class_name=example["be_test_class_name"],
                nl_context=f'{comment}\n{example["indent"]}{example["function_signature"].strip()}'+" {\n"
            )
            prompt_complete = PROMPT_COMPLETE_FS[context["few_shot"]].format(
                function_name=example["function_name"],
                be_test_class_name=example["be_test_class_name"],
                nl_context=f'{comment}\n{example["indent"]}{example["function_signature"].strip()}'+" {\n"
            )
            
        prompt_chat_with_comment = get_prompt_with_comment(prompt_chat)
        prompt_complete_with_comment = get_prompt_with_comment(prompt_complete)
        
        prompted_example = {
                "task_id": hash_string(example["task_id"]),
                "project":example["project_id"], "bug_id":example["bug_id"], 
                "testmethods": example["testmethods"], "source_dir":example["source_dir"], 
                "location":example["location"],
                "start": example["function_star_line"], "end": example["function_end_line"], 
                "function":example["function"], "comment":example["function_comment"], 
                "function_name": example["function_name"],
                "prompt_chat":prompt_chat, 
                "prompt_chat_with_comment":prompt_chat_with_comment,
                "prompt_complete":prompt_complete, 
                "prompt_complete_with_comment":prompt_complete_with_comment, 
                "be_test_import_context": example["be_test_import_context"],
                "be_test_class_signature": example["be_test_class_signature"],
                "be_test_class_field_context": example["be_test_class_field_context"],
                "be_test_class_function_signature_context": example["be_test_class_function_signature_context"],
                "test_import_context": example["test_import_context"],
                "test_class_signature": example["test_class_signature"],
                "test_class_field_context": example["test_class_field_context"],
                "test_class_function_signature_context": example["test_class_function_signature_context"],
                "function_signature":example["function_signature"], 
                "source":example["source"], 
                "classmethods": example["classmethods"],
                "be_test_class_long_name": example["classmethods"][0]["be_test_class_name"],
                "indent": example["indent"],
            }
        
        prompted_data.append(prompted_example)
    prompted_data.sort(key=lambda x: x["task_id"])
    dataset = {"code_ucb_testgen": prompted_data}
    json.dump(dataset, open(os.path.join(FILE_DIR, 'data', f'task_testgen_bench_{save_suffix}.json'), 'w'), indent=4, ensure_ascii=False)

def process_get_correct_result(FILE_DIR, save_suffix=""):
    data = json.load(open(os.path.join(FILE_DIR, 'data', f'task_testgen_bench_{save_suffix}.json')))
    correct_result = []
    for idx, example in enumerate(data["code_ucb_testgen"]):
        result = {
            "task_id": idx,
            "outputs": [example["prompt_complete_with_comment"] + "\n" + example["function"]]*10
        }
        correct_result.append(result)
        
    json.dump(correct_result, open(os.path.join(FILE_DIR, 'data', f'task_testgen_correct_result_{save_suffix}.json'), 'w'), indent=4, ensure_ascii=False)


def generate_default_dataset(FILE_DIR):
    save_suffix_list = ["default|2048"]
    for save_suffix in save_suffix_list:
        context_str, context_length = save_suffix.split("|")
        context = {
            "few_shot": -1,
            "be_test_import_context": 1,
            "be_test_class_signature": 1,
            "be_test_class_field_context": 1,
            "be_test_class_function_signature_context": 1,
            "test_import_context": 1,
            "test_class_signature": 1,
            "test_class_field_context": 1,
            "test_class_function_signature_context": 1
        }
        process_get_prompt(FILE_DIR, context, context_length=int(context_length), save_suffix=save_suffix)
        process_get_correct_result(FILE_DIR, save_suffix=save_suffix)

def generate_fs_dataset(FILE_DIR):
    save_suffix_list = ["fs0|2048", "fs1|2048", "fs2|2048", "fs3|2048", "fs4|2048"]
    for save_suffix in save_suffix_list:
        context_str, context_length = save_suffix.split("|")
        context = {
            "few_shot": int(context_str[-1]),
            "be_test_import_context": 1,
            "be_test_class_signature": 1,
            "be_test_class_field_context": 1,
            "be_test_class_function_signature_context": 1,
            "test_import_context": 1,
            "test_class_signature": 1,
            "test_class_field_context": 1,
            "test_class_function_signature_context": 1
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
    
    print("Done")
    

if __name__ == '__main__':
    main()    