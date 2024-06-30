import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import re
import chardet
import openai
import time 
import traceback

from tqdm import tqdm

import hashlib

def robust_multiprocessing(worker, tasks):
    results = []
    pool_count = 3*os.cpu_count()//4
    with ProcessPoolExecutor(max_workers=pool_count) as executor:
        for i in tqdm(range(0, len(tasks), pool_count*4),  desc="Processing..."):
            sub_tasks = tasks[i:i+pool_count*4]
            results.extend(
                list(executor.map(worker, *zip(*sub_tasks)))
            )

        if len(tasks) > len(results):
            results.extend(
                list(executor.map(worker, *zip(*tasks[len(results):])))
            )
        
    return results

def fast_multiprocessing(worker, tasks):
    results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()//4) as executor:
        # Submit all your tasks to the executor
        future_tasks = set()
        for task in tqdm(tasks):
            future_tasks.add(executor.submit(worker, *task))
            time.sleep(0.001)
        # Use tqdm to display progress
        with tqdm(as_completed(future_tasks), total=len(tasks), desc="Processing...") as progress_bar:
            for future in progress_bar:
                # Append the result to a list
                results.append(future.result())
        
    return results

def get_indent(function):
    signture = function.split("\n")[0]
    indent_size = len(signture) - len(signture.strip())
    return function[:indent_size]

def hash_string(s):
    assert isinstance(s, str), 'Input must be string'
    
    s = s.encode() # 转为二进制
    return hashlib.sha256(s).hexdigest() # 计算哈希，然后以十六进制格式返回

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

def check_is_complete_function(node):
    if node.path.endswith("class_body|method_declaration"):
        return True
    if node.path.endswith("enum_body_declarations|method_declaration"):
        return True
    if node.path.endswith("class_body|constructor_declaration"):
        return True
    return False

def read_file(file_path):
    with open(file_path, 'rb') as f:
        content = f.read()

    # Detect the encoding
    encoding = chardet.detect(content)['encoding']

    # Decode the content
    decoded_content = content.decode(encoding)
    decoded_content = "\n".join(decoded_content.splitlines())
    # decoded_content = decoded_content.replace("\r\n", "\n")

    return decoded_content

def create_chatgpt_config(
    message: str,
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "gpt-3.5-turbo",
):
    config = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": batch_size,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": message},
        ],
    }
    return config

def call_openai_api(item):
    message, args_dict = item
    model_id = args_dict.get("model_id", "gpt-3.5-turbo")
    temperature = args_dict.get("temperature", 0.2)
    temperature = temperature if temperature > 0 else 0.01
    max_length = args_dict.get("max_length", 1024)
    num_return_sequences = args_dict.get("num_return_sequences", 1)
    config = create_chatgpt_config(
        message=message,
        max_tokens=max_length,
        temperature=temperature,
        batch_size=num_return_sequences,
        model=model_id,
    )
    backoff_time = 3
    
    for _ in range(10):
        try:
            result = openai.ChatCompletion.create(**config)
            break
        except Exception:
            traceback.print_exc()
            time.sleep(backoff_time)
            backoff_time *= 1.5
            result = None

    return result

def gen_worker(item):
    def get_comment(result):
        if result is None: return "", ""
        if "error" in result: return "", ""
        if "choices" not in result: return "", ""
        if len(result["choices"]) == 0: return "", ""
        content = result["choices"][0]["message"]["content"]
        
        comment = ""
        start, end = False, False
        for line in content.split("\n"):
            if "/**" in line:
                start = True
            if "*/" in line:
                end = True
            if start:
                comment += line + "\n"
            if end:
                break
        
        return comment, content
        
    task, model_id, prompt = item
    args = {
        "model_id": model_id,
        "temperature": 0.2,
        "max_tokens": 1024
    }
    # print(prompt)
    # input("Continue?")
    result = call_openai_api((prompt, args))
    # print(result)
    task["better_comment"], task["content_comment"] = get_comment(result)
    # print(task["better_comment"])
    return task


def get_prompt_with_comment(prompt):
    code_lines = prompt.splitlines()
    for idx in range(len(code_lines)-1, 0, -1):
        if "*/" in code_lines[idx]:
            code_lines = code_lines[:idx+1]
            break
    prompt_with_comment = "\n".join(code_lines) + "\n"
    return prompt_with_comment

def pretty_comment(comment, indent):
    index = comment.index("/")
    new_comment = []
    for line in comment.split("\n"):
        line = indent + line[index:]
        new_comment.append(line)
    return "\n".join(new_comment).rstrip()