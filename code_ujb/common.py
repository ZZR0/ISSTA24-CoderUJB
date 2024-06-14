import os
import random
import time
import json
from openai import OpenAI

OPENAI_MODEL_ID = [
    "gpt-3.5-turbo",
    'gpt-3.5-turbo-0125',
    'gpt-3.5-turbo-0301',
    'gpt-3.5-turbo-0613',
    'gpt-3.5-turbo-1106',
    "gpt-3.5-turbo-16k",
    'gpt-3.5-turbo-16k-0613',
    'gpt-3.5-turbo-instruct',
    'gpt-4-0125-preview',
    'gpt-4-0613',
    'gpt-4-1106-preview',
    "gpt-4-turbo",
    "gpt-4",
    'claude-2',
    'claude-2.0',
    'claude-2.1',
    'claude-3-haiku-20240307',
    'claude-3-opus-20240229',
    'claude-3-sonnet-20240229',
]

API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"

def chat_compeletion_openai(api_urls, gen_mode, model, question, messages, temperature, max_tokens, stop_str_list):
    output = API_ERROR_OUTPUT

    for _ in range(API_MAX_RETRY):
        try:
            api_url = random.choice(api_urls)
            api_url, api_key = api_url.split("::")
            client = OpenAI(
                api_key=api_key,
                base_url=api_url,
            )
            if gen_mode == "chat":
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    n=1,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                output = response.choices[0].message.content
            else:
                prompt = question
                output = client.completions.create(model=model, prompt=prompt, 
                                                    max_tokens=max_tokens, 
                                                    temperature=temperature, 
                                                    stop=stop_str_list)
                output = prompt + output
            break
        except Exception as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)

    return output

def reorg_output_file(save_generations_path, num_samples):
    def merged_output(a, b):
        a["outputs"].extend(b["outputs"])
        return a
    """Sort by question id and de-duplication"""
    answers = {}
    with open(save_generations_path+".tmp", "r") as fin:
        for l in fin:
            output = json.loads(l)
            tid = output["task_id"]
            if tid not in answers:
                answers[tid] = output
            else:
                answers[tid] = merged_output(answers[tid], output)

    qids = sorted(list(answers.keys()))
    answers = [answers[qid] for qid in qids]
    for answer in answers:
        answer["outputs"] = answer["outputs"][:num_samples]
    
    json.dump(answers, open(save_generations_path, "w"), indent=4)

