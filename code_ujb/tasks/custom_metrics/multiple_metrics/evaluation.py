from copy import deepcopy
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Optional

from .containerized_eval import eval_string_script

# Get working directory
WORKING_DIR = Path(__file__).parent.parent

# program: str => Result
CACHE = dict()
CACHE_LOCK = Lock()


def cache_get(program: str) -> Optional[dict]:
    if program in CACHE:
        result = CACHE[program]
        return result
    else:
        return None


def cache_set(program: str, result: dict):
    if program in CACHE:
        print("Setting already-existing cache")
    CACHE[program] = result


def cached_eval_script(problem, index) -> dict:
    # here prompt is already included in completions
    time.sleep(random.uniform(0.0, 0.5))
    program = problem["completions"][index] + "\n" + problem["tests"]
    result_yaml = dict()
    result_yaml["g_idx"] = index
    result_dict = eval_string_script(problem["language"], program)
    for k in result_dict.keys():
        result_yaml[k] = result_dict[k]
        result_yaml["timestamp"] = int(time.time())
    return result_yaml


def get_test_results_json_path(
    output_dir: str, problem_json_path: str, input_dir: Path
) -> Path:
    suffixes = ".results.json"
    problem_name = problem_json_path[: -len(".json")]
    if input_dir:
        raise ValueError("input dir given")
        return Path(output_dir) / (
            problem_json_path.relative_to(input_dir).parent / (problem_name + suffixes)
        )
    return Path(output_dir) / (problem_name + suffixes)


def evaluate_problem(
    output_dir: str, problem_json_path: str, max_workers: int, input_dir: Path = None
):
    with open(problem_json_path, "r") as f:
        problem = json.load(f)
    test_results_path = get_test_results_json_path(
        output_dir, problem_json_path, input_dir
    )
    test_results_path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)

    test_results = problem.copy()
    del test_results["completions"]
    test_results["results"] = []

    num_problems = len(problem["completions"])

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(
            lambda index: cached_eval_script(problem, index),
            range(0, num_problems),
        ):
            test_results["results"].append(result)

    return test_results