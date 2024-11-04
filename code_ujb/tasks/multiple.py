"""MultiPL-E: A Scalable and Extensible Approach to Benchmarking Neural Code Generation
https://arxiv.org/abs/2107.03374

MultiPL-E is a dataset for evaluating large language models for code generation that supports 18 programming languages.
It takes the OpenAI "HumanEval" and the MBPP Python benchmarks and uses little compilers to translate them to other languages.

Homepage: https://nuprl.github.io/MultiPL-E/
"""

import json
import os
import re
import tempfile
from multiprocessing import cpu_count
from pathlib import Path
from time import time

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from code_ujb.Task import Task
from code_ujb.tasks.custom_metrics.multiple_metrics.evaluation import \
    evaluate_problem
from code_ujb.tasks.custom_metrics.multiple_metrics.single_experiment_pass_k import \
    for_file

_CITATION = """
@article{cassano2022scalable,
  title={A Scalable and Extensible Approach to Benchmarking NL2Code for 18 Programming Languages},
  author={Cassano, Federico and Gouwar, John and Nguyen, Daniel and Nguyen, Sydney and Phipps-Costin, Luna and Pinckney, Donald and Yee, Ming Ho and Zi, Yangtian and Anderson, Carolyn Jane and Feldman, Molly Q and others},
  journal={arXiv preprint arXiv:2208.08227},
  year={2022}
}
"""

LANGUAGES = [
    "py",
    "sh",
    "cpp",
    "cs",
    "d",
    "go",
    "java",
    "js",
    "jl",
    "lua",
    "pl",
    "php",
    "r",
    "rkt",
    "rb",
    "rs",
    "scala",
    "swift",
    "ts",
]


def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {multiple-py: Task, multiple-java: Task}
    """
    return {f"multiple-{language}": create_task(language) for language in LANGUAGES}


def create_task(language):
    class MultiPLE(GeneralMultiPLE):
        def __init__(self):
            super().__init__(language)

    return MultiPLE

def for_detail(data):
    results = []
    for r in data["results"]:
        passed = True if r["status"] == "OK" and r["exit_code"] == 0 else False
        results.append({"task_idx": data["task_idx"], "g_idx": r["g_idx"], "passed": passed})
    return results

class GeneralMultiPLE(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "nuprl/MultiPL-E"
    DATASET_NAME = None
    DATASET_REVISION = "d23b094346c5dbda1080a74bb2a24c18adbf7409"

    def __init__(self, language):
        self.language = language
        self.DATASET_NAME = f"humaneval-{language}"
        # we need the dataset to get stop words for each language
        for _ in range(10):
            try:
                self.dataset = load_dataset(
                    GeneralMultiPLE.DATASET_PATH,
                    self.DATASET_NAME,
                    revision=self.DATASET_REVISION,
                    trust_remote_code=True)
                break
            except Exception as e:
                print(f"Error loading dataset: {e}")
        if not hasattr(self, "dataset"):
            raise Exception(f"Failed to load dataset after 10 attempts")
        stop_words = self.dataset["test"][0]["stop_tokens"] + ["<file_sep>"]
        super().__init__(
            stop_words=stop_words,
            requires_execution=True,
        )
        self.dataset = [item for item in self.dataset["test"]]
        for t_idx, item in enumerate(self.dataset):
            item["task_idx"] = t_idx

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset

    def get_prompt_complete(self, doc):
        """Builds the prompt for the LM to generate from."""
        return doc["prompt"].strip()

    def get_prompt_chat(self, doc):
        raise NotImplementedError()

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return doc

    @staticmethod
    def remove_last_block(string, stop_words):
        # Remove the last block of the code containing stop_words for HumanEval
        string_list = re.split("(%s)" % "|".join(stop_words), string)
        # last string should be ""
        return "".join(string_list[:-2])


    def postprocess_generation_complete(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for this task)
        """
        prompt = self.get_prompt(self.get_dataset()[idx])
        completion = generation[len(prompt) :]
        return prompt + self._stop_at_stop_token(completion, self.stop_words)

    def postprocess_generation_chat(self, generation, idx):
        raise NotImplementedError()

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        # get prompts and problem names
        prompts_names = [
            {"prompt": doc["prompt"], "name": doc["name"]}
            for i, doc in enumerate(self.get_dataset())
            if i < len(generations)
        ]
        # a common temp dir for all the problems
        temp_dir = tempfile.mkdtemp()
        [os.remove(p) for p in Path(temp_dir).glob("*.json")]
        list_files = []
        for (prompt_name, generation, reference) in zip(
            prompts_names, generations, references
        ):
            problem = {
                "task_idx": reference["task_idx"],
                "name": prompt_name["name"],
                "language": self.language,
                "prompt": prompt_name["prompt"],
                "completions": generation,
                "tests": reference["tests"],
            }
            # each problem is save in a json file
            temp_file_name = os.path.join(temp_dir, f"{prompt_name['name']}.json")
            list_files.append(temp_file_name)
            with open(temp_file_name, "wt") as f:
                json.dump(problem, f)
        print(
            f"Saved {len(list_files)} problems in {temp_dir} for evaluation, each problem has {len(generations[0])} completions"
        )

        all_results = []
        # execute the problems to evaluate them
        max_workers = cpu_count()//2 if cpu_count() > 2 else 1
        for file in tqdm(list_files):
            one_results = evaluate_problem(temp_dir, file, max_workers)
            all_results.append(one_results)

        # compute pass@k scores
        result_array = np.array(
            [for_file(r, k=[1, 2, 5, 10, 20, 40, 50, 80, 100]) for r in all_results]
        )
        detail = []
        for r in all_results:
            detail.extend(for_detail(r))
            
        result = result_array.mean(axis=0)
        results = {
            f"pass@{k}": v
            for k, v in zip([1, 2, 5, 10, 20, 40, 50, 80, 100], result)
            if k <= len(generations[0])
        }
        results["detail"] = detail
        [os.remove(p) for p in Path(temp_dir).glob("*.json")]
        os.removedirs(temp_dir)
        return results
