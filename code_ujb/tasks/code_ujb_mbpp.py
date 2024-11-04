"""Program Synthesis with Large Language Models
https://arxiv.org/abs/2108.07732

The benchmark consists of around 1,000 crowd-sourced Python programming problems, 
designed to be solvable by entry level programmers, covering programming fundamentals, 
standard library functionality, and so on. Each problem consists of a task description, 
code solution and 3 automated test cases. As described in the paper, a subset of the data
has been hand-verified by the authors.

Homepage:: https://github.com/google-research/google-research/tree/master/mbpp
"""

import re
import time
import itertools
import os
import numpy as np

from datasets import load_dataset
from tqdm import tqdm
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from code_ujb.Task import Task
from code_ujb.tasks.custom_metrics.execute import check_correctness

_CITATION = """
@article{austin2021program,
  title={Program Synthesis with Large Language Models},
  author={Austin, Jacob and Odena, Augustus and Nye, Maxwell and Bosma, Maarten and Michalewski, Henryk and Dohan, David and Jiang, Ellen and Cai, Carrie and Terry, Michael and Le, Quoc and others},
  journal={arXiv preprint arXiv:2108.07732},
  year={2021}
}
"""



def compute_code_eval(predictions, references, k=[1, 10, 100], num_workers=32, timeout=3.0):
    """Returns the scores"""

    if os.name == "nt":
        raise NotImplementedError("This metric is currently not supported on Windows.")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        for task_id, (candidates, test_case) in enumerate(zip(predictions, references)):
            for candidate in candidates:
                test_program = candidate + "\n" + test_case
                args = (test_program, timeout, task_id, completion_id[task_id])
                future = executor.submit(check_correctness, *args)
                futures.append(future)
                completion_id[task_id] += 1
                n_samples += 1

        for future in as_completed(futures):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    ks = k
    if not isinstance(ks, (list, tuple)):
        ks = [ks]
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean() for k in ks if (total >= k).all()}

    return pass_at_k, results


def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

class StreamStopUJBComplete():
    def __init__(self, function_signature, mode="complete"):
        self.function_signature = function_signature
        self.mode = mode
    
    def check_stop(self, generation):
        return False

class MBPP(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "ZHENGRAN/mbpp"

    def __init__(self):
        super().__init__(
            stop_words=["\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/", "\n```"],
            requires_execution=True,
        )
        print("Using Dataset:", self.DATASET_PATH)
        self.dataset = load_dataset(self.DATASET_PATH)

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        dataset = self.dataset["train"]
        # the wrong split of mbpp can be loaded with old datasets cache
        assert (
            len(dataset) == 500
        ), "please ensure you have the latest version of MBPP dataset, try deleting its old cache"
        return dataset

    def get_prompt_complete(self, doc):
        """Builds the prompt for the LM to generate from."""
        return doc["prompt_complete"].strip()
    
    def get_prompt_chat(self, doc):
        return doc["prompt_chat"].strip()
    
    def get_prompt_byidx(self, idx, mode="complete"):
        """Builds the prompt for the LM to generate from."""
        return self.get_prompt(self.get_dataset()[idx], mode=mode)

    def get_id_byidx(self, idx):
        """Builds the prompt for the LM to generate from."""
        return self.get_dataset()[idx]["task_id"]
    
    def get_stream_stop(self, idx, mode="complete"):
        return StreamStopUJBComplete(self.get_dataset()[idx]["function_signature"], mode=mode)

    def get_reference(self, idx):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return "\n".join(self.get_dataset()[idx]["test_list"])

    def postprocess_generation_complete(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        generation = generation.replace(self.get_prompt_byidx(idx), "")
        generation = generation.split("if __name__ == '__main__':")[0]
        return generation
        # prompt = self.get_prompt(self.dataset["train"][idx])
        function_signature = self.dataset["train"][idx]["function_signature"]
        # print("prompt", prompt_with_signature)
        # print("generation", generation)
        generation = generation.split(function_signature)[-1]
        for stop_word in self.stop_words:
            generation = generation.split(stop_word)[0]
        return function_signature + generation

    def postprocess_generation_chat(self, generation, idx):
        def extract_code_block(gen, pattern):
            try:
                code_block = re.findall(pattern, gen, re.DOTALL | re.IGNORECASE)[0]
                return code_block
            except (IndexError, TypeError):
                return None
        
        patterns = [
            r'```python\n(.*?)```',
            r'```\n(.*?)```',
            r'\[PYTHON\]\n(.*?)\[/PYTHON\]'
        ]
        
        for pattern in patterns:
            code_block = extract_code_block(generation, pattern)
            if code_block is not None:
                return code_block
        if generation is None:
            return "$ERROR$"
        return generation
    
    
    def evaluate(self, generations):
        # generations = generations[:10]
        all_generations = []
        all_references = []
        all_results = []
        for generation in tqdm(generations, total=len(generations)):
            idx = generation["task_idx"]
            gens = generation["outputs"]

            reference = self.get_reference(idx)
            all_references.append(reference)
            all_generations.append(gens)

            results, _ = compute_code_eval(
                references=[reference],
                predictions=[gens],
                k=[1, 5, 10, 20, 50, 100]
            )
            results['idx'] = idx
            all_results.append(results)
            # print(results)
        
        return {k:np.mean([item[k] for item in all_results]) for k in all_results[0] if 'pass' in k}

        # results, _ = compute_code_eval(
        #         references=all_references,
        #         predictions=all_generations,
        #         k=[1, 5, 10, 20, 50, 100]
        #     )
        # return results