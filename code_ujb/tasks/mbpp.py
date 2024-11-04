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
import itertools
import os

import numpy as np
from code_ujb.Task import Task

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from code_ujb.tasks.custom_metrics.execute import check_correctness

_CITATION = """
@article{austin2021program,
  title={Program Synthesis with Large Language Models},
  author={Austin, Jacob and Odena, Augustus and Nye, Maxwell and Bosma, Maarten and Michalewski, Henryk and Dohan, David and Jiang, Ellen and Cai, Carrie and Terry, Michael and Le, Quoc and others},
  journal={arXiv preprint arXiv:2108.07732},
  year={2021}
}
"""

def compute_code_eval(predictions, references, k=[1, 10, 100], num_workers=4, timeout=3.0):
    """Returns the scores"""

    if os.name == "nt":
        raise NotImplementedError("This metric is currently not supported on Windows.")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        n_samples = 0
        results = defaultdict(list)

        for task_id, (candidates, ref) in enumerate(zip(predictions, references)):
            test_case, task_idx = ref[0], ref[1]
            for g_idx, candidate in enumerate(candidates):
                test_program = candidate + "\n" + test_case
                args = (test_program, timeout, task_idx, g_idx)
                future = executor.submit(check_correctness, *args)
                futures.append(future)
                n_samples += 1

        for future in as_completed(futures):
            result = future.result()
            result["task_idx"] = result["task_id"]
            result["g_idx"] = result["completion_id"]
            results[result["task_idx"]].append(result)

    total, correct = [], []
    for result in results.values():
        passed = [r["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    ks = k
    if not isinstance(ks, (list, tuple)):
        ks = [ks]
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean() for k in ks if (total >= k).all()}

    details = []
    for result in results.values():
        details.extend(result)
    return pass_at_k, details


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

class MBPP(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "mbpp"

    def __init__(self):
        super().__init__(
            stop_words=["\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/", "\n```"],
            requires_execution=True,
        )
        self.dataset = [item for item in self.dataset["test"]]
        for t_idx, item in enumerate(self.dataset):
            item["task_idx"] = t_idx

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        # the wrong split of mbpp can be loaded with old datasets cache
        assert (
            len(self.dataset) == 500
        ), "please ensure you have the latest version of MBPP dataset, try deleting its old cache"
        return self.dataset

    def get_prompt_complete(self, doc):
        """Builds the prompt for the LM to generate from.
        MBPP prompt is built following to InCoder (Fried et al.) approach
        prompt = docstring that includes one test
        """
        description = doc["text"]
        test_example = doc["test_list"][0]
        prompt = f'"""\n{description}\n{test_example}\n"""\n'
        return prompt

    def get_prompt_chat(self, doc):
        raise NotImplementedError()

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return ["\n".join(doc["test_list"]), doc["task_idx"]]

    def postprocess_generation_complete(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        prompt = self.get_prompt(self.get_dataset()[idx])
        generation = generation[len(prompt) :]
        return prompt + self._stop_at_stop_token(generation, self.stop_words)

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
        results, detail = compute_code_eval(
            references=references,
            predictions=generations,
            k=[1,2,5,10,20,40,100],
            num_workers=os.cpu_count()//2
        )
        results["detail"] = detail
        return results
