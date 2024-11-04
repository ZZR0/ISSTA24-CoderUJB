"""Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation
https://openreview.net/forum?id=1qvx610Cu7

The MBPP+ dataset is created by the EvalPlus framework which extends the original MBPP dataset
by adding more automatically generated test cases to each problem. Note MBPP+ only includes 399
tasks which are a subset of the original MBPP dataset. The subset is selected from the sanitized
MBPP (a subset of manually examined tasks by the original MBPP authors) and EvalPlus further
removes low-quality and ill-formed tasks for benchmark quality control.

Homepage: https://github.com/evalplus/evalplus
"""

import os

from code_ujb.tasks.mbpp import MBPP
from code_ujb.tasks.custom_metrics.code_eval import compute_code_eval

_CITATION = """
@inproceedings{evalplus,
  title = {Is Your Code Generated by Chat{GPT} Really Correct? Rigorous Evaluation of Large Language Models for Code Generation},
  author = {Liu, Jiawei and Xia, Chunqiu Steven and Wang, Yuyao and Zhang, Lingming},
  booktitle = {Thirty-seventh Conference on Neural Information Processing Systems},
  year = {2023},
  url = {https://openreview.net/forum?id=1qvx610Cu7},
}
"""


class MBPPPlus(MBPP):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "evalplus/mbppplus"

    def get_prompt_complete(self, doc):
        """Builds the prompt for the LM to generate from.
        MBPP prompt is built following to InCoder (Fried et al.) approach
        prompt = docstring that includes one test
        """
        description = doc["prompt"]  # sanitized testset use "prompt" instead of "text"
        test_example = doc["test_list"][0]
        prompt = f'"""\n{description}\n{test_example}\n"""\n'
        return prompt

    def get_prompt_chat(self, doc):
        raise NotImplementedError()

    # NOTE(@ganler): MBPP+ extends the original MBPP jsonl data with a "test" field which
    #                includes the testing code ready for execution. Note the "test" field
    #                is different from HumanEval(+) which further requires a `check` func
    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        use_mbpp_tests = os.getenv("MBBPPLUS_USE_MBPP_TESTS", "0")
        if use_mbpp_tests == "1":
            return "\n".join(doc["test_list"])
        return "\n" + doc["test"]

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        dataset = self.dataset["test"]
        return dataset

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        results, _ = compute_code_eval(
            references=references,
            predictions=generations,
            timeout=10.0,  # 10s timeout
        )
        return results
