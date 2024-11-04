"""Mapping Language to Code in Programmatic Context (Concode)
https://arxiv.org/abs/1808.09588

CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation
https://arxiv.org/abs/2102.04664

Java code generation in CodeXGLUE text-to-code dataset (built from Concode dataset)
Available at https://huggingface.co/datasets/code_x_glue_ct_code_to_text
2000 samples are available in the test set.

Here we use two-shot evaluation (the original paper evaluates finetuned models)
"""
from datasets import load_dataset

from code_ujb.Task import Task
from code_ujb.tasks.custom_metrics.crosscodeeval_metric.eval_metric import compute_metric_stmt

_CITATION = """
@article{iyer2018mapping,
  title={Mapping language to code in programmatic context},
  author={Iyer, Srinivasan and Konstas, Ioannis and Cheung, Alvin and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:1808.09588},
  year={2018}
}
"""

def create_all_tasks():
    def create_task(key, mode):
        class CrossCodeEval(GeneralCrossCodeEval):
            def __init__(self, **kwargs):
                super().__init__(key, mode, **kwargs)

        return CrossCodeEval

    return {
        f"crosscodeeval-{key.lower()}-{mode.lower()}": create_task(key, mode)
        for key in [
            "python",
            "java",
            "csharp",
            "typescript",
        ]
        for mode in ["base", "retrieval", "retrievalwref"]
    }


class GeneralCrossCodeEval(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = None

    def __init__(self, language, mode):
        self.language = language
        self.mode = mode
        self.DATASET_NAME = f"ZHENGRAN/cross_code_eval_{language}"
        # we need the dataset to get stop words for each language
        self.dataset = load_dataset(
            self.DATASET_NAME,
            trust_remote_code=True)
        stop_words = ["\n", "\n\n", "\n\n\n"]
        super().__init__(
            stop_words=stop_words,
            requires_execution=True,
        )
        self.dataset = [item for item in self.dataset["train"]]
        for t_idx, item in enumerate(self.dataset):
            item["task_idx"] = t_idx

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        # test split of the dataset doesn't have targets
        return self.dataset

    def get_prompt_complete(self, doc):
        """Builds the prompt for the LM to generate from."""
        if self.mode == "base":
            return doc['prompt']
        elif self.mode == "retrieval":
            context_list = doc['crossfile_context_retrieval']['list']
        elif self.mode == "retrievalwref":
            context_list = doc['crossfile_context_retrievalwref']['list']
        else:
            raise ValueError(f"Unknown mode {self.mode}")
        
        context = ""
        for c in context_list[::-1]:
            context += "# Here are some relevant code fragments from other files of the repo:\n"
            context += f"# the below code fragment can be found in:\n# {c['filename']}\n"
            context += "\n".join(['# '+line for line in c['retrieved_chunk'].splitlines()])+"\n\n"
        
        prompt = context + "\n" + doc['prompt']
        return prompt

    def get_prompt_chat(self, doc):
        raise NotImplementedError()
    
    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return doc

    def postprocess_generation_complete(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for this task)
        """
        prompt = self.get_prompt(self.get_dataset()[idx])
        prompt_lines = prompt.splitlines()
        sub_prompt = None
        for i in range(1, len(prompt_lines)):
            sub_prompt = "\n".join(prompt_lines[-i:])
            if generation.count(sub_prompt) == 1:
                break
        
        output = generation.split(sub_prompt)[-1].strip()
        # output = output.split("\n")[0].strip()
        return output

    def postprocess_generation_chat(self, generation, idx):
        raise NotImplementedError()
    
    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing references
        """
        results = compute_metric_stmt(self.language, generations, references)
        return results
