"""Mapping Language to Code in Programmatic Context (Concode)
https://arxiv.org/abs/1808.09588

CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation
https://arxiv.org/abs/2102.04664

Java code generation in CodeXGLUE text-to-code dataset (built from Concode dataset)
Available at https://huggingface.co/datasets/code_x_glue_ct_code_to_text
2000 samples are available in the test set.

Here we use two-shot evaluation (the original paper evaluates finetuned models)
"""
import json

from evaluate import load

from code_ujb.Task import Task

_CITATION = """
@article{iyer2018mapping,
  title={Mapping language to code in programmatic context},
  author={Iyer, Srinivasan and Konstas, Ioannis and Cheung, Alvin and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:1808.09588},
  year={2018}
}
"""


class Spider(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "NESPED-GEN/spider"

    def __init__(self, max_order=4, smooth=True):
        super().__init__(
            stop_words=["\n"],
            requires_execution=False,
        )
        self.max_order = max_order
        self.smooth = smooth

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        # test split of the dataset doesn't have targets
        return self.dataset["dev"]

    def fewshot_examples(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        examples = {
            "context1": "CREATE TABLE department (Department_ID number, Name text, Creation text, Ranking number, Budget_in_Billions number, Num_Employees number); CREATE TABLE head (head_ID number, name text, born_state text, age number); CREATE TABLE management (department_ID number, head_ID number, temporary_acting text)",
            "question1": "How many heads of the departments are older than 56 ?", 
            "answer1": "SELECT count(*) FROM head WHERE age > 56",
            "context2": "CREATE TABLE department (Department_ID number, Name text, Creation text, Ranking number, Budget_in_Billions number, Num_Employees number); CREATE TABLE head (head_ID number, name text, born_state text, age number); CREATE TABLE management (department_ID number, head_ID number, temporary_acting text)", 
            "question2": "List the name, born state and age of the heads of departments ordered by age.", 
            "answer2": "SELECT name, born_state, age FROM head ORDER BY age"
        }
        return examples

    @staticmethod
    def two_shot_prompt(entry, question, context, examples):
        """Two shot prompt format as instructions & solutions"""
        prompt = f"\n# USER_QUESTION:\n{examples['question1']}\
                   \n# TABLE SCHEMA:\n{examples['context1']}\
                   \n# SQL_QUERY:\n{examples['answer1']}\
                   \n# USER_QUESTION:\n{examples['question2']}\
                   \n# TABLE SCHEMA:\n{examples['context2']}\
                   \n# SQL_QUERY:\n{examples['answer2']}\
                   \n# USER_QUESTION:\n{question}\
                   \n# TABLE SCHEMA:\n{context}\
                   \n# SQL_QUERY:\n"
        assert (
            prompt.count("# SQL_QUERY:\n") == 3
        ), "Splitting operation in postprocess_generation is invalid"
        return entry + prompt

    def get_prompt_complete(self, doc):
        """Builds the prompt for the LM to generate from."""
        examples = self.fewshot_examples()
        question = doc["question_en"]
        context = doc["schema"]
        entry = "Please give me the right SQL query based on following instructions:\n"
        prompt = self.two_shot_prompt(entry, question, context, examples)
        return prompt

    def get_prompt_chat(self, doc):
        raise NotImplementedError()

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return doc["query"].lower()

    def postprocess_generation_complete(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for this task)
        """
        output = generation.split("# SQL_QUERY:\n", 3)[-1].strip().split("# USER_QUESTION:")[0].strip()
        return output.lower()

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
        bleu = load("bleu")
        gens = [gen[0] for gen in generations]
        results = bleu.compute(
            references=references, predictions=gens, max_order=self.max_order, smooth=self.smooth
        )
        return results
