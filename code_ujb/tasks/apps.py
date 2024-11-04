"""Measuring Coding Challenge Competence With APPS
https://arxiv.org/abs/2105.09938

APPS is a benchmark for code generation with 10000 problems. With three difficulty levels: introductory, interview and competition.
It can be used to evaluate the ability of language models to generate code from natural language specifications.

Homepage: https://github.com/hendrycks/apps
"""

import json
import os
import multiprocessing as mp
import random
import re
from code_ujb.Task import Task
from code_ujb.tasks.custom_metrics.apps_metric.utils import compute_metrics, get_results

_CITATION = """
@article{hendrycksapps2021,
  title={Measuring Coding Challenge Competence With APPS},
  author={Dan Hendrycks and Steven Basart and Saurav Kadavath and Mantas Mazeika and Akul Arora and Ethan Guo and Collin Burns and Samir Puranik and Horace He and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}
"""


LEVELS = ["introductory", "interview", "competition"]


def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {apps-interview: Task, apps-competitoon: Task}
    """
    return {f"apps-{level}": create_task(level) for level in LEVELS}


def create_task(level):
    class APPS(GeneralAPPS):
        def __init__(self, **kwargs):
            super().__init__(level, **kwargs)

    return APPS


class GeneralAPPS(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "codeparrot/apps"
    DATASET_NAME = None

    def __init__(self, level, k_list=[1, 2, 5, 10, 20, 100]):
        self.DATASET_NAME = level
        super().__init__(
            # stop_words=["\nQUESTION", "\n---", "\nANSWER"],
            requires_execution=True,
        )
        self.k_list = k_list

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]
    
    def get_prompt_complete(self, doc):
        """Generate prompts for APPS
        Finetuning setup: prompt=question  with some starter code and function name if they exist.
        We also specify the type of the prompt, i.e. whether it is call-based or standard input-based.
        """
        starter_code = None if len(doc["starter_code"]) == 0 else doc["starter_code"]
        try:
            input_outpout = json.loads(doc["input_output"])
            fn_name = (
                None if not input_outpout.get("fn_name") else input_outpout["fn_name"]
            )
        except ValueError:
            fn_name = None
        prompt = "\nQUESTION:\n"
        prompt += doc["question"]
        if not fn_name:
            call_format = "\nUse Standard Input format"
            prompt += call_format
        else:
            call_format = "\nUse Call-Based format"
            prompt += call_format
        prompt += "\nANSWER:\n"
        if starter_code is not None:
            prompt += starter_code
        else:
            prompt += "\ndef solve():\n    "
        return prompt

    def get_prompt_chat(self, doc):
        """Generate prompts for APPS
        Chat setup: prompt=question  with some starter code and function name if they exist.
        """
        raise NotImplementedError("Chat mode is not implemented for APPS")
    
    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return doc

    def extract_code_snippet(self, content):
        """ Extract code snippet from the content by the last pattern ```<code>```. """
        code_blocks = re.findall(r'```python(.*?)```', content, re.DOTALL)
        return code_blocks[-1].strip() if code_blocks else None
    
    def postprocess_generation_complete(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for APPS)
        """
        try:
            # generation = generation.split("\nANSWER:", 1)[1]
            if "```python" in generation:
                generation = self.extract_code_snippet(generation)
            else:
                generation = generation
        except:
            # happens when prompts were very long and got truncated
            pass
        return generation
    
    def postprocess_generation_chat(self, generation, idx):
        raise NotImplementedError("Chat mode is not implemented for APPS")

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences (not needed for APPS Task)
        """
        result_queue = mp.Manager().Queue()
        tasks = []
        for gens, ref in zip(generations, references):
            ref['input_output'] = eval(ref['input_output'])
            # tasks.append((json.loads(ref['solutions'])[:20], ref))
            tasks.append((gens, json.dumps(ref)))
        # random.shuffle(tasks)
        num_cpus = os.cpu_count()//2
        num_cpus = min(num_cpus, len(tasks))
        task_count = len(tasks) // num_cpus
        processes = list()
        for i in range(num_cpus):
            p = mp.Process(target=compute_metrics, kwargs={"generations":tasks[i*task_count:(i+1)*task_count], 
                                                           "level":self.DATASET_NAME, "debug":False,
                                                           "result_queue":result_queue})
            processes.append(p)
            p.start()
        if len(tasks) % num_cpus != 0:
            p = mp.Process(target=compute_metrics, kwargs={"generations":tasks[num_cpus*task_count:], 
                                                           "level":self.DATASET_NAME, "debug":False,
                                                           "result_queue":result_queue})
            processes.append(p)
            p.start()
        
        for p in processes:
            p.join()
            
        results = {}
        while not result_queue.empty():
            results.update(result_queue.get())
        results = get_results(results, count_errors=True, k_list=self.k_list)
        print(results)
        
        return results
        
