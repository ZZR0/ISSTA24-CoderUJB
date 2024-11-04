"""
DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation

https://arxiv.org/pdf/2211.11501.pdf

DS-1000 is a code generation benchmark with a thousand data science questions spanning seven Python libraries that (1) reflects diverse, realistic, and practical use cases, (2) has a reliable metric, (3) defends against memorization by perturbing questions.

Homepage: https://ds1000-code-gen.github.io/
"""

import concurrent.futures as cfuts
import os 
import pandas as pd
from tqdm import tqdm
from code_ujb.tasks.custom_metrics import execution

from code_ujb.Task import Task

_CITATION = """
@article{Lai2022DS1000,
  title={DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation},
  author={Yuhang Lai and Chengxi Li and Yiming Wang and Tianyi Zhang and Ruiqi Zhong and Luke Zettlemoyer and Scott Wen-tau Yih and Daniel Fried and Sida Wang and Tao Yu},
  journal={ArXiv},
  year={2022},
  volume={abs/2211.11501}
}
"""


def create_all_tasks():
    def create_task(key, mode):
        class DS1000(GeneralDS1000):
            def __init__(self, **kwargs):
                super().__init__(key, mode, **kwargs)

        return DS1000

    return {
        f"ds1000-{key.lower()}-{mode.lower()}": create_task(key, mode)
        for key in [
            "All",
            "Numpy",
            "Pandas",
            "Scipy",
            "Matplotlib",
            "Sklearn",
            "Tensorflow",
            "Pytorch",
        ]
        for mode in ["Completion"]
    }


class GeneralDS1000(Task):
    DATASET_PATH = "xlangai/DS-1000"
    DATASET_NAME = None

    def __init__(self, key, mode):
        super().__init__(
            stop_words=["</code>", "# SOLUTION END"], requires_execution=True
        )
        self._key = key
        self._mode = mode
        self.dataset = [item for item in self.dataset["test"]]
        for t_idx, item in enumerate(self.dataset):
            item["task_idx"] = t_idx
        if self._key != "All":
            self.dataset = [item for item in self.dataset if item["metadata"]["library"] == self._key]

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset

    def get_prompt_complete(self, doc):
        """
        Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str | dict[str: str]
        """
        if self._mode == "Completion":
            return doc["prompt"]
        else:
            raise ValueError(f"Invalid mode: {self._mode}")

    def get_prompt_chat(self, doc):
        raise NotImplementedError()
    
    def get_reference(self, doc):
        """
        Builds the reference solution for the doc (sample from the test dataset).
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        return doc
        # return doc["reference_code"]

    def postprocess_generation_complete(self, generation, idx):
        """
        Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int (if needed)
            index of doc in the dataset to which the generation belongs
        :return: str
        """
        if self._mode == "Completion":
            for start in ["BEGIN SOLUTION\n<code>", "# SOLUTION START"]:
                try:
                    generation = generation.split(start, 1)[-1]
                except IndexError:
                    pass
        for stop in self.stop_words:
            generation = generation.split(stop)[0]
        return generation.strip()

    def postprocess_generation_chat(self, generation, idx):
        raise NotImplementedError()
    
    def process_results(self, generations, references):
        """
        Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations as in {"metric_name": result}.
        We encourage to directly load the metric from `evaluate` library to keep the code concise.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        :return: dict[str: float]
        """
        # disable tensorflow logging and no GPU
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        tasks = []
        for gens, reference in zip(generations, references):
            for g_idx, gen in enumerate(gens):
                tasks.append((g_idx, gen, reference))
        ds1000_results = []
        ds1000 = self.get_dataset()
        
        # for (g_idx, gen, ref) in tqdm(tasks):
        #     id = int(ref['metadata']['problem_id'])
        #     lib = ref['metadata']['library']
        #     test_program = (
        #         ref['code_context'] + '\n'
        #         + f'code = {repr(gen)}\n'
        #         + 'test_execution(code)\n'
        #         + ('test_string(code)\n'  if 'test_string(' in ref['code_context']  else '\n')
        #     )
        #     # you have some options on how to actually execute the program here.
        #     result = execution.check_correctness(test_program, 
        #                                 timeout=30, completion_id=id, task_idx=ref["task_idx"], g_idx=g_idx)

        #     cid = result['completion_id']
        #     result['score'] = 1 if result['passed'] else 0
        #     result['library'] = ds1000[cid]['metadata']['library']
        #     result['perturbation_type'] = ds1000[cid]['metadata']['perturbation_type']
        #     ds1000_results.append(result)
        
        with cfuts.ProcessPoolExecutor(
            max_workers=16
        ) as executor:
            futs = []
            for (g_idx, gen, ref) in tasks:
                id = int(ref['metadata']['problem_id'])
                lib = ref['metadata']['library']
                test_program = (
                    ref['code_context'] + '\n'
                    + f'code = {repr(gen)}\n'
                    + 'test_execution(code)\n'
                    + ('test_string(code)\n'  if 'test_string(' in ref['code_context']  else '\n')
                )
                # you have some options on how to actually execute the program here.
                futs.append(executor.submit(execution.check_correctness, test_program, 
                                            timeout=120, completion_id=id, task_idx=ref["task_idx"], g_idx=g_idx))

            for f in tqdm(cfuts.as_completed(futs), total=len(futs)):
                result = f.result()
                cid = result['completion_id']
                result['score'] = 1 if result['passed'] else 0
                result['library'] = ds1000[cid]['metadata']['library']
                result['perturbation_type'] = ds1000[cid]['metadata']['perturbation_type']
                ds1000_results.append(result)

        df_res = pd.DataFrame.from_records(ds1000_results)
        pd.set_option('display.precision', 3)
        results = {}
        tmp = df_res.agg({'score': ['count', 'mean']}).to_dict()
        results["All_Count"] = tmp["score"]["count"]
        results["All_Accuracy"] = tmp["score"]["mean"]
        tmp = df_res[['library', 'score']].groupby('library').agg({'score': ['count', 'mean']}).to_dict()
        for lib in list(tmp.values())[0]:
            results[f"Library_{lib}_Count"] = tmp[('score', 'count')][lib]
            results[f"Library_{lib}_Accuracy"] = tmp[('score', 'mean')][lib]
        
        tmp = df_res[['perturbation_type', 'score']].groupby('perturbation_type').agg({'score': ['count', 'mean']}).to_dict()
        for lib in list(tmp.values())[0]:
            results[f"Perturbation_Type_{lib}_Count"] = tmp[('score', 'count')][lib]
            results[f"Perturbation_Type_{lib}_Accuracy"] = tmp[('score', 'mean')][lib]
            
        results["detail"] = ds1000_results
        return results
