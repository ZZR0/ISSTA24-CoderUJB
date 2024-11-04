"""
DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation

https://arxiv.org/pdf/2211.11501.pdf

DS-1000 is a code generation benchmark with a thousand data science questions spanning seven Python libraries that (1) reflects diverse, realistic, and practical use cases, (2) has a reliable metric, (3) defends against memorization by perturbing questions.

Homepage: https://ds1000-code-gen.github.io/
"""

import json
import os
import pathlib
import random
import sqlite3

import multiprocessing as mp
import sys

from tqdm import tqdm
from func_timeout import func_timeout, FunctionTimedOut

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
    def create_task(mode):
        class BirdSQL(GeneralBirdSQL):
            def __init__(self, **kwargs):
                super().__init__(mode, **kwargs)

        return BirdSQL

    return {
        f"birdsql-{mode.lower()}": create_task(mode)
        for mode in [
            "dev",
            # "test",
        ]
    }

def decouple_question_schema(datasets, db_root_path):
    question_list = []
    db_path_list = []
    for i, data in enumerate(datasets):
        question_list.append(data)
        cur_db_path = os.path.join(db_root_path, data['db_id'], data['db_id'] +'.sqlite')
        db_path_list.append(cur_db_path)
    
    return question_list, db_path_list

def nice_look_table(column_names: list, values: list):
    rows = []
    # Determine the maximum width of each column
    widths = [max(len(str(value[i])) for value in values + [column_names]) for i in range(len(column_names))]

    # Print the column names
    header = ''.join(f'{column.rjust(width)} ' for column, width in zip(column_names, widths))
    # print(header)
    # Print the values
    for value in values:
        row = ''.join(f'{str(v).rjust(width)} ' for v, width in zip(value, widths))
        rows.append(row)
    rows = "\n".join(rows)
    final_output = header + '\n' + rows
    return final_output

def generate_schema_prompt(db_path, num_rows=None):
    # extract create ddls
    '''
    :param root_place:
    :param db_name:
    :return:
    '''
    full_schema_prompt_list = []
    conn = sqlite3.connect(db_path)
    # Create a cursor object
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    schemas = {}
    for table in tables:
        if table == 'sqlite_sequence':
            continue
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='{}';".format(table[0]))
        create_prompt = cursor.fetchone()[0]
        schemas[table[0]] = create_prompt
        if num_rows:
            cur_table = table[0]
            if cur_table in ['order', 'by', 'group']:
                cur_table = "`{}`".format(cur_table)

            cursor.execute("SELECT * FROM {} LIMIT {}".format(cur_table, num_rows))
            column_names = [description[0] for description in cursor.description]
            values = cursor.fetchall()
            rows_prompt = nice_look_table(column_names=column_names, values=values)
            verbose_prompt = "/* \n {} example rows: \n SELECT * FROM {} LIMIT {}; \n {} \n */".format(num_rows, cur_table, num_rows, rows_prompt)
            schemas[table[0]] = "{} \n {}".format(create_prompt, verbose_prompt)

    for k, v in schemas.items():
        full_schema_prompt_list.append(v)

    schema_prompt = "\n\n".join(full_schema_prompt_list)

    return schema_prompt

def generate_comment_prompt(question, knowledge=None):
    pattern_prompt_no_kg = "-- Using valid SQLite, answer the following questions for the tables provided above."
    pattern_prompt_kg = "-- Using valid SQLite and understading External Knowledge, answer the following questions for the tables provided above."
    # question_prompt = "-- {}".format(question) + '\n SELECT '
    question_prompt = "-- {}".format(question)
    knowledge_prompt = "-- External Knowledge: {}".format(knowledge)

    if not knowledge_prompt:
        result_prompt = pattern_prompt_no_kg + '\n' + question_prompt
    else:
        result_prompt = knowledge_prompt + '\n' + pattern_prompt_kg + '\n' + question_prompt

    return result_prompt

def generate_combined_prompts_one(db_path, question):
    question, knowledge = question["question"], question["evidence"]
    schema_prompt = generate_schema_prompt(db_path, num_rows=None) # This is the entry to collect values
    comment_prompt = generate_comment_prompt(question, knowledge)

    combined_prompts = schema_prompt + '\n\n' + comment_prompt + '\nSELECT '

    return combined_prompts

def execute_sql(predicted_sql,ground_truth, db_path):
    conn = sqlite3.connect(db_path)
    # Connect to the database
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    res = 0
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    return res

def execute_model(tasks, db_path, meta_time_out, exec_result):
    for (g_idx, predicted_sql, ref) in tqdm(tasks):
        ground_truth = ref["question"]["SQL"]
        db_id = ref["question"]["db_id"]
        db_place = os.path.join(db_path, db_id, db_id +'.sqlite')
        try:
            res = func_timeout(meta_time_out, execute_sql,
                                    args=(predicted_sql, ground_truth, db_place))
        except KeyboardInterrupt:
            sys.exit(0)
        except FunctionTimedOut:
            result = [(f'timeout',)]
            res = 0
        except Exception as e:
            result = [(f'error',)]  # possibly len(query) > 512 or not executable
            res = 0
        # print(result)
        # result = str(set([ret[0] for ret in result]))
        result = {'task_idx': ref["task_idx"], 'g_idx': g_idx, 'res': res, 'difficulty': ref["question"]["difficulty"]}
        # print(result)
        exec_result.put(result)

def run_sqls_parallel(generations, references, db_path, num_cpus=1, meta_time_out=30.0):
    result_queue = mp.Manager().Queue()
    tasks = []
    for gens, ref in zip(generations, references):
        for g_idx, gen in enumerate(gens):
            tasks.append((g_idx, gen, ref))
    random.shuffle(tasks)
    num_cpus = min(num_cpus, len(tasks))
    task_count = len(tasks) // num_cpus
    processes = list()
    for i in range(num_cpus):
        p = mp.Process(target=execute_model, args=(tasks[i*task_count:(i+1)*task_count], db_path, meta_time_out, result_queue))
        processes.append(p)
        p.start()
    if len(tasks) % num_cpus != 0:
        p = mp.Process(target=execute_model, args=(tasks[num_cpus*task_count:], db_path, meta_time_out, result_queue))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
        
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    return results

def compute_acc_by_diff(exec_results):
    num_queries = len(exec_results)
    results = [res['res'] for res in exec_results]
    simple_results, moderate_results, challenging_results = [], [], []

    for res in exec_results:
        if res['difficulty'] == 'simple':
            simple_results.append(res)
        if res['difficulty'] == 'moderate':
            moderate_results.append(res)
        if res['difficulty'] == 'challenging':
            challenging_results.append(res)

    simple_acc = sum([res['res'] for res in simple_results])/len(simple_results) if len(simple_results) > 0 else 0
    moderate_acc = sum([res['res'] for res in moderate_results])/len(moderate_results) if len(moderate_results) > 0 else 0
    challenging_acc = sum([res['res'] for res in challenging_results])/len(challenging_results) if len(challenging_results) > 0 else 0
    all_acc = sum(results)/num_queries
    count_lists = [len(simple_results), len(moderate_results), len(challenging_results), num_queries]
    return simple_acc * 100, moderate_acc * 100, challenging_acc * 100, all_acc * 100, count_lists

class GeneralBirdSQL(Task):
    DATASET_PATH = None
    DATASET_NAME = None

    def __init__(self, mode):
        super().__init__(
            stop_words=[], requires_execution=True
        )
        self._mode = mode
        assert mode in ["dev"], f"Bird-SQL '{mode}' is not a valid mode."
        self._dir = pathlib.Path(__file__).parent / "bird_data"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._data = self._dir / f"{mode}_databases"
        self._download_dataset()
        self.dataset = None
        self._build_dataset()

    def _download_dataset(self):
        url = "https://bird-bench.github.io/"
        if not self._data.exists():
            raise ValueError(
                f"Please download the dataset from {url} first."
            )
    
    def _build_dataset(self):
        eval_data = json.load(open(self._dir / f"{self._mode}.json"))
        question_list, db_path_list = decouple_question_schema(datasets=eval_data, db_root_path=self._data)
        assert len(question_list) == len(db_path_list)
        results = []
        for i, question in enumerate(question_list):
            # print('the question is: {}'.format(question))
            cur_prompt = generate_combined_prompts_one(db_path=db_path_list[i], question=question)
            # print('the prompt is: {}'.format(prmpt))
            results.append({
                "task_idx": i,
                "question": question,
                "prompt": cur_prompt,
            })

        self.dataset = results

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset
    
    def get_prompt_chat(self, doc):
        raise NotImplementedError("Chat mode is not implemented for BirdSQL")
    
    def get_prompt_complete(self, doc):
        """
        Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str | dict[str: str]
        """
        return doc["prompt"]

    def get_reference(self, doc):
        """
        Builds the reference solution for the doc (sample from the test dataset).
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        return doc

    def postprocess_generation_chat(self, generation, idx):
        raise NotImplementedError("Chat mode is not implemented for BirdSQL")
    
    def postprocess_generation_complete(self, generation, idx):
        """
        Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int (if needed)
            index of doc in the dataset to which the generation belongs
        :return: str
        """
        prompt = self.get_prompt(self.get_dataset()[idx])
        prompt_lines = prompt.splitlines()
        sub_prompt = None
        for i in range(1, len(prompt_lines)):
            sub_prompt = "\n".join(prompt_lines[-i:])
            if generation.count(sub_prompt) == 1:
                break
        
        output = "SELECT " + generation.split(sub_prompt)[-1].strip()
        stop = ['--', '\n\n', ';', '#']
        for s in stop:
            if s in output:
                output = output.split(s)[0] + s
        
        # db_id = self.get_dataset()[idx]["question"]["db_id"]
        # output = output + '\t----- bird -----\t' + db_id # to avoid unpredicted \t appearing in codex results
        
        return output

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
        assert len(generations) == len(references), "Number of generations and references must be equal."

        exec_result = run_sqls_parallel(generations, references, db_path=self._data, num_cpus=os.cpu_count()//2, meta_time_out=60)
        exec_result = sorted(exec_result, key=lambda x: x['task_idx'])
        
        simple_acc, moderate_acc, challenging_acc, acc, count_lists = compute_acc_by_diff(exec_result)

        results = {
            "accuracy": acc,
            "simple_acc": simple_acc,
            "moderate_acc": moderate_acc,
            "challenging_acc": challenging_acc,
            "count_lists": count_lists,
            "detail": exec_result
        }
        # print(results)
        return results