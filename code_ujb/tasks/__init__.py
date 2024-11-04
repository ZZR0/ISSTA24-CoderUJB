import inspect
from pprint import pprint

from . import (
    code_ujb_repair, 
    code_ujb_complete, 
    code_ujb_testgen, 
    code_ujb_testgenissue,
    code_ujb_defectdetection,
    code_ujb_multiple_java,
    code_ujb_multiple_python,
    code_ujb_mbpp)
from . import (apps, codexglue_code_to_text, codexglue_text_to_text, conala,
               concode, ds1000, gsm, humaneval, humanevalplus, humanevalpack,
               instruct_humaneval, instruct_wizard_humaneval, mbpp, mbppplus,
               multiple, parity, python_bugs, quixbugs, recode, santacoder_fim,
               studenteval, mercury, crosscodeeval, birdsql, spider, livecodebench)

TASK_REGISTRY = {
    "codeujbrepair": code_ujb_repair.CodeUJBRepair,
    "codeujbcomplete": code_ujb_complete.CodeUJBComplete,
    "codeujbtestgen": code_ujb_testgen.CodeUJBTestGen,
    "codeujbtestgenissue": code_ujb_testgenissue.CodeUJBTestGenIssue,
    "codeujbdefectdetection": code_ujb_defectdetection.CodeUJBDefectDetection,
    "multiplejava": code_ujb_multiple_java.MultipleJava,
    "multiplepython": code_ujb_multiple_python.MultiplePython,
    "codeujbmbpp": code_ujb_mbpp.MBPP,
    **apps.create_all_tasks(),
    **codexglue_code_to_text.create_all_tasks(),
    **codexglue_text_to_text.create_all_tasks(),
    **multiple.create_all_tasks(),
    "codexglue_code_to_text-python-left": codexglue_code_to_text.LeftCodeToText,
    "conala": conala.Conala,
    "concode": concode.Concode,
    **ds1000.create_all_tasks(),
    **humaneval.create_all_tasks(),
    **humanevalplus.create_all_tasks(),
    **humanevalpack.create_all_tasks(),
    "mbpp": mbpp.MBPP,
    "mbppplus": mbppplus.MBPPPlus,
    "parity": parity.Parity,
    "python_bugs": python_bugs.PythonBugs,
    "quixbugs": quixbugs.QuixBugs,
    "instruct_wizard_humaneval": instruct_wizard_humaneval.HumanEvalWizardCoder,
    **gsm.create_all_tasks(),
    **instruct_humaneval.create_all_tasks(),
    **recode.create_all_tasks(),
    **santacoder_fim.create_all_tasks(),
    "studenteval": studenteval.StudentEval,
    "mercury": mercury.Mercury,
    **crosscodeeval.create_all_tasks(),
    **birdsql.create_all_tasks(),
    "spider": spider.Spider,
    **livecodebench.create_all_tasks(),
}

ALL_TASKS = sorted(list(TASK_REGISTRY))

def get_task(task_name, prompt=None, load_data_path=None):
    try:
        kwargs = {}
        if "prompt" in inspect.signature(TASK_REGISTRY[task_name]).parameters and prompt is not None:
            kwargs["prompt"] = prompt
        if "load_data_path" in inspect.signature(TASK_REGISTRY[task_name]).parameters and load_data_path is not None:
            kwargs["load_data_path"] = load_data_path
        return TASK_REGISTRY[task_name](**kwargs)
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
