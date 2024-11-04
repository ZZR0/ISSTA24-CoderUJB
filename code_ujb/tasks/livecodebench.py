
import json
import pickle
import re
import zlib
import base64
from datasets import load_dataset

from code_ujb.Task import Task
from code_ujb.tasks.custom_metrics.livecodebench_eval import evaluate_generations, compute_metrics_from_results

def create_all_tasks():
    code_generation = {f"lcb-cg-{version}": create_task("cg", version) for version in ["v1", "v2", "v3", "v4", "latest"]}
    return {**code_generation}

def create_task(task_name, version):
    if task_name == "cg": 
        class LiveCodeBench_CG(LiveCodeBench_CG_Base):
            def __init__(self):
                super().__init__(version)
        return LiveCodeBench_CG
    else: raise KeyError(f"Task {task_name} not found")

class PromptConstants:
    SYSTEM_MESSAGE_GENERIC = f"You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program."

    SYSTEM_MESSAGE_GEMINI = f"You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program. Do NOT use system calls like `exit` in the generated program."

    SYSTEM_MESSAGE_DEEPSEEK = f"You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you answer questions related to computer science."

    SYSTEM_MESSAGE_MAGIC = f"You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.\n\n@@ Instruction\n"

    SYSTEM_MESSAGE_WIZARD = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

    SYSTEM_MESSAGE_PHIND = f"""You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program. Put your fixed program within code delimiters, for example: 
```python 
# YOUR CODE HERE
```"""

    SYSTEM_MESSAGE_CODEQWEN = (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user"
    )

    FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."

    FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows."


class LiveCodeBench_CG_Base(Task):
    DATASET_PATH: str = "livecodebench/code_generation_lite"
    DATASET_NAME: str = None
    
    def __init__(self, version):
        super().__init__()
        self.version = version
        print("Using Dataset:", self.DATASET_PATH)
        self.dataset = load_dataset(self.DATASET_PATH, version_tag=f"release_{version}")
        self.dataset = [item for item in self.dataset["test"]]
        
    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        
        return self.dataset[:20]

    def get_prompt_complete(self, doc):
        """Builds the prompt for the LM to generate from."""
        raise NotImplementedError()
    
    def get_prompt_chat(self, doc):
        prompt = f"### Question:\n{doc['question_content'].strip()}\n\n"
        if "starter_code" in doc:
            prompt += (
                f"### Format: {PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
            )
            prompt += f"```python\n{doc['starter_code']}\n```\n\n"
        else:
            prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
            prompt += "```python\n# YOUR CODE HERE\n```\n\n"
        prompt += f"### Answer: (use the provided format with backticks)\n\n"
        return prompt
    
    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return doc
    
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
        
        all_code_blocks = []
        for pattern in patterns:
            code_block = extract_code_block(generation, pattern)
            if code_block is not None:
                all_code_blocks.append(code_block)
        if len(all_code_blocks) == 0:
            return generation if generation is not None else "$ERROR$"
        all_code_blocks.sort(key=lambda x: len(x), reverse=True)
        return all_code_blocks[0]
    

    def process_results(self, generations, references,
        k_list=[1, 5, 10, 20, 40, 50, 75, 100, 125, 150, 200, 500, 1000],
        num_process_evaluate=16,
        timeout=6,
        debug=False,
    ):
        def get_evaluation_sample(instance):
            public_test_cases = []
            private_test_cases = []
            try:
                public_test_cases = json.loads(instance["public_test_cases"])
            except:
                pass
            try:
                private_test_cases = json.loads(instance["private_test_cases"])  # type: ignore
            except:
                private_test_cases = json.loads(
                    pickle.loads(
                        zlib.decompress(
                            base64.b64decode(instance["private_test_cases"].encode("utf-8"))  # type: ignore
                        )
                    )
                )
                
            if len(public_test_cases) == 0 and len(private_test_cases) == 0:
                raise ValueError(f"No test cases found for instance {instance}, {type(instance)}")
            
            return {
                "input_output": json.dumps(
                    {
                        "inputs": [
                            t["input"]
                            for t in public_test_cases + private_test_cases
                        ],
                        "outputs": [
                            t["output"]
                            for t in public_test_cases + private_test_cases
                        ],
                        "fn_name": json.loads(instance["metadata"]).get("func_name", None),
                    }
                ),
            }

        samples_linear = []
        generations_linear = []
        remap_index = []
        results = {}
        metadatas = {}
        for idx, (sample, generation_list) in enumerate(
            zip(references, generations)
        ):
            assert isinstance(generation_list, list), generations[0]
            sample = get_evaluation_sample(sample)
            results[idx] = []
            metadatas[idx] = []
            for generation in generation_list:
                assert isinstance(generation, str), generation_list[0]
                samples_linear.append(sample)
                generations_linear.append([generation])
                remap_index.append(idx)

        print(f"Evaluating {len(samples_linear)}...")

        results_linear, metadatas_linear = evaluate_generations(
            samples_linear,
            generations_linear,
            debug=debug,
            num_process_evaluate=num_process_evaluate,
            timeout=timeout,
        )

        for idx, sub_results in sorted(results_linear.items(), key=lambda x: x[0]):
            results[remap_index[idx]].append(sub_results[0])

        for idx, sub_metadatas in sorted(metadatas_linear.items(), key=lambda x: x[0]):
            metadatas[remap_index[idx]].append(sub_metadatas[0])

        metrics = compute_metrics_from_results(results, k_list=k_list)

        final_metadata = []
        for key in sorted(list(metadatas.keys())):
            final_metadata.append(metadatas[key])
        for i in range(len(final_metadata)):
            if type(final_metadata[i]) is not list:
                final_metadata[i] = [json.dumps(final_metadata[i])]
            else:
                final_metadata[i] = [json.dumps(x) for x in final_metadata[i]]

            assert len(final_metadata[i]) == len(
                generations[0]
            ), f"{len(final_metadata[i])=}"

        return {"metrics": metrics, "results": results, "final_metadata": final_metadata}
