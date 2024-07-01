# CoderUJB Dataset Construction Process

This tutorial will guide you through the process of constructing the CoderUJB dataset from defects4j from scratch. Before starting, ensure that you have defects4j installed and running successfully. Please note that the construction process for the CoderUJB dataset involves extensive file read/write operations, so we strongly recommend running the subsequent code on an SSD.

## Construction Process

### Navigate to the Project Working Directory
```bash
cd ISSTA24-CoderUJB
```

### Extract Data for Each Version of Projects from defects4j
```bash
python datasets/extract_defects4j_info.py
```

### Extract Test Coverage Relationships from defects4j Projects
```bash
# This step will take a significant amount of time (>10 hours) and will spawn many processes simultaneously.
# Refer to the following link to ensure your environment supports a large number of concurrent processes:
# https://stackoverflow.com/questions/32283003/python-cant-start-new-thread-100-active-threads
python datasets/extract_function_test_map.py
```

### Construct the CoderUJB-FCG (Functional Code Generation) Dataset
```bash
# The CoderUJB-FCG dataset will be stored in 'ISSTA24-CoderUJB/datasets/data/task_complete_bench_default|2048.json'
python datasets/extract_task_complete.py
```

Below is an example of a CoderUJB-FCG task. Examples for other tasks are similar.
```json
{
    "task_id": "Unique identifier for the example task",
    "project": "Project name from which the example task is derived", 
    "bug_id": "Bug ID from which the example task is derived", 
    "testmethods": "Test cases related to the example task", 
    "source_dir": "Source code directory of the example task's project", 
    "location": "Source code file path of the example task",
    "start": "Start line of the example task's function", 
    "end": "End line of the example task's function", 
    "function": "Source code of the example task's function", 
    "comment": "Comments for the example task's function", 
    "function_name": "Name of the example task's function",
    "prompt_chat": "Prompt for conversational models", 
    "prompt_chat_with_comment": "Prompt containing context and function comments but excluding function signature", 
    "prompt_complete": "Prompt for the base model", 
    "prompt_complete_with_comment": "Prompt containing context and function comments but excluding function signature",
    "function_signature": "Signature of the example task's function", 
    "import_context": "Import context of the example task",
    "class_signature": "Signature of the class containing the example task",
    "class_field_context": "Field context of the class containing the example task",
    "class_function_signature_context": "Function signature context of the class containing the example task",
    "code_context": "Merged context of the example task",
    "source": "Source code of the file containing the example task", 
    "indent": "Indentation unit of the example task's function",
    "function_tested_rate": "Test coverage rate of the example task's function",
}
```

### Construct the CoderUJB-CTG (Code-based Test Generation) Dataset
```bash
# The CoderUJB-CTG dataset will be stored in 'ISSTA24-CoderUJB/datasets/data/task_testgen_bench_default|2048.json'
python datasets/extract_task_testgen.py
```

### Construct the CoderUJB-APR (Automated Program Repair) Dataset
```bash
# The CoderUJB-APR dataset will be stored in 'ISSTA24-CoderUJB/datasets/data/task_repair_bench_default|2048.json'
python datasets/extract_task_repair.py
```

### Construct the CoderUJB-DD (Defect Detection) Dataset
```bash
# The CoderUJB-DD dataset will be stored in 'ISSTA24-CoderUJB/datasets/data/task_defectdetection_bench_default|2048.json'
python datasets/extract_task_defectdetection.py
```

### Construct the CoderUJB-ITG (Issue-based Test Generation) Dataset
```bash
# The CoderUJB-ITG dataset will be stored in 'ISSTA24-CoderUJB/datasets/data/task_testgenissue_bench_default|2048.json'
python datasets/extract_task_testgenissue.py
```

### Upload the Data to Hugging Face
```bash
# Make sure to modify the YOUR_HF_ID variable in datasets/upload_datasets.py to ensure the data is uploaded correctly.
# Refer to the tutorial at https://huggingface.co/docs/datasets/upload_dataset for more detailed instructions.
python datasets/upload_datasets.py
```

### Update Dataset Names in code_ujb for Each Task to Use the Newly Constructed Dataset
Modify the `DATASET_PATH` in `code_ujb/tasks/code_ujb_complete.py` to the new Hugging Face dataset name:
```python
class CodeUJBComplete(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings, and evaluation methods.
    """
    DATASET_PATH = "YOUR_HF_ID/code_ujb_complete"

    def __init__(self):
        super().__init__(
            stop_words=["/**", "/**\n", "public", "private", "protected",  
                        "\t/**", "\t/**\n", "\tpublic", "\tprivate", "\tprotected"],
            requires_execution=False,
        )
        print("Using Dataset:", self.DATASET_PATH)
        self.dataset = load_dataset(self.DATASET_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
```
