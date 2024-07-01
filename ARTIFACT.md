# Artifact Evaluation

This repository demonstrates how to evaluate the artifacts presented in the paper **CoderUJB: An Executable and Unified Java Benchmark for Practical Programming Scenarios**, published in ISSTA'24. Specifically, we introduce CoderUJB, a new benchmark designed to evaluate LLMs across diverse Java programming tasks that are executable and reflective of actual development scenarios, acknowledging Java’s prevalence in real-world software production. CoderUJB comprises 2,239 programming questions derived from 17 real open-source Java projects and spans five practical programming tasks. Our empirical study on this benchmark investigates the coding abilities of various open-source and closed-source LLMs, examining the effects of continued pre-training in specific programming languages and instruction fine-tuning on their performance.

Our original evaluation in the paper includes dataset construction, answer generation, and evaluation. However, these processes can be rather time-consuming. To facilitate the artifact evaluation of our paper, we provide two options:

**Quick Result Analysis:** This option provides result analysis scripts to generate the corresponding tables/figures in the paper directly from the cached results of our prior runs.

**Complete Evaluation:** This option provides a complete evaluation of our artifacts, including dataset construction, answer generation, and answer evaluation.

Due to the random nature of neural networks, users may obtain slightly different results when regenerating the answers. Please note that such variations are usually tolerable and do not typically conflict with the conclusions of the paper.

## Environment Preparation

All experiments in this project were conducted on a server with 8 NVIDIA A800 80GB PCIe GPUs, running Ubuntu 22.04, with 1024GB of RAM, 26TB SSD storage, and an Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz. We used Conda for Python environment configuration, with Python version 3.10.

## Quick Result Analysis

Since running the experiments completely can be extremely time-consuming, we provide result analysis scripts to generate the corresponding tables/figures in the paper directly from the cached results of our prior runs. Specifically, we prepared a cached result directory `log` to avoid the time-consuming re-execution of experiments using the following commands.

### 1. Install CoderUJB

```bash
# Create a new conda environment
conda create -n ujb python=3.10
conda activate ujb

# Clone and install CoderUJB
git clone https://github.com/ZZR0/ISSTA24-CoderUJB.git
cd ISSTA24-CoderUJB
pip install -r requirements.txt
pip install -e .
```

For more details on package versions, please refer to `requirements.txt`.

### 2. Download Cached Results

You can download the cached results from [Zenodo](https://doi.org/10.5281/zenodo.12608143). Extract the downloaded `log.tar.gz` and move the `log` folder to the `ISSTA24-CoderUJB` directory.

### 3. Running Analysis Script

By running the following command, you can find the tables and figures from the paper in the `log/_combine_result` folder:

```bash
python scripts/analyze_result.py
```

## Dataset Construction

1. Refer to the [defects4j](https://github.com/rjust/defects4j) repository to install the `defects4j` execution environment.

2. Please refer to the [CoderUJB Construction Readme](./datasets/README.md) to reconstruct the CoderUJB dataset.

## Generation and Evaluation

### 1. Download Models

```bash
cd ISSTA24-CoderUJB
huggingface-cli download --resume-download meta-llama/CodeLlama-7b-hf --local-dir ./models/codellama-7b
huggingface-cli download --resume-download meta-llama/CodeLlama-13b-hf --local-dir ./models/codellama-13b
huggingface-cli download --resume-download meta-llama/CodeLlama-34b-hf --local-dir ./models/codellama-34b
huggingface-cli download --resume-download bigcode/starcoderbase --local-dir ./models/starcoderbase-15b
huggingface-cli download --resume-download meta-llama/CodeLlama-7b-Python-hf --local-dir ./models/codellama-python-7b
huggingface-cli download --resume-download meta-llama/CodeLlama-13b-Python-hf --local-dir ./models/codellama-python-13b
huggingface-cli download --resume-download meta-llama/CodeLlama-34b-Python-hf --local-dir ./models/codellama-python-34b
huggingface-cli download --resume-download bigcode/starcoder --local-dir ./models/starcoder-15b
huggingface-cli download --resume-download meta-llama/CodeLlama-7b-Instruct-hf --local-dir ./models/codellama-instruct-7b
huggingface-cli download --resume-download meta-llama/CodeLlama-13b-Instruct-hf --local-dir ./models/codellama-instruct-13b
huggingface-cli download --resume-download meta-llama/CodeLlama-34b-Instruct-hf --local-dir ./models/codellama-instruct-34b
huggingface-cli download --resume-download WizardLMTeam/WizardCoder-Python-7B-V1.0 --local-dir ./models/wizardcoder-python-7b
huggingface-cli download --resume-download WizardLMTeam/WizardCoder-Python-13B-V1.0 --local-dir ./models/wizardcoder-python-13b
huggingface-cli download --resume-download WizardLMTeam/WizardCoder-Python-34B-V1.0 --local-dir ./models/wizardcoder-python-34b
huggingface-cli download --resume-download WizardLMTeam/WizardCoder-15B-V1.0 --local-dir ./models/wizardcoder-15b
```

### 2. Set Dialogue Templates

Since some dialogue models do not have a default dialogue template set in the `tokenizer_config.json` file, we need to manually set these templates. We need to manually modify the `tokenizer_config.json` file for the following four models: `wizardcoder-python-7b`, `wizardcoder-python-13b`, `wizardcoder-python-34b`, and `wizardcoder-15b`. The `chat_template` below is applicable to all four models.

```bash
# Open the tokenizer_config.json file for the model
vim ./models/wizardcoder-python-7b/tokenizer_config.json
# Add the following key-value pair to the JSON file, ensuring correct JSON syntax.
"chat_template": "{% for message in messages %}{% if message['role'] == 'system' %}{% endif %}{% if message['role'] == 'user' %}{{ '### Instruction:\n' }}{% endif %}{% if message['role'] == 'assistant' %}{{ '### Response:\n' }}{% endif %}{{ message['content'].strip() }}{% if not loop.last %}{{ '\n\n' }}{% endif %}{% if message['role'] == 'user' and loop.last %}{{ '### Response:\n' }}{% endif %}{% endfor %}",
```

### 3. Generation

The following commands will help you quickly run the generation scripts, applying the specified model to a particular dataset to generate new answers. The `scripts/run_code_ujb.sh` script requires five input parameters. An example of its usage is shown below:

```bash
./scripts/run_code_ujb.sh <task> <request_type> <CoderUJB_task> <model_path> <model_save_id>
```

- **task**: Predefined tasks in the script, with options [`api_gen`, `tgi_gen`, `local_gen`, `eval`]. `api_gen` is used for OpenAI models like GPT-4, GPT-3.5-Turbo, etc. `tgi_gen` is using [Text-Generation-Inference](https://github.com/huggingface/text-generation-inference) for generation, and `local_gen` is using transformers package for gneration. `eval` is used for evaluating the pass@k of the generated answer.
    
- **request_type**: Specifies whether to use completion mode or chat mode to query the model, with options [`complete`, `chat`]. `complete` is used for base models like codellama-7b, and `chat` is used for instruction-tuned models like codellama-instruct-7b.
- **CoderUJB_task**: Tasks in CoderUJB, where `codeujbcomplete` represents `CoderUJB-FCG`, `codeujbtestgen` represents `CoderUJB-CTG`, `codeujbtestgenissue` represents `CoderUJB-ITG`, `codeujbrepair` represents `CoderUJB-APR`, and `codeujbdefectdetection` represents `CoderUJB-DD`.
- **model_path**: Path to the locally stored model, e.g., `./models/codellama-7b`, or the name of a remotely deployed LLM model like gpt-3.5-turbo.
- **model_save_id**: Folder name for saving the model results.

Below are examples of using the `run_code_ujb.sh` script to generate answers for CoderUJB tasks. After running the following commands, the generated answers will be saved in the `log/<model_save_id>/<CoderUJB_task>` folder.

```bash
./scripts/run_code_ujb.sh local_gen complete codeujbcomplete ./models/codellama-7b/ codellama-7b
./scripts/run_code_ujb.sh local_gen complete codeujbtestgen ./models/codellama-7b/ codellama-7b
./scripts/run_code_ujb.sh local_gen complete codeujbtestgenissue ./models/codellama-7b/ codellama-7b
./scripts/run_code_ujb.sh local_gen complete codeujbrepair ./models/codellama-7b/ codellama-7b
./scripts/run_code_ujb.sh local_gen complete codeujbdefectdetection ./models/codellama-7b/ codellama-7b

./scripts/run_code_ujb.sh local_gen chat codeujbcomplete ./models/codellama-instruct-7b/ codellama-instruct-7b
./scripts/run_code_ujb.sh local_gen chat codeujbtestgen ./models/codellama-instruct-7b/ codellama-instruct-7b
./scripts/run_code_ujb.sh local_gen chat codeujbtestgenissue ./models/codellama-instruct-7b/ codellama-instruct-7b
./scripts/run_code_ujb.sh local_gen chat codeujbrepair ./models/codellama-instruct-7b/ codellama-instruct-7b
./scripts/run_code_ujb.sh local_gen chat codeujbdefectdetection ./models/codellama-instruct-7b/ codellama-instruct-7b

# Set your OpenAI API key
# export OPENAI_API_BASE=''
# export OPENAI_API_KEY=''
./scripts/run_code_ujb.sh api_gen chat codeujbcomplete gpt-3.5-turbo gpt-3.5-turbo
./scripts/run_code_ujb.sh api_gen chat codeujbtestgen gpt-3.5-turbo gpt-3.5-turbo
./scripts/run_code_ujb.sh api_gen chat codeujbtestgenissue gpt-3.5-turbo gpt-3.5-turbo
./scripts/run_code_ujb.sh api_gen chat codeujbrepair gpt-3.5-turbo gpt-3.5-turbo
./scripts/run_code_ujb.sh api_gen chat codeujbdefectdetection gpt-3.5-turbo gpt-3.5-turbo
```

### 4. Evaluation

Below are the commands for evaluating CoderUJB. After running the following commands, the evaluation results will be saved in the `log/<model_save_id>/<CoderUJB_task>` folder.

```bash
./scripts/run_code_ujb.sh eval complete codeujbcomplete ./models/codellama-7b/ codellama-7b
./scripts/run_code_ujb.sh eval complete codeujbtestgen ./models/codellama-7b/ codellama-7b
./scripts/run_code_ujb.sh eval complete codeujbtestgenissue ./models/codellama-7b/ codellama-7b
./scripts/run_code_ujb.sh eval complete codeujbrepair ./models/codellama-7b/ codellama-7b
./scripts/run_code_ujb.sh eval complete codeujbdefectdetection ./models/codellama-7b/ codellama-7b

./scripts/run_code_ujb.sh eval chat codeujbcomplete ./models/codellama-instruct-7b/ codellama-instruct-7b
./scripts/run_code_ujb.sh eval chat codeujbtestgen ./models/codellama-instruct-7b/ codellama-instruct-7b
./scripts/run_code_ujb.sh eval chat codeujbtestgenissue ./models/codellama-instruct-7b/ codellama-instruct-7b
./scripts/run_code_ujb.sh eval chat codeujbrepair ./models/codellama-instruct-7b/ codellama-instruct-7b
./scripts/run_code_ujb.sh eval chat codeujbdefectdetection ./models/codellama-instruct-7b/ codellama-instruct-7b

./scripts/run_code_ujb.sh eval chat codeujbcomplete gpt-3.5-turbo gpt-3.5-turbo
./scripts/run_code_ujb.sh eval chat codeujbtestgen gpt-3.5-turbo gpt-3.5-turbo
./scripts/run_code_ujb.sh eval chat codeujbtestgenissue gpt-3.5-turbo gpt-3.5-turbo
./scripts/run_code_ujb.sh eval chat codeujbrepair gpt-3.5-turbo gpt-3.5-turbo
./scripts/run_code_ujb.sh eval chat codeujbdefectdetection gpt-3.5-turbo gpt-3.5-turbo
```