import os, json
from datasets import load_dataset
from huggingface_hub import HfApi

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

def upload(file_path, hf_repo_id):
    field = list(json.load(open(file_path, 'r')).keys())[0]
    dataset = load_dataset("json", data_files=file_path, field=field)
    api = HfApi()
    api.create_repo(repo_id=hf_repo_id, repo_type="dataset")
    dataset.push_to_hub(repo_id)

if __name__ == "__main__":
    repo_id = "YOUR_HF_ID/code_ujb_complete"
    file_path = os.path.join(FILE_DIR, 'task_complete_bench_default|2048.json')
    upload(file_path, repo_id)
    
    repo_id = "YOUR_HF_ID/code_ujb_testgen"
    file_path = os.path.join(FILE_DIR, 'task_testgen_bench_default|2048.json')
    upload(file_path, repo_id)
    
    repo_id = "YOUR_HF_ID/code_ujb_testgenissue"
    file_path = os.path.join(FILE_DIR, 'task_testgenissue_bench_default|2048.json')
    upload(file_path, repo_id)
    
    repo_id = "YOUR_HF_ID/code_ujb_repair"
    file_path = os.path.join(FILE_DIR, 'task_repair_bench_default|2048.json')
    upload(file_path, repo_id)
    
    repo_id = "YOUR_HF_ID/code_ujb_defectdetection"
    file_path = os.path.join(FILE_DIR, 'task_defectdetection_bench_default|2048.json')
    upload(file_path, repo_id)

