from datasets import load_dataset

dataset = load_dataset("livecodebench/code_generation_lite", version_tag="release_v1")
print(dataset)
