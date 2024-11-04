from abc import ABC, abstractmethod
from warnings import warn
from datasets import load_dataset


def remove_line_comment(signature):
    pure_signature = ""
    line_comment = False
    for idx, c in enumerate(signature):
        if c == "/" and idx<len(signature)-1 and signature[idx+1] == "/":
            line_comment = True
        if line_comment:
            if c == "\n":
                line_comment = False
            continue
        pure_signature += c
    return pure_signature

def clean_signature(signature):
    signature = remove_line_comment(signature)
    if signature.startswith("@"):
        for idx, c in enumerate(signature):
            if c == " " or c == "\n":
                break
        pre_signature = signature[:idx]+"\n"
        sub_signature = signature[idx:].strip()
        sub_signature = sub_signature.split("(")[0].strip()
    else:
        pre_signature = ""
        sub_signature = signature.split("(")[0]
    return pre_signature, sub_signature

class Task(ABC):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    # The name of the `Task` benchmark as denoted in the HuggingFace datasets Hub
    DATASET_PATH: str = None

    # The name of a subset within `DATASET_PATH`.
    DATASET_NAME: str = None

    def __init__(self, stop_words=None, requires_execution=True):
        """
        :param stop_words: list
            list of stop words if the generation uses a stopping criteria during generation
        :param requires_execution: bool
            wheter the task requires code execution during evaluation or not
        """
        self.stop_words = stop_words
        self.requires_execution = requires_execution
        try:
            if hasattr(self, "dataset") and self.dataset is None: raise Exception("Use locally downloaded dataset.")
            revision = self.DATASET_REVISION if hasattr(self, "DATASET_REVISION") else None
            self.dataset = load_dataset(path=self.DATASET_PATH, name=self.DATASET_NAME, revision=revision, trust_remote_code=True)
        except Exception as e:
            print(e)
            warn(
                "This task will use a locally downloaded dataset, not from the HF hub."
            )
    
    def get_prompt_byidx(self, idx, mode="complete"):
        """Builds the prompt for the LM to generate from."""
        return self.get_prompt(self.get_dataset()[idx], mode=mode)
    
    def get_id_byidx(self, idx):
        """Builds the prompt for the LM to generate from."""
        return idx

    def get_stream_stop(self, idx, mode="complete"):
        return None
    
    @abstractmethod
    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return []

    def fewshot_examples(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        pass

    def get_prompt(self, doc, mode="complete"):
        """Builds the prompt for the LM to generate from."""
        if mode == "complete":
            return self.get_prompt_complete(doc)
        elif mode == "chat":
            return self.get_prompt_chat(doc)
        else:
            raise KeyError()

    @abstractmethod
    def get_reference(self, doc):
        """Builds the reference solution for the doc.
        :param doc: dict[str: str]
            sample from the test dataset
        """
        pass

    def postprocess_generations(self, generations, idx, mode="complete"):
        if mode == "complete":
            return [self.postprocess_generation_complete(gen, idx) for gen in generations]
        elif mode == "chat":
            return [self.postprocess_generation_chat(gen, idx) for gen in generations]
        else:
            raise KeyError()
    
    @abstractmethod
    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations as in {"metric_name": result}.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        :return: dict[str: float]
        """
        pass

    @staticmethod
    def _stop_at_stop_token(decoded_string, stop_tokens):
        """
        Produces the prefix of decoded_string that ends at the first occurrence of
        a stop_token.
        WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
        itself.
        """
        min_stop_index = len(decoded_string)
        for stop_token in stop_tokens:
            stop_index = decoded_string.find(stop_token)
            if stop_index != -1 and stop_index < min_stop_index:
                min_stop_index = stop_index
        return decoded_string[:min_stop_index]

    def evaluate(self, generations):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations as in {"metric_name": result}.
        :param generations: list(list(str))
            list of lists containing generations
        :return: dict[str: float]
        """
        references = [self.get_reference(self.get_dataset()[gen["task_idx"]]) for gen in generations]
        generations = [gen["outputs"] for gen in generations]
        return self.process_results(generations, references)
