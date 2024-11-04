import os
import torch.multiprocessing as mp
from tqdm import tqdm
import tree_sitter_languages  # pants: no-infer-dep

from code_ujb.tasks.custom_metrics.crosscodeeval_metric.eval_utils import (
    postprocess_code_lines,
    extract_identifiers,
    cal_edit_sim,
    remove_comments
)

PARSER = None
LANGUAGE = None

def compute_id_match(pred_ids, target_ids):
    pred_ids = list(set(pred_ids))
    target_ids = list(set(target_ids))
    tp = 0
    fp = 0
    fn = 0
    for pid in pred_ids:
        if pid in target_ids:
            tp += 1
        else:
            fp += 1
    for tid in target_ids:
        if tid not in pred_ids:
            fn += 1
    return tp, fp, fn


def compute_edit_sim(samples):
    refs, hyps = [], []
    for s in samples:
        refs.append(s["target"])
        hyps.append(s["pred"])
    return cal_edit_sim(refs, hyps)


def process_examples(args):
    sample, ex, language = args

    prediction = postprocess_code_lines(ex["prompt"], sample["pred"], PARSER, language)
    prediction = remove_comments(prediction)
    target = ex["groundtruth"]
    target = remove_comments(target)

    pred_lines = [l.strip() for l in prediction.split("\n") if l.strip()]
    gt_lines = [l.strip() for l in target.split("\n") if l.strip()]
    em_label = int(pred_lines == gt_lines)

    pred_ids = extract_identifiers(prediction, language)
    target_ids = extract_identifiers(target, language)

    trunc_s = {
        "task_idx": sample["task_idx"],
        "g_idx": sample["g_idx"],
        "pred": prediction,
        "target": target,
        "pred_ids": pred_ids,
        "target_ids": target_ids
    }
    return trunc_s, em_label


def compute_metric_stmt(language, generations, references):
    global PARSER
    # global parser
    language = "c_sharp" if language == "csharp" else language
    PARSER = tree_sitter_languages.get_parser(language)
    
    truncated_samples = []
    em_labels = []

    # print("post-processing samples ...")
    cpu_num = min(os.cpu_count() - 1, len(generations))
    pool = mp.Pool(cpu_num)
    tasks = []
    for idx, samples, ex in zip(range(len(generations)), generations, references):
        for g_idx, sample in enumerate(samples):
            tasks.append(({"task_idx":ex["task_idx"], "g_idx":g_idx, "pred":sample}, ex, language))
    
    with tqdm(total=len(generations)) as pbar:
        for output in pool.imap_unordered(process_examples, tasks):
            trunc_s, em_label = output
            em_labels.append(em_label)
            truncated_samples.append(trunc_s)
            pbar.update()

    exact_match = 0
    for trunc_s, em_label in zip(truncated_samples, em_labels):
        if em_label == 1:
            exact_match += 1

    ### Score calculation

    id_em = []
    edit_similarities = []
    detailed_results = []

    for idx, trunc_s in enumerate(truncated_samples):
        identifier_em = int(trunc_s["pred_ids"] == trunc_s["target_ids"])
        es = cal_edit_sim([trunc_s["target"]], [trunc_s["pred"]])
        id_tp, id_fp, id_fn = compute_id_match(trunc_s["pred_ids"], trunc_s["target_ids"])
        id_em.append(identifier_em)
        edit_similarities.append(es)

        detailed_results.append({
            "task_idx": trunc_s["task_idx"],
            "g_idx": trunc_s["g_idx"],
            "em": em_labels[idx],
            "es": es,
            "id_em": identifier_em,
            "id_precision": id_tp / (id_tp + id_fp) if (id_tp + id_fp) != 0 else 0,
            "id_recall": id_tp / (id_tp + id_fn) if (id_tp + id_fn) != 0 else 0,
            "id_f1": 2 * id_tp / (2 * id_tp + id_fp + id_fn) if (2 * id_tp + id_fp + id_fn) != 0 else 0,
        })

    em_ratio = round(exact_match / len(tasks) * 100, 2)
    edit_sim = round(sum(edit_similarities) / len(edit_similarities), 2)

    id_em_ratio = round(
        sum(detailed_results[idx]['id_em'] for idx in range(len(detailed_results))) / len(detailed_results) * 100, 2)
    id_precision = round(sum(detailed_results[idx]['id_precision'] for idx in range(len(detailed_results))) / len(
        detailed_results) * 100, 2)
    id_recall = round(
        sum(detailed_results[idx]['id_recall'] for idx in range(len(detailed_results))) / len(detailed_results) * 100,
        2)
    id_f1 = round(
        sum(detailed_results[idx]['id_f1'] for idx in range(len(detailed_results))) / len(detailed_results) * 100, 2)

    print(
        f"Code Matching: "
        f"EM {em_ratio:.2f}, "
        f"ES {edit_sim:.2f}"
    )

    print(
        f"ID matching: "
        f"EM {id_em_ratio}, "
        #f"Precision {id_precision}, "
        #f"Recall {id_recall}, "
        f"F1 {id_f1}"
    )

    results = {
        "em": em_ratio,
        "es": edit_sim,
        "id_em": id_em_ratio,
        "id_precision": id_precision,
        "id_recall": id_recall,
        "id_f1": id_f1,
        "total": len(truncated_samples),
        "detail": detailed_results
    }
    
    return results
