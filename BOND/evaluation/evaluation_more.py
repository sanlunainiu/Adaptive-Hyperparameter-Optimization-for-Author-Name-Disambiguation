import numpy as np
import os
import csv
from os.path import join
from tqdm import tqdm
from datetime import datetime
from dataset.load_data import load_json
from functools import reduce
import math


def evaluate(predict_result, ground_truth, metric_type='pairwise'):
    """
    
    Parameters
    ----------
    predict_result: dict or str
    ground_truth: dict or str
    metric_type: str
        
    Returns
    -------
    float: metrics
    """
    if isinstance(predict_result, str):
        predict_result = load_json(predict_result)
    if isinstance(ground_truth, str):
        ground_truth = load_json(ground_truth)

    valid_metrics = {'pairwise', 'kmetric', 'b3'}
    if metric_type not in valid_metrics:
        raise ValueError(f"metric_type must be one of {valid_metrics}")

    name_nums = 0
    result_list = []
    output_file = rf'out\evaluation_valid_results_{metric_type}.csv'
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Name', 'Precision', 'Recall', 'F1'])

        for name in predict_result:
            predicted_pubs = dict()
            for idx, pids in enumerate(predict_result[name]):
                for pid in pids:
                    predicted_pubs[pid] = idx
            
            pubs = []
            ilabel = 0
            true_labels = []
            for aid in ground_truth[name]:
                pubs.extend(ground_truth[name][aid])
                true_labels.extend([ilabel] * len(ground_truth[name][aid]))
                ilabel += 1

            predict_labels = []
            for pid in pubs:
                predict_labels.append(predicted_pubs.get(pid, -1))

            if metric_type == 'pairwise':
                precision, recall, f1 = pairwise_evaluate(true_labels, predict_labels)
            else:
                true_clusters = {}
                pred_clusters = {}
                
                unique_true_labels = set(true_labels)
                for label in unique_true_labels:
                    true_clusters[str(label)] = [pubs[i] for i, l in enumerate(true_labels) if l == label]
                
                unique_pred_labels = set(predict_labels)
                for label in unique_pred_labels:
                    if label != -1:  
                        pred_clusters[str(label)] = [pubs[i] for i, l in enumerate(predict_labels) if l == label]

                if metric_type == 'kmetric':
                    precision, recall, f1 = kmetric_precision_recall_fscore(true_clusters, pred_clusters)
                else:  # b3
                    precision, recall, f1 = evaluate_b3(predict_result[name], ground_truth[name])

            writer.writerow([name, precision, recall, f1])
            result_list.append((precision, recall, f1))
            name_nums += 1

        avg_precision = sum([result[0] for result in result_list]) / name_nums
        avg_recall = sum([result[1] for result in result_list]) / name_nums
        avg_f1 = sum([result[2] for result in result_list]) / name_nums
        writer.writerow(['Average', avg_precision, avg_recall, avg_f1])

    return avg_f1

def pairwise_evaluate(correct_labels, pred_labels):
    TP = 0.0  # Pairs Correctly Predicted To Same Author
    TP_FP = 0.0  # Total Pairs Predicted To Same Author
    TP_FN = 0.0  # Total Pairs To Same Author

    for i in range(len(correct_labels)):
        for j in range(i + 1, len(correct_labels)):
            if correct_labels[i] == correct_labels[j]:
                TP_FN += 1
            if pred_labels[i] == pred_labels[j]:
                TP_FP += 1
            if (correct_labels[i] == correct_labels[j]) and (pred_labels[i] == pred_labels[j]):
                TP += 1

    if TP == 0:
        pairwise_precision = 0
        pairwise_recall = 0
        pairwise_f1 = 0
    else:
        pairwise_precision = TP / TP_FP
        pairwise_recall = TP / TP_FN
        pairwise_f1 = (2 * pairwise_precision * pairwise_recall) / (pairwise_precision + pairwise_recall)

    return pairwise_precision, pairwise_recall, pairwise_f1


def kmetric_precision_recall_fscore(cluster_to_signatures_true, cluster_to_signatures_pred):
    """k-metric: precision (ACP), recall (APP) and F-score
    
    Parameters
    ----------
    cluster_to_signatures_true: Dict[str, List[str]] 
    cluster_to_signatures_pred: Dict[str, List[str]]
    
    Returns
    -------
    precision: float (ACP)
    recall: float (APP)
    f_score: float
    """
    signature_to_pred_cluster = {}
    pred_cluster_sizes = {}
    
    for cluster_id, signatures in cluster_to_signatures_pred.items():
        pred_cluster_sizes[cluster_id] = len(signatures)
        for signature in signatures:
            signature_to_pred_cluster[signature] = cluster_id

    aap_sum = 0  # Author-Paper Precision sum
    acp_sum = 0  # Author-Cluster Precision sum
    total_signatures = 0
    
    for true_cluster_id, true_signatures in cluster_to_signatures_true.items():
        total_signatures += len(true_signatures)
        
        pred_cluster_counts = {}
        for signature in true_signatures:
            if signature in signature_to_pred_cluster:
                pred_cluster_id = signature_to_pred_cluster[signature]
                pred_cluster_counts[pred_cluster_id] = pred_cluster_counts.get(pred_cluster_id, 0) + 1
        
        for pred_cluster_id, count in pred_cluster_counts.items():
            aap_sum += (count ** 2) / len(true_signatures)
            acp_sum += (count ** 2) / pred_cluster_sizes[pred_cluster_id]
    
    recall = aap_sum / total_signatures if total_signatures > 0 else 0  # APP
    precision = acp_sum / total_signatures if total_signatures > 0 else 0  # ACP
    
    # F-score
    try:
        f_score = math.sqrt(recall * precision)
    except:
        f_score = 0.0

    return precision, recall, f_score

def evaluate_kmetric(predict_result, ground_truth):
    """
    Evaluate using K-Metric with same interface as original evaluate function
    
    Parameters
    ----------
    predict_result: dict or str
        Dictionary or path to JSON with predicted clusters
    ground_truth: dict or str  
        Dictionary or path to JSON with ground truth clusters
        
    Returns
    -------
    precision: float (ACP)
    recall: float (APP)
    f_score: float
    """
    
    if isinstance(predict_result, str):
        predict_result = load_json(predict_result)
    if isinstance(ground_truth, str):
        ground_truth = load_json(ground_truth)

    for name in predict_result:
        # Convert predict_result format
        pred_clusters = {}
        for idx, pids in enumerate(predict_result[name]):
            pred_clusters[str(idx)] = pids
            
        # Convert ground_truth format 
        true_clusters = {}
        for aid, pids in ground_truth[name].items():
            true_clusters[aid] = pids

        precision, recall, f_score = kmetric_precision_recall_fscore(
            true_clusters, pred_clusters)
            
        return precision, recall, f_score




def f1_score(precision: float, recall: float) -> float:
    if precision == 0 or recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def b3_precision_recall_fscore(true_clus, pred_clus, skip_signatures=None):
    """
    Compute the B^3 variant of precision, recall and F-score.
    Modified from: https://github.com/glouppe/beard/blob/master/beard/metrics/clustering.py

    Parameters
    ----------
    true_clus: Dict
        dictionary with cluster id as keys and 1d array containing
        the ground-truth signature id assignments as values.
    pred_clus: Dict
        dictionary with cluster id as keys and 1d array containing
        the predicted signature id assignments as values.
    skip_signatures: List[string]
        in the incremental setting blocks can be partially supervised,
        hence those instances are not used for evaluation.

    Returns
    -------
    float: calculated precision
    float: calculated recall
    float: calculated F1
    Dict: P/R/F1 per signature

    Reference
    ---------
    Amigo, Enrique, et al. "A comparison of extrinsic clustering evaluation
    metrics based on formal constraints." Information retrieval 12.4
    (2009): 461-486.
    """

    true_clusters = true_clus.copy()
    pred_clusters = pred_clus.copy()

    tcset = set(reduce(lambda x, y: x + y, true_clusters.values()))
    pcset = set(reduce(lambda x, y: x + y, pred_clusters.values()))

    if tcset != pcset:
        raise ValueError("Predictions do not cover all the signatures!")

    # incremental evaluation contains partially observed signatures
    # skip_signatures are observed signatures, which we skip for b3 calc.
    if skip_signatures is not None:
        tcset = tcset.difference(skip_signatures)

    for cluster_id, cluster in true_clusters.items():
        true_clusters[cluster_id] = frozenset(cluster)
    for cluster_id, cluster in pred_clusters.items():
        pred_clusters[cluster_id] = frozenset(cluster)

    precision = 0.0
    recall = 0.0

    rev_true_clusters = {}
    for k, v in true_clusters.items():
        for vi in v:
            rev_true_clusters[vi] = k

    rev_pred_clusters = {}
    for k, v in pred_clusters.items():
        for vi in v:
            rev_pred_clusters[vi] = k

    intersections = {}
    per_signature_metrics = {}
    n_samples = len(tcset)

    true_bigger_ratios, pred_bigger_ratios = [], []
    for item in list(tcset):
        pred_cluster_i = pred_clusters[rev_pred_clusters[item]]
        true_cluster_i = true_clusters[rev_true_clusters[item]]

        if len(pred_cluster_i) >= len(true_cluster_i):
            pred_bigger_ratios.append(len(pred_cluster_i) / len(true_cluster_i))
        else:
            true_bigger_ratios.append(len(true_cluster_i) / len(pred_cluster_i))

        if (pred_cluster_i, true_cluster_i) in intersections:
            intersection = intersections[(pred_cluster_i, true_cluster_i)]
        else:
            intersection = pred_cluster_i.intersection(true_cluster_i)
            intersections[(pred_cluster_i, true_cluster_i)] = intersection
        _precision = len(intersection) / len(pred_cluster_i)
        _recall = len(intersection) / len(true_cluster_i)
        precision += _precision
        recall += _recall
        per_signature_metrics[item] = (
            _precision,
            _recall,
            f1_score(_precision, _recall),
        )

    precision /= n_samples
    recall /= n_samples

    f_score = f1_score(precision, recall)

    return precision, recall, f_score



def evaluate_b3(predict_result, ground_truth):
    """
    
    Parameters
    ----------
    predict_result: List[List[str]] or List[str]
    ground_truth: Dict[str, List[str]]
        
    Returns
    -------
    precision: float
    recall: float 
    f_score: float
    """
    pred_clusters = {}
    for idx, pids in enumerate(predict_result):
        pred_clusters[str(idx)] = pids
        
    true_clusters = {}
    for aid, pids in ground_truth.items():
        true_clusters[aid] = pids
    
    precision, recall, f_score = b3_precision_recall_fscore(
        true_clusters, pred_clusters)

    return precision, recall, f_score


if __name__ == '__main__':
    predict = r'out\res.json'
    ground_truth = r'dataset\data\src\sna-valid\sna_valid_ground_truth.json'
    # evaluate(predict, ground_truth, metric_type='pairwise')
    # evaluate(predict, ground_truth, metric_type='kmetric')
    evaluate(predict, ground_truth, metric_type='b3')