import sys
sys.path.append(r'C:\Users\Jason Burne\Desktop\WhoIsWho\bond')
import numpy as np
import os
from os.path import join
from tqdm import tqdm
from datetime import datetime
from dataset.load_data import load_json

import sys
import math
from functools import reduce
import numpy as np
from os.path import join
from tqdm import tqdm
from datetime import datetime
from dataset.load_data import load_json


def evaluate(predict_result, ground_truth, metric="pairwise"):
    """
    Evaluate clustering results using specified metric
    
    Parameters
    ----------
    predict_result : dict or str
        Predicted clustering results or path to JSON file
    ground_truth : dict or str
        Ground truth clustering or path to JSON file
    metric : str
        Evaluation metric to use: "pairwise", "kmetric", or "b3"
        
    Returns
    -------
    precision, recall, f1 scores
    """
    if isinstance(predict_result, str):
        predict_result = load_json(predict_result)
    if isinstance(ground_truth, str):
        ground_truth = load_json(ground_truth)

    for name in predict_result:
        if metric == "pairwise":
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
                predict_labels.append(predicted_pubs[pid])

            return pairwise_evaluate(true_labels, predict_labels)
            
        elif metric == "kmetric":
            pred_clusters = {}
            for idx, pids in enumerate(predict_result[name]):
                pred_clusters[str(idx)] = pids
                
            true_clusters = {}
            for aid, pids in ground_truth[name].items():
                true_clusters[aid] = pids
                
            return kmetric_precision_recall_fscore(true_clusters, pred_clusters)
            
        elif metric == "b3":
            pred_clusters = {}
            for idx, pids in enumerate(predict_result[name]):
                pred_clusters[str(idx)] = pids
                
            true_clusters = {}
            for aid, pids in ground_truth[name].items():
                true_clusters[aid] = pids
                
            precision, recall, f_score, _, _, _ = b3_precision_recall_fscore(
                true_clusters, pred_clusters)
            return precision, recall, f_score
        
        else:
            raise ValueError(f"Unknown metric: {metric}")

def f1_score(precision, recall):
    """Calculate F1 score"""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def kmetric_precision_recall_fscore(cluster_to_signatures_true, cluster_to_signatures_pred):
    """Calculate k-metric precision (ACP), recall (APP) and F-score"""
    signature_to_pred_cluster = {}
    pred_cluster_sizes = {}
    
    for cluster_id, signatures in cluster_to_signatures_pred.items():
        pred_cluster_sizes[cluster_id] = len(signatures)
        for signature in signatures:
            signature_to_pred_cluster[signature] = cluster_id

    aap_sum = 0  
    acp_sum = 0  
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
    
    recall = aap_sum / total_signatures if total_signatures > 0 else 0
    precision = acp_sum / total_signatures if total_signatures > 0 else 0
    
    try:
        f_score = math.sqrt(recall * precision)
    except:
        f_score = 0.0

    return precision, recall, f_score

def b3_precision_recall_fscore(true_clus, pred_clus, skip_signatures=None):
    """Compute B^3 variant of precision, recall and F-score"""
    true_clusters = true_clus.copy()
    pred_clusters = pred_clus.copy()

    tcset = set(reduce(lambda x, y: x + y, true_clusters.values()))
    pcset = set(reduce(lambda x, y: x + y, pred_clusters.values()))

    if tcset != pcset:
        raise ValueError("Predictions do not cover all the signatures!")

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
        per_signature_metrics[item] = (_precision, _recall, f1_score(_precision, _recall))

    precision /= n_samples
    recall /= n_samples
    f_score = f1_score(precision, recall)

    return (precision, recall, f_score, per_signature_metrics, 
            pred_bigger_ratios, true_bigger_ratios)



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

if __name__ == '__main__':
    predict = r'C:\Users\Jason Burne\Desktop\WhoIsWho\bond\out\dong_chen_res.json'
    ground_truth = r'C:\Users\Jason Burne\Desktop\WhoIsWho\bond\dataset\data\src\sna-valid\sna_valid_ground_truth.json'

    # Evaluate using all three metrics
    print("Pairwise metrics:")
    p, r, f1 = evaluate(predict, ground_truth, metric="pairwise")
    print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
    
    print("\nK-metrics:")
    p, r, f1 = evaluate(predict, ground_truth, metric="kmetric")
    print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
    
    print("\nB3 metrics:")
    p, r, f1 = evaluate(predict, ground_truth, metric="b3")
    print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")