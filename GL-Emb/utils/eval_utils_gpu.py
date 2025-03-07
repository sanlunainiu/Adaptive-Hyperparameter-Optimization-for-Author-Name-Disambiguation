# This version is suitable for GPU training cluster_size/count.py program
import numpy as np
from tensorflow.keras import backend as K
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Model
from collections import defaultdict
import math

# Huang et al (2006) Efficient name disambiguation for large-scale databases
def cluster_precision_recall_f1(T, R):

    T_clusters = {i: set() for i in set(T)}
    R_clusters = {i: set() for i in set(R)}
    
    for idx, t in enumerate(T):
        T_clusters[t].add(idx)
        
    for idx, r in enumerate(R):
        R_clusters[r].add(idx)
        

    T_sets = list(T_clusters.values())
    R_sets = list(R_clusters.values())
    

    correct_clusters = sum(1 for r_set in R_sets for t_set in T_sets if r_set == t_set)
    

    cluster_precision = correct_clusters / len(R_sets) if R_sets else 0

    cluster_recall = correct_clusters / len(T_sets) if T_sets else 0

    cluster_f1 = 2 * cluster_precision * cluster_recall / (cluster_precision + cluster_recall) if (cluster_precision + cluster_recall) > 0 else 0
    
    return cluster_precision, cluster_recall, cluster_f1


def improved_fast_pairwise_precision_recall_f1(preds, truths):
    """
    Compute the pairwise precision, recall, and F1-score, considering special handling when each element in preds
    and truths is unique (indicating perfect prediction).
    
    Args:
    preds (list of int): The predicted cluster labels.
    truths (list of int): The ground truth cluster labels.
    
    Returns:
    tuple of (float, float, float): precision, recall, and F1-score.
    """
    # If every element is unique, then it's a perfect prediction
    if len(preds) == len(set(preds)) and len(truths) == len(set(truths)):
        return 1.0, 1.0, 1.0

    # Otherwise, use the original pairwise computation
    pred_pairs = defaultdict(int)
    true_pairs = defaultdict(int)
    common_pairs = defaultdict(int)

    for pred, true in zip(preds, truths):
        pred_pairs[pred] += 1
        true_pairs[true] += 1
        common_pairs[(pred, true)] += 1

    tp = sum(common_pairs[(pred, true)] * (common_pairs[(pred, true)] - 1) / 2 for pred, true in common_pairs)
    fp = sum(pred_pairs[pred] * (pred_pairs[pred] - 1) / 2 - sum(common_pairs[(pred, true)] * (common_pairs[(pred, true)] - 1) / 2 for true in true_pairs) for pred in pred_pairs)
    fn = sum(true_pairs[true] * (true_pairs[true] - 1) / 2 - sum(common_pairs[(pred, true)] * (common_pairs[(pred, true)] - 1) / 2 for pred in pred_pairs) for true in true_pairs)

    tn = 0  
    tp_plus_fp = tp + fp
    tp_plus_fn = tp + fn
    precision = tp / tp_plus_fp if tp_plus_fp > 0 else 0
    recall = tp / tp_plus_fn if tp_plus_fn > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


def pairwise_precision_recall_f1(preds, truths):
    tp = 0
    fp = 0
    fn = 0
    n_samples = len(preds)
    for i in range(n_samples - 1):
        pred_i = preds[i]
        for j in range(i + 1, n_samples):
            pred_j = preds[j]
            if pred_i == pred_j:
                if truths[i] == truths[j]:
                    tp += 1
                else:
                    fp += 1
            elif truths[i] == truths[j]:
                fn += 1
    tp_plus_fp = tp + fp
    tp_plus_fn = tp + fn
    if tp_plus_fp == 0:
        precision = 0.
    else:
        precision = tp / tp_plus_fp
    if tp_plus_fn == 0:
        recall = 0.
    else:
        recall = tp / tp_plus_fn

    if not precision or not recall:
        f1 = 0.
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def cal_f1(prec, rec):
    return 2*prec*rec/(prec+rec)


def get_hidden_output(model, inp):
    intermediate_model = Model(inputs=model.input, outputs=model.layers[5].output)
    activations = intermediate_model.predict(inp)
    return activations



def predict(anchor_emb, test_embs):
    score1 = np.linalg.norm(anchor_emb-test_embs[0])
    score2 = np.linalg.norm(anchor_emb-test_embs[1])
    return [score1, score2]


def full_auc(model, test_triplets):
    """
    Measure AUC for model and ground truth on all items.

    Returns:
    - float AUC
    """

    grnds = []
    preds = []
    preds_before = []
    embs_anchor, embs_pos, embs_neg = test_triplets

    inter_embs_anchor = get_hidden_output(model, embs_anchor)
    inter_embs_pos = get_hidden_output(model, embs_pos)
    inter_embs_neg = get_hidden_output(model, embs_neg)
    # print(inter_embs_pos.shape)

    accs = []
    accs_before = []

    for i, e in enumerate(inter_embs_anchor):
        if i % 10000 == 0:
            print('test', i)

        emb_anchor = e
        emb_pos = inter_embs_pos[i]
        emb_neg = inter_embs_neg[i]
        test_embs = np.array([emb_pos, emb_neg])

        emb_anchor_before = embs_anchor[i]
        emb_pos_before = embs_pos[i]
        emb_neg_before = embs_neg[i]
        test_embs_before = np.array([emb_pos_before, emb_neg_before])

        predictions = predict(emb_anchor, test_embs)
        predictions_before = predict(emb_anchor_before, test_embs_before)

        acc_before = 1 if predictions_before[0] < predictions_before[1] else 0
        acc = 1 if predictions[0] < predictions[1] else 0
        accs_before.append(acc_before)
        accs.append(acc)

        grnd = [0, 1]
        grnds += grnd
        preds += predictions
        preds_before += predictions_before

    auc_before = roc_auc_score(grnds, preds_before)
    auc = roc_auc_score(grnds, preds)
    print('test accuracy before', np.mean(accs_before))
    print('test accuracy after', np.mean(accs))

    print('test AUC before', auc_before)
    print('test AUC after', auc)
    return auc

# k_metric from https://github.com/sanlunainiu/CluEval/blob/main/metrics.py
def k_metric(preds, truths):
    """
    Compute the K-metric precision, recall, and F1-score.
    
    Args:
    preds (list of int): The predicted cluster labels.
    truths (list of int): The ground truth cluster labels.
    
    Returns:
    tuple of (float, float, float): precision, recall, and F1-score.
    """
    # Convert lists to cluster mappings
    cluster_to_signatures_true = defaultdict(list)
    cluster_to_signatures_pred = defaultdict(list)
    
    # Build the cluster mappings
    for idx, (pred, truth) in enumerate(zip(preds, truths)):
        signature = str(idx)  # Use index as signature ID
        cluster_to_signatures_true[str(truth)].append(signature)
        cluster_to_signatures_pred[str(pred)].append(signature)
    
    # Initialize counters
    aap_sum = 0  # Author-Paper Precision sum
    acp_sum = 0  # Author-Cluster Precision sum
    total_signatures = len(preds)
    
    # Calculate cluster sizes for predicted clusters
    pred_cluster_sizes = {cluster: len(sigs) for cluster, sigs in cluster_to_signatures_pred.items()}
    
    # For each true cluster
    for true_cluster, true_signatures in cluster_to_signatures_true.items():
        # Count signatures in predicted clusters
        pred_cluster_counts = defaultdict(int)
        for signature in true_signatures:
            # Find which predicted cluster contains this signature
            for pred_cluster, pred_signatures in cluster_to_signatures_pred.items():
                if signature in pred_signatures:
                    pred_cluster_counts[pred_cluster] += 1
        
        # Calculate metrics
        for pred_cluster, count in pred_cluster_counts.items():
            aap_sum += (count ** 2) / len(true_signatures)
            acp_sum += (count ** 2) / pred_cluster_sizes[pred_cluster]
    
    # Calculate final metrics
    precision = acp_sum / total_signatures if total_signatures > 0 else 0
    recall = aap_sum / total_signatures if total_signatures > 0 else 0
    f1 = math.sqrt(precision * recall) if precision * recall > 0 else 0
    
    return precision, recall, f1

# b3_metric from s2and: https://github.com/allenai/S2AND
def b3_metric(preds, truths):
    """
    Compute the B³ precision, recall, and F1-score.
    
    Args:
    preds (list of int): The predicted cluster labels.
    truths (list of int): The ground truth cluster labels.
    
    Returns:
    tuple of (float, float, float): precision, recall, and F1-score.
    """
    # Convert to cluster dictionaries
    true_clusters = defaultdict(set)
    pred_clusters = defaultdict(set)
    
    for idx, (pred, truth) in enumerate(zip(preds, truths)):
        true_clusters[truth].add(idx)
        pred_clusters[pred].add(idx)
    
    # Convert sets to frozensets
    true_clusters = {k: frozenset(v) for k, v in true_clusters.items()}
    pred_clusters = {k: frozenset(v) for k, v in pred_clusters.items()}
    
    # Create reverse mappings
    rev_true_clusters = {}
    rev_pred_clusters = {}
    
    for k, v in true_clusters.items():
        for vi in v:
            rev_true_clusters[vi] = k
            
    for k, v in pred_clusters.items():
        for vi in v:
            rev_pred_clusters[vi] = k
    
    # Calculate metrics
    precision = 0.0
    recall = 0.0
    n_samples = len(preds)
    intersections = {}
    
    for idx in range(n_samples):
        pred_cluster = pred_clusters[rev_pred_clusters[idx]]
        true_cluster = true_clusters[rev_true_clusters[idx]]
        
        if (pred_cluster, true_cluster) in intersections:
            intersection = intersections[(pred_cluster, true_cluster)]
        else:
            intersection = pred_cluster.intersection(true_cluster)
            intersections[(pred_cluster, true_cluster)] = intersection
            
        precision += len(intersection) / len(pred_cluster)
        recall += len(intersection) / len(true_cluster)
    
    precision /= n_samples
    recall /= n_samples
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1