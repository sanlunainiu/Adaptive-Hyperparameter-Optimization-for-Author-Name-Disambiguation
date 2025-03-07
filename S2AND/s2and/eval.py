from typing import Dict, Optional, Any, List, Tuple, TYPE_CHECKING, Union

import logging
import pickle
import json
import warnings
from collections import Counter

if TYPE_CHECKING:  # need this for circular import issues
    from s2and.model import Clusterer
    from s2and.data import ANDData

import os
from os.path import join
from functools import reduce
from collections import defaultdict

import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.calibration import CalibratedClassifierCV
import copy
from tqdm import tqdm

from s2and.featurizer import many_pairs_featurize
import math
logger = logging.getLogger("s2and")

sns.set(context="talk")


def cluster_eval(
    dataset: "ANDData",
    clusterer: "Clusterer",
    split: str = "test",
    use_s2_clusters: bool = False,
) -> Tuple[Dict[str, Tuple], Dict[str, Tuple[float, float, float]]]:
    """
    Performs clusterwise evaluation.
    Returns B3, Cluster F1, and Cluster Macro F1.

    Parameters
    ----------
    dataset: ANDData
        Dataset that has ground truth
    clusterer: Clusterer
        Clusterer object that will do predicting.
    split: string
        Which split in the dataset are we evaluating?

    Returns
    -------
    Dict: Dictionary of clusterwise metrics.
    Dict: Same as above but broken down by signature.
    """
    
    train_block_dict, val_block_dict, test_block_dict = dataset.split_blocks_helper(dataset.get_blocks())
    if split == "test":
        block_dict = test_block_dict
    elif split == "val":
        block_dict = val_block_dict
    elif split == "train":
        block_dict = train_block_dict
    else:
        raise Exception("Split must be one of: {train, val, test}!")

    # block ground truth labels: cluster_to_signatures
    cluster_to_signatures = dataset.construct_cluster_to_signatures(block_dict)

    # predict
    pred_clusters, _ = clusterer.predict(block_dict, dataset, use_s2_clusters=use_s2_clusters)

    # get metrics
    (
        b3_p,
        b3_r,
        b3_f1,
        b3_metrics_per_signature,
        pred_bigger_ratios,
        true_bigger_ratios,
    ) = b3_precision_recall_fscore(cluster_to_signatures, pred_clusters)
    metrics: Dict[str, Tuple] = {"B3 (P, R, F1)": (b3_p, b3_r, b3_f1)}
    metrics["Cluster (P, R F1)"] = pairwise_precision_recall_fscore(
        cluster_to_signatures, pred_clusters, block_dict, "clusters"
    )
    metrics["Cluster Macro (P, R, F1)"] = pairwise_precision_recall_fscore(
        cluster_to_signatures, pred_clusters, block_dict, "cmacro"
    )
    metrics["Pred bigger ratio (mean, count)"] = (
        np.round(np.mean(pred_bigger_ratios), 2),
        len(pred_bigger_ratios),
    )
    metrics["True bigger ratio (mean, count)"] = (
        np.round(np.mean(true_bigger_ratios), 2),
        len(true_bigger_ratios),
    )

    return metrics, b3_metrics_per_signature


def cluster_eval_all(
    dataset: "ANDData",
    clusterer: "Clusterer",
    split: str = "test",
    use_s2_clusters: bool = False,
) -> Tuple[Dict[str, Tuple], Dict[str, Tuple[float, float, float]], Dict[str, Dict]]:
    """
    Performs clusterwise evaluation.
    
    Parameters
    ----------
    dataset: ANDData
        Dataset that has ground truth
    clusterer: Clusterer
        Clusterer object that will do predicting.
    split: string
        Which split in the dataset are we evaluating?
        
    Returns
    -------
    Dict: Dictionary of overall clusterwise metrics
    Dict: Metrics broken down by signature
    Dict: Metrics broken down by block
    """

    train_block_dict, val_block_dict, test_block_dict = dataset.split_blocks_helper(dataset.get_blocks())
    

    all_block_dict = {}
    all_block_dict.update(train_block_dict)
    all_block_dict.update(val_block_dict)
    all_block_dict.update(test_block_dict)
    
    if split == "test":
        block_dict = test_block_dict
    elif split == "val":
        block_dict = val_block_dict
    elif split == "train":
        block_dict = train_block_dict
    elif split == "all":
        block_dict = all_block_dict
    else:
        raise Exception("Split must be one of: {train, val, test}!")


    block_metrics = {}
    
    for block_name, signatures in block_dict.items():
        block_cluster_to_signatures = dataset.construct_cluster_to_signatures({block_name: signatures})
        
        block_pred_clusters, _ = clusterer.predict({block_name: signatures}, dataset, use_s2_clusters=use_s2_clusters)
        
        (
            block_b3_p,
            block_b3_r,
            block_b3_f1,
            block_b3_metrics_per_signature,
            block_pred_bigger_ratios,
            block_true_bigger_ratios,
        ) = b3_precision_recall_fscore(block_cluster_to_signatures, block_pred_clusters)
        
        block_kmetric_p, block_kmetric_r, block_kmetric_f1 = kmetric_precision_recall_fscore(
            block_cluster_to_signatures, 
            block_pred_clusters
        )
        
        block_metrics[block_name] = {
            "B3 (P, R, F1)": (block_b3_p, block_b3_r, block_b3_f1),
            "Cluster (P, R F1)": pairwise_precision_recall_fscore(
                block_cluster_to_signatures, block_pred_clusters, {block_name: signatures}, "clusters"
            ),
            "Cluster Macro (P, R, F1)": pairwise_precision_recall_fscore(
                block_cluster_to_signatures, block_pred_clusters, {block_name: signatures}, "cmacro"
            ),
            "K-metric (ACP, APP, F1)": (block_kmetric_p, block_kmetric_r, block_kmetric_f1),
            "Pred bigger ratio (mean, count)": (
                np.round(np.mean(block_pred_bigger_ratios), 2) if block_pred_bigger_ratios else 0,
                len(block_pred_bigger_ratios),
            ),
            "True bigger ratio (mean, count)": (
                np.round(np.mean(block_true_bigger_ratios), 2) if block_true_bigger_ratios else 0,
                len(block_true_bigger_ratios),
            ),
        }

    cluster_to_signatures = dataset.construct_cluster_to_signatures(block_dict)
    pred_clusters, _ = clusterer.predict(block_dict, dataset, use_s2_clusters=use_s2_clusters)

    # B3
    (
        b3_p,
        b3_r,
        b3_f1,
        b3_metrics_per_signature,
        pred_bigger_ratios,
        true_bigger_ratios,
    ) = b3_precision_recall_fscore(cluster_to_signatures, pred_clusters)
    
    # kmetric
    kmetric_p, kmetric_r, kmetric_f1 = kmetric_precision_recall_fscore(
        cluster_to_signatures,
        pred_clusters
    )
    
    metrics: Dict[str, Tuple] = {
        "B3 (P, R, F1)": (b3_p, b3_r, b3_f1),
        "Cluster (P, R F1)": pairwise_precision_recall_fscore(
            cluster_to_signatures, pred_clusters, block_dict, "clusters"
        ),
        "Cluster Macro (P, R, F1)": pairwise_precision_recall_fscore(
            cluster_to_signatures, pred_clusters, block_dict, "cmacro"
        ),
        "K-metric (ACP, APP, F1)": (kmetric_p, kmetric_r, kmetric_f1),
        "Pred bigger ratio (mean, count)": (
            np.round(np.mean(pred_bigger_ratios), 2),
            len(pred_bigger_ratios),
        ),
        "True bigger ratio (mean, count)": (
            np.round(np.mean(true_bigger_ratios), 2),
            len(true_bigger_ratios),
        )
    }

    return metrics, b3_metrics_per_signature, block_metrics


def cluster_eps_optimize_b3(
    dataset: "ANDData",
    clusterer: "Clusterer", 
    split: str = "test",
    n_iter: int = 20
) -> Dict[str, Dict[str, float]]:
    """
    Perform eps optimization evaluation for all blocks.
    
    Returns
    -------
    Dict[str, Dict[str, float]]: 
        A mapping from block to a dictionary of metrics, 
        where each metrics dictionary includes 'eps', 'f1', 'precision', 'recall'.
    """
    
    train_block_dict, val_block_dict, test_block_dict = dataset.split_blocks_helper(dataset.get_blocks())

    all_block_dict = {}
    all_block_dict.update(train_block_dict)
    all_block_dict.update(val_block_dict)
    all_block_dict.update(test_block_dict)
    
    if split == "test":
        block_dict = test_block_dict
    elif split == "val":
        block_dict = val_block_dict
    elif split == "train":
        block_dict = train_block_dict
    elif split == "all":    
        block_dict = all_block_dict
    else:
        raise Exception("Split must be one of: {train, val, test, all}!")

    block_metrics = {}
    
    for block_key, signatures in block_dict.items():
        if len(signatures) == 1:
            block_metrics[block_key] = {
                'eps': 1.0,
                'f1': 1.0,
                'precision': 1.0,
                'recall': 1.0
            }
            logger.info(f"Block {block_key}: Single signature, F1=1.000")
        else:
            block_dict_single = {block_key: signatures}
            block_true_clusters = dataset.construct_cluster_to_signatures(block_dict_single)
            
            metrics = clusterer.optimize_block_eps_b3(
                block_key=block_key,
                signatures=signatures,
                dataset=dataset,
                true_clusters=block_true_clusters,
                n_iter=n_iter
            )
            
            block_metrics[block_key] = metrics
            
            logger.info(f"Block {block_key}: optimal_eps={metrics['eps']:.3f}, "
                        f"best_f1={metrics['f1']:.3f}, "
                        f"best_precision={metrics['precision']:.3f}, "
                        f"best_recall={metrics['recall']:.3f}")
    
    return block_metrics


def cluster_eps_optimize_pairwise(
    dataset: "ANDData",
    clusterer: "Clusterer", 
    split: str = "test",
    n_iter: int = 20,
    optimize_method: str = "cmacro"
) -> Dict[str, Dict[str, float]]:
    """
    Perform eps optimization evaluation for all blocks.
    
    Parameters
    ----------
    dataset : ANDData
        Dataset object
    clusterer : Clusterer
        Clusterer object
    split : str, optional
        Dataset split, options include "train", "val", "test", "all", default is "test"
    n_iter : int, optional
        Number of optimization iterations, default is 20
    optimize_method : str, optional
        Optimization method selection:
        - "b3": Use B³ metric
        - "pairwise_cmacro": Use pairwise cmacro strategy
        - "pairwise_clusters": Use pairwise clusters strategy
        Default is "b3"
    
    Returns
    -------
    Dict[str, Dict[str, float]]: 
        A mapping from block to a dictionary of metrics, 
        where each metrics dictionary includes 'eps', 'f1', 'precision', 'recall'
    """
    logger.info(f"Starting cluster optimization with method: {optimize_method}, split: {split}")
    

    if split not in ["train", "val", "test", "all"]:
        raise ValueError("Split must be one of: {train, val, test, all}")
    
    if optimize_method not in ["b3", "pairwise_cmacro", "pairwise_clusters"]:
        raise ValueError("optimize_method must be one of: {b3, pairwise_cmacro, pairwise_clusters}")
    
    try:

        train_block_dict, val_block_dict, test_block_dict = dataset.split_blocks_helper(dataset.get_blocks())
        

        all_block_dict = {
            **train_block_dict,
            **val_block_dict,
            **test_block_dict
        }
        

        split_mapping = {
            "test": test_block_dict,
            "val": val_block_dict,
            "train": train_block_dict,
            "all": all_block_dict
        }
        block_dict = split_mapping[split]
        
    except Exception as e:
        logger.error(f"Error during data split: {str(e)}")
        raise
    

    block_metrics = {}
    total_blocks = len(block_dict)
    processed_blocks = 0
    

    for block_key, signatures in block_dict.items():
        try:
            logger.debug(f"Processing block {block_key} ({processed_blocks + 1}/{total_blocks})")
            
            if len(signatures) <= 1:

                block_metrics[block_key] = {
                    'eps': 1.0,
                    'f1': 1.0,
                    'precision': 1.0,
                    'recall': 1.0
                }
                logger.info(f"Block {block_key}: Single signature, perfect metrics")
            else:

                block_dict_single = {block_key: signatures}
                block_true_clusters = dataset.construct_cluster_to_signatures(block_dict_single)
                

                metric_type = "cmacro" if optimize_method == "pairwise_cmacro" else "clusters"
                metrics = clusterer.optimize_block_eps_pairwise(
                    block_key=block_key,
                    signatures=signatures,
                    dataset=dataset,
                    true_clusters=block_true_clusters,  # 传入true_clusters
                    n_iter=n_iter,
                    metric=metric_type
                )
            
                block_metrics[block_key] = metrics
                
                logger.info(
                    f"Block {block_key} ({optimize_method}): "
                    f"signatures={len(signatures)}, "
                    f"optimal_eps={metrics['eps']:.3f}, "
                    f"best_f1={metrics['f1']:.3f}, "
                    f"best_precision={metrics['precision']:.3f}, "
                    f"best_recall={metrics['recall']:.3f}"
                )
            
            processed_blocks += 1
            
        except Exception as e:
            logger.error(f"Error processing block {block_key}: {str(e)}")

            block_metrics[block_key] = {
                'eps': 0.5,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }
            processed_blocks += 1
    

    successful_blocks = sum(1 for metrics in block_metrics.values() if metrics['f1'] > 0)
    logger.info(f"Optimization completed: {successful_blocks}/{total_blocks} blocks processed successfully")
    
    return block_metrics



def cluster_eps_optimize_kmetric(
    dataset: "ANDData",
    clusterer: "Clusterer", 
    split: str = "test",
    n_iter: int = 20,
    optimize_method: str = "k_metric",
    use_s2_clusters: bool = False
) -> Dict[str, Dict[str, float]]:
    """
    Perform eps optimization evaluation for all blocks using the K metric.
    
    Parameters
    ----------
    dataset : ANDData
        Dataset object
    clusterer : Clusterer
        Clusterer object
    split : str, optional
        Dataset split, default is "test"
    n_iter : int, optional
        Number of optimization iterations, default is 20
    optimize_method : str, optional
        Supported methods are "k_metric" and "acp"
    
    Returns
    -------
    Dict[str, Dict[str, float]]:
        A mapping from block to a dictionary of metrics, including 'eps', 'k', 'acp', and 'aap'
    """
    logger.info(f"Starting cluster optimization with {optimize_method}, split: {split}")
    
    if optimize_method not in ["k_metric", "acp"]:
        raise ValueError("Currently only 'k_metric' and 'acp' methods are implemented")
    
    try:

        train_block_dict, val_block_dict, test_block_dict = dataset.split_blocks_helper(dataset.get_blocks())
        

        all_block_dict = {
            **train_block_dict,
            **val_block_dict,
            **test_block_dict
        }
        

        split_mapping = {
            "test": test_block_dict,
            "val": val_block_dict,
            "train": train_block_dict,
            "all": all_block_dict
        }
        
        if split not in split_mapping:
            raise ValueError("Split must be one of: {train, val, test, all}")
            
        block_dict = split_mapping[split]
        
    except Exception as e:
        logger.error(f"Error during data split: {str(e)}")
        raise


    block_metrics = {}
    total_blocks = len(block_dict)
    processed_blocks = 0
    

    total_signatures = sum(len(sigs) for sigs in block_dict.values())
    logger.info(f"Total blocks: {total_blocks}, Total signatures: {total_signatures}")
    

    for block_key, signatures in block_dict.items():
        try:
            logger.debug(f"Processing block {block_key} ({processed_blocks + 1}/{total_blocks})")
            logger.debug(f"Number of signatures in block: {len(signatures)}")
            
            if len(signatures) <= 1:

                metrics = {'eps': 1.0, 'k': 1.0, 'acp': 1.0, 'aap': 1.0}
                block_metrics[block_key] = metrics
                logger.info(f"Block {block_key}: Single signature, K=1.000")
            else:

                current_block_dict = {block_key: signatures}
                true_clusters = dataset.construct_cluster_to_signatures(current_block_dict)
                
                if not true_clusters:
                    logger.warning(f"No true clusters found for block {block_key}")
                    metrics = {'eps': 0.5, 'k': -float('inf'), 'acp': -float('inf'), 'aap': -float('inf')}
                else:

                    best_metrics = {
                        'eps': None,
                        'k': -float('inf'),
                        'acp': -float('inf'),
                        'aap': -float('inf')
                    }
                    

                    distance_matrices = clusterer.make_distance_matrices(current_block_dict, dataset)
                    
                    for _ in range(n_iter):
                        eps = np.random.uniform(0, 1)
                        cluster_params = {'eps': eps}
                        

                        pred_clusters, _ = clusterer.predict(
                            block_dict=current_block_dict,
                            dataset=dataset,
                            dists=distance_matrices,
                            cluster_model_params=cluster_params,
                            use_s2_clusters=use_s2_clusters
                        )
                        
                        if not pred_clusters:
                            continue
                            

                        acp, aap, k = kmetric_precision_recall_fscore(
                            true_clusters,
                            pred_clusters
                        )
                        
                        if k > best_metrics['k']:
                            best_metrics['eps'] = eps
                            best_metrics['k'] = k
                            best_metrics['acp'] = acp
                            best_metrics['aap'] = aap
                    
                    metrics = best_metrics
                    
                block_metrics[block_key] = metrics
                
                logger.info(
                    f"Block {block_key}: "
                    f"signatures={len(signatures)}, "
                    f"true_clusters={len(true_clusters)}, "
                    f"optimal_eps={metrics['eps']:.3f}, "
                    f"best_k={metrics['k']:.3f}, "
                    f"best_acp={metrics['acp']:.3f}, "
                    f"best_aap={metrics['aap']:.3f}"
                )
            
            processed_blocks += 1
            
        except Exception as e:
            logger.error(f"Error processing block {block_key}: {str(e)}")
            block_metrics[block_key] = {
                'eps': 0.5,
                'k': -float('inf'),
                'acp': -float('inf'),
                'aap': -float('inf')
            }
            processed_blocks += 1
            
        except Exception as e:
            logger.error(f"Error processing block {block_key}: {str(e)}")

            block_metrics[block_key] = {
                'eps': 0.5,
                'k': -float('inf'),
                'acp': -float('inf'),
                'aap': -float('inf')
            }
            processed_blocks += 1
    

    successful_blocks = sum(1 for metrics in block_metrics.values() if metrics['k'] > -float('inf'))
    valid_ks = [metrics['k'] for metrics in block_metrics.values() if metrics['k'] > -float('inf')]
    valid_acps = [metrics['acp'] for metrics in block_metrics.values() if metrics['acp'] > -float('inf')]
    valid_aaps = [metrics['aap'] for metrics in block_metrics.values() if metrics['aap'] > -float('inf')]
    
    mean_k = np.mean(valid_ks) if valid_ks else -float('inf')
    mean_acp = np.mean(valid_acps) if valid_acps else -float('inf')
    mean_aap = np.mean(valid_aaps) if valid_aaps else -float('inf')
    
    median_k = np.median(valid_ks) if valid_ks else -float('inf')
    median_acp = np.median(valid_acps) if valid_acps else -float('inf')
    median_aap = np.median(valid_aaps) if valid_aaps else -float('inf')
    
    logger.info(f"Optimization completed:")
    logger.info(f"- Successful blocks: {successful_blocks}/{total_blocks}")
    logger.info(f"- Mean K: {mean_k:.3f}")
    logger.info(f"- Mean ACP: {mean_acp:.3f}")
    logger.info(f"- Mean AAP: {mean_aap:.3f}")
    logger.info(f"- Median K: {median_k:.3f}")
    logger.info(f"- Median ACP: {median_acp:.3f}")
    logger.info(f"- Median AAP: {median_aap:.3f}")
    
    return block_metrics

def kmetric_precision_recall_fscore(cluster_to_signatures_true, cluster_to_signatures_pred):
    """
    Calculate the precision (ACP), recall (APP), and F-score for the k-metric.
    
    Parameters
    ----------
    cluster_to_signatures_true: Dict[str, List[str]] 
        Mapping of true clusters to signatures
    cluster_to_signatures_pred: Dict[str, List[str]]
        Mapping of predicted clusters to signatures
    
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


def incremental_cluster_eval(
    dataset: "ANDData", clusterer: "Clusterer", split: str = "test"
) -> Tuple[Dict[str, Tuple[float, float, float]], Dict[str, Tuple[float, float, float]]]:
    """
    Performs clusterwise evaluation for the incremental clustering setting.
    This includes both time-split and random split of signatures.
    Returns B3, Cluster F1, and Cluster Macro F1.

    Parameters
    ----------
    dataset: ANDData
        Dataset that has ground truth
    clusterer: Clusterer
        Clusterer object that will do predicting.
    split: string
        Which split in the dataset are we evaluating?

    Returns
    -------
    Dict: Dictionary of clusterwise metrics.
    Dict: Same as above but broken down by signature.
    """
    block_dict = dataset.get_blocks()
    (
        train_block_dict,
        val_block_dict,
        test_block_dict,
    ) = dataset.split_cluster_signatures()
    # evaluation must happen only on test-signatures in blocks, so remove train/val signatures
    observed_signatures = set()
    for _, signatures in train_block_dict.items():
        for signature in signatures:
            observed_signatures.add(signature)

    # use entire block of signatures for predictions
    # NOTE: train/val/test block dicts can have overlapping signatures in the incremental case
    eval_block_dict_full = {}
    if split == "test":
        for block_key, _ in test_block_dict.items():
            eval_block_dict_full[block_key] = block_dict[block_key]
        cluster_to_signatures = dataset.construct_cluster_to_signatures(test_block_dict)
        for _, signatures in val_block_dict.items():
            for signature in signatures:
                observed_signatures.add(signature)
    elif split == "val":
        cluster_to_signatures = dataset.construct_cluster_to_signatures(val_block_dict)
        eval_block_dict_full = copy.deepcopy(val_block_dict)
        for block_key, signatures in train_block_dict.items():
            if block_key in eval_block_dict_full:
                eval_block_dict_full[block_key].extend(signatures)
    else:
        raise Exception("Evaluation split must be one of: {val, test}!")

    partial_supervision: Dict[Tuple[str, str], Union[int, float]] = {}
    list_obs_signatures = list(observed_signatures)
    # considers the supervision as distances
    for i, signature_i in enumerate(list_obs_signatures):
        for signature_j in list_obs_signatures[i + 1 : len(list_obs_signatures)]:
            if dataset.signature_to_cluster_id[signature_i] == dataset.signature_to_cluster_id[signature_j]:
                partial_supervision[(signature_i, signature_j)] = 0
            else:
                partial_supervision[(signature_i, signature_j)] = 1

    # predict on test-blocks
    pred_clusters, _ = clusterer.predict(eval_block_dict_full, dataset, partial_supervision=partial_supervision)
    # to avoid sparsity in b3 computation, we use all the signatures' ground-truth
    full_cluster_to_signatures = dataset.construct_cluster_to_signatures(pred_clusters)

    eval_only_pred_clusters = {}
    for cluster_key, signatures in pred_clusters.items():
        test_signatures = list(set(signatures).difference(observed_signatures))
        assert len(set(test_signatures).intersection(observed_signatures)) == 0
        if len(test_signatures) > 0:
            eval_only_pred_clusters[cluster_key] = test_signatures

    # get metrics
    b3_p, b3_r, b3_f1, b3_metrics_per_signature, _, _ = b3_precision_recall_fscore(
        full_cluster_to_signatures, pred_clusters, skip_signatures=observed_signatures
    )
    metrics = {"B3 (P, R, F1)": (b3_p, b3_r, b3_f1)}
    metrics["Cluster (P, R F1)"] = pairwise_precision_recall_fscore(
        cluster_to_signatures, eval_only_pred_clusters, test_block_dict, "clusters"
    )
    metrics["Cluster Macro (P, R, F1)"] = pairwise_precision_recall_fscore(
        cluster_to_signatures, eval_only_pred_clusters, test_block_dict, "cmacro"
    )

    return metrics, b3_metrics_per_signature


def facet_eval(
    dataset: "ANDData",
    metrics_per_signature: Dict[str, Tuple[float, float, float]],
    block_type: str = "original",
) -> Tuple[
    Dict[str, List],
    Dict[str, List],
    Dict[int, List],
    Dict[int, List],
    Dict[int, List],
    Dict[int, List],
    Dict[int, List],
    Dict[int, List],
    Dict[int, List],
    Dict[int, List],
    Dict[int, List],
    Dict[int, List],
    Dict[int, List],
    Dict[int, List],
    Dict[int, List],
    List[dict],
]:
    """
    Extracts B3 per facets.
    The returned dictionaries are keyed by the metric itself. For example, the keys of the
    homonymity_f1 variable are floating points between 0 and 1 indicating the amount
    of homonymity. The values are the per-signature B3s that have this amount of homonymity.

    Parameters
    ----------
    dataset: ANDData Input dataset
    metrics_per_signature: Dict
        B3 P/R/F1 per signature.
        Second output of cluster_eval function.
    block_type: string
        Whether to use Semantic Scholar ("s2") or "original" blocks

    Returns
    -------
    Dict: B3 F1 broken down by perceived estimated gender.
    Dict: B3 F1 broken down by perceived estimated ethnicity.
    Dict: B3 F1 broken down by number of paper authors.
    Dict: B3 F1 broken down by year.
    Dict: B3 F1 broken down by block size.
    Dict: B3 F1 broken down by true cluster size.
    Dict: B3 F1 broken down by within-block homonymity fraction.
          Definition (per signature): Fraction of same names but within different clusters.
    Dict: B3 F1 broken down by within-block synonymity fraction.
          Definition (per signature): Fraction of different names but within same clusters.
    """
    block_len_dict = {}
    if block_type == "original":
        blocks = dataset.get_original_blocks()
    elif block_type == "s2":
        blocks = dataset.get_s2_blocks()
    else:
        raise Exception("block_type must one of: {'original', 's2'}!")

    for block_key, signature_ids in blocks.items():
        block_len_dict[block_key] = len(signature_ids)

    # we need to know the length of each cluster
    assert dataset.clusters is not None
    cluster_len_dict = {}
    for cluster_id, cluster_dict in dataset.clusters.items():
        cluster_len_dict[cluster_id] = len(cluster_dict["signature_ids"])

    # for homonymity and synonymity we need to check all pairs
    homonymity: Dict[str, int] = defaultdict(int)
    synonymity: Dict[str, int] = defaultdict(int)
    denominator: Dict[str, int] = defaultdict(int)
    signature_keys = list(metrics_per_signature.keys())
    for i, signature_key_a in enumerate(signature_keys):
        for signature_key_b in signature_keys[i + 1 :]:
            signature_a = dataset.signatures[signature_key_a]
            signature_b = dataset.signatures[signature_key_b]
            # these counts only make sense within blocks
            if block_type == "original":
                same_block = signature_a.author_info_given_block == signature_b.author_info_given_block
            elif block_type == "s2":
                same_block = signature_a.author_info_block == signature_b.author_info_block
            if same_block:
                same_name = signature_a.author_info_full_name == signature_b.author_info_full_name
                same_cluster = (
                    dataset.signature_to_cluster_id[signature_key_a] == dataset.signature_to_cluster_id[signature_key_b]
                )
                if same_name and not same_cluster:
                    homonymity[signature_key_a] += 1
                    homonymity[signature_key_b] += 1
                elif not same_name and same_cluster:
                    synonymity[signature_key_a] += 1
                    synonymity[signature_key_b] += 1
                denominator[signature_key_a] += 1
                denominator[signature_key_b] += 1

    # Keep track of facet specific f-score performance
    gender_f1 = defaultdict(list)
    ethnicity_f1 = defaultdict(list)
    author_num_f1 = defaultdict(list)
    year_f1 = defaultdict(list)
    block_len_f1 = defaultdict(list)
    cluster_len_f1 = defaultdict(list)
    homonymity_f1 = defaultdict(list)
    synonymity_f1 = defaultdict(list)
    # keep track feature availability facet specific f-score
    firstname_f1 = defaultdict(list)
    affiliation_f1 = defaultdict(list)
    email_f1 = defaultdict(list)
    abstract_f1 = defaultdict(list)
    venue_f1 = defaultdict(list)
    references_f1 = defaultdict(list)
    coauthors_f1 = defaultdict(list)

    signature_lookup = list()

    for signature_key, (p, r, f1) in metrics_per_signature.items():
        _signature_dict = dict()

        cluster_id = dataset.signature_to_cluster_id[signature_key]
        signature = dataset.signatures[signature_key]
        paper = dataset.papers[str(signature.paper_id)]

        if signature.author_info_estimated_gender is not None:
            gender_f1[signature.author_info_estimated_gender].append(f1)
            _signature_dict["estimated_gender"] = signature.author_info_estimated_gender

        if signature.author_info_estimated_ethnicity is not None:
            ethnicity_f1[signature.author_info_estimated_ethnicity[0:3]].append(f1)
            _signature_dict["estimated_ethnicity"] = signature.author_info_estimated_ethnicity

        author_num_f1[len(paper.authors)].append(f1)
        year_f1[paper.year].append(f1)
        cluster_len_f1[cluster_len_dict[cluster_id]].append(f1)

        # full first-name
        if signature.author_info_first is not None and len(signature.author_info_first.replace(".", "")) >= 2:
            firstname_f1[1].append(f1)
            _signature_dict["first name"] = 1
        else:
            firstname_f1[0].append(f1)
            _signature_dict["first name"] = 0

        if len(signature.author_info_affiliations) > 0:
            affiliation_f1[1].append(f1)
            _signature_dict["affiliation"] = 1
        else:
            affiliation_f1[0].append(f1)
            _signature_dict["affiliation"] = 0

        if signature.author_info_email not in {"", None}:
            email_f1[1].append(f1)
            _signature_dict["email"] = 1
        else:
            email_f1[0].append(f1)
            _signature_dict["email"] = 0

        if paper.has_abstract:
            abstract_f1[1].append(f1)
            _signature_dict["abstract"] = 1
        else:
            abstract_f1[0].append(f1)
            _signature_dict["abstract"] = 0

        if paper.venue not in {"", None} or paper.journal_name not in {"", None}:
            venue_f1[1].append(f1)
            _signature_dict["venue"] = 1
        else:
            venue_f1[0].append(f1)
            _signature_dict["venue"] = 0

        if len(paper.references) > 0:
            references_f1[1].append(f1)
            _signature_dict["references"] = 1
        else:
            references_f1[0].append(f1)
            _signature_dict["references"] = 0

        if len(signature.author_info_coauthors) > 0:
            coauthors_f1[1].append(f1)
            _signature_dict["multiple_authors"] = 1
        else:
            coauthors_f1[0].append(f1)
            _signature_dict["multiple_authors"] = 0

        if block_type == "original":
            block_len_f1[block_len_dict[signature.author_info_given_block]].append(f1)
            _signature_dict["block size"] = block_len_dict[signature.author_info_given_block]
        elif block_type == "s2":
            block_len_f1[block_len_dict[signature.author_info_block]].append(f1)
            _signature_dict["block size"] = block_len_dict[signature.author_info_block]

        if homonymity[signature_key] > 0:
            homonymity_f1[np.round(homonymity[signature_key] / denominator[signature_key], 2)].append(f1)
            _signature_dict["homonymity"] = np.round(homonymity[signature_key] / denominator[signature_key], 2)
        else:
            _signature_dict["homonymity"] = 0

        if synonymity[signature_key] > 0:
            synonymity_f1[np.round(synonymity[signature_key] / denominator[signature_key], 2)].append(f1)
            _signature_dict["synonymity"] = np.round(synonymity[signature_key] / denominator[signature_key], 2)
        else:
            _signature_dict["synonymity"] = 0

        _signature_dict["signature_id"] = signature_key
        _signature_dict["precision"] = p
        _signature_dict["recall"] = r
        _signature_dict["f1"] = f1
        _signature_dict["#authors"] = len(paper.authors)
        _signature_dict["year"] = paper.year
        _signature_dict["cluster size"] = cluster_len_dict[cluster_id]

        signature_lookup.append(_signature_dict)

    return (
        gender_f1,
        ethnicity_f1,
        author_num_f1,
        year_f1,
        block_len_f1,
        cluster_len_f1,
        homonymity_f1,
        synonymity_f1,
        firstname_f1,
        affiliation_f1,
        email_f1,
        abstract_f1,
        venue_f1,
        references_f1,
        coauthors_f1,
        signature_lookup,
    )


def pairwise_eval(
    X: np.ndarray,
    y: np.ndarray,
    classifier: Any,
    figs_path: str,
    title: str,
    shap_feature_names: List[str],
    thresh_for_f1: float = 0.5,
    shap_plot_type: Optional[str] = "dot",
    nameless_classifier: Optional[Any] = None,
    nameless_X: Optional[np.ndarray] = None,
    nameless_feature_names: Optional[List[str]] = None,
    skip_shap: bool = False,
) -> Dict[str, float]:
    """
    Performs pairwise model evaluation, without using blocks.
    Also writes plots to the provided file path

    Parameters
    ----------
    X: np.array
        Feature matrix of features to do eval on.
    y: np.array
        Feature matrix of labels to do eval on.
    classifier: sklearn compatible classifier
        Classifier to do eval on.
    figs_path: string
        Where to put the resulting evaluation figures.
    title: string
        Title to stick on all the plots and use for file name.
    shap_feature_names: List[str]
        List of feature names for the SHAP plots.
    thresh_for_f1: float
        Threshold for F1 computation. Defaults to 0.5.
    shap_plot_type: str
        Type of shap plot. Defaults to 'dot'.
        Can also be: 'bar', 'violin', 'compact_dot'
    nameless_classifier: sklearn compatible classifier
        Classifier to do eval on that doesn't use name features.
    nameless_X: np.array
        Feature matrix of features to do eval on excluding name features.
    nameless_feature_names: List[str]
        List of feature names for the SHAP plots excluding name features.
    skip_shap: bool
        Whether to skip SHAP entirely.

    Returns
    -------
    Dict: A dictionary of common pairwise metrics.
    """
    if not os.path.exists(figs_path):
        os.makedirs(figs_path)

    # filename base will be title but lower and underscores
    base_name = title.lower().replace(" ", "_")
    if hasattr(classifier, "classifier"):
        classifier = classifier.classifier

    if nameless_classifier is not None and hasattr(nameless_classifier, "classifier"):
        nameless_classifier = nameless_classifier.classifier

    if nameless_classifier is not None:
        y_prob = (classifier.predict_proba(X)[:, 1] + nameless_classifier.predict_proba(nameless_X)[:, 1]) / 2
    else:
        y_prob = classifier.predict_proba(X)[:, 1]

    # plot AUROC
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(0, figsize=(15, 15))
    plt.plot(fpr, tpr, lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for {title}")
    plt.legend(loc="lower right")
    plt.savefig(join(figs_path, base_name + "_roc.png"))
    plt.clf()
    plt.close()

    # plot AUPR
    precision, recall, _ = precision_recall_curve(y, y_prob)
    avg_precision = average_precision_score(y, y_prob)

    plt.figure(1, figsize=(15, 15))
    plt.plot(
        precision,
        recall,
        lw=2,
        label="PR curve (average precision = %0.2f)" % avg_precision,
    )
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title(f"PR Curve for {title}")
    plt.legend(loc="lower left")
    plt.savefig(join(figs_path, base_name + "_pr.png"))
    plt.clf()
    plt.close()

    # plot SHAP
    # note that SHAP doesn't support model stacking directly
    # so we have to approximate by getting SHAP values for each
    # of the models inside the stack
    if not skip_shap:
        from s2and.model import VotingClassifier  # avoid circular import

        if isinstance(classifier, VotingClassifier):
            shap_values_all = []
            for c in classifier.estimators:
                if isinstance(c, CalibratedClassifierCV):
                    shap_values_all.append(shap.TreeExplainer(c.base_estimator).shap_values(X)[1])
                else:
                    shap_values_all.append(shap.TreeExplainer(c).shap_values(X)[1])
            shap_values = [np.mean(shap_values_all, axis=0)]
        elif nameless_classifier is not None:
            shap_values = []
            for c, d in [(classifier, X), (nameless_classifier, nameless_X)]:
                if isinstance(classifier, CalibratedClassifierCV):
                    shap_values.append(shap.TreeExplainer(c.base_estimator).shap_values(d)[1])
                else:
                    shap_values.append(shap.TreeExplainer(c).shap_values(d)[1])
        elif isinstance(classifier, CalibratedClassifierCV):
            shap_values = shap.TreeExplainer(classifier.base_estimator).shap_values(X)[1]
        else:
            shap_values = shap.TreeExplainer(classifier).shap_values(X)[1]

        if isinstance(shap_values, list):
            for i, (shap_value, feature_names, d) in enumerate(
                zip(
                    shap_values,
                    [shap_feature_names, nameless_feature_names],
                    [X, nameless_X],
                )
            ):
                assert feature_names is not None, "neither feature_names should be None here"
                plt.figure(2 + i)
                shap.summary_plot(
                    shap_value,
                    d,
                    plot_type=shap_plot_type,
                    feature_names=feature_names,
                    show=False,
                    max_display=len(feature_names),
                )
                # plt.title(f"{i}: SHAP Values for {title}")
                plt.tight_layout()
                plt.savefig(join(figs_path, base_name + f"_shap_{i}.png"))
                plt.clf()
                plt.close()
        else:
            plt.figure(2)
            shap.summary_plot(
                shap_values,
                X,
                plot_type=shap_plot_type,
                feature_names=shap_feature_names,
                show=False,
                max_display=len(shap_feature_names),
            )
            # plt.title(f"SHAP Values for {title}")
            plt.tight_layout()
            plt.savefig(join(figs_path, base_name + "_shap.png"))
            plt.clf()
            plt.close()

    # collect metrics and return
    pr, rc, f1, _ = precision_recall_fscore_support(y, y_prob > thresh_for_f1, beta=1.0, average="macro")
    metrics = {
        "AUROC": np.round(roc_auc, 3),
        "Average Precision": np.round(avg_precision, 3),
        "F1": np.round(f1, 3),
        "Precision": np.round(pr, 3),
        "Recall": np.round(rc, 3),
    }

    return metrics


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

    return (
        np.round(precision, 3),
        np.round(recall, 3),
        np.round(f_score, 3),
        per_signature_metrics,
        pred_bigger_ratios,
        true_bigger_ratios,
    )


def cluster_precision_recall_fscore(
    true_clus: Dict[str, List[str]], pred_clus: Dict[str, List[str]]
) -> Tuple[float, float, float]:
    """
    Compute cluster-wise pair-wise precision, recall and F-score.

    The function also contains the fix proposed in
    https://arxiv.org/pdf/1808.04216.pdf to handle singleton clusters.

    Parameters
    ----------
    true_clus: Dict
        dictionary with cluster id as keys and 1d array
        containing the ground-truth signature id assignments as values.
    pred_clus: Dict
        dictionary with cluster id as keys and 1d array
        containing the predicted signature id assignments as values.

    Returns
    -------
    float: calculated precision
    float: calculated recall
    float: calculated F1

    Reference
    ---------
    Levin, Michael, et al. "Citation‐based bootstrapping for
    large‐scale author disambiguation." Journal of the American Society for Information
    Science and Technology (2012): 1030-1047.
    """

    goldpairs = set()
    syspairs = set()

    for _, signatures in true_clus.items():
        if len(signatures) == 1:
            goldpairs.add((signatures[0], signatures[0]))
            continue

        sort_sign = sorted(signatures)

        for i in range(len(sort_sign) - 1):
            for j in range(i + 1, len(sort_sign)):
                goldpairs.add((sort_sign[i], sort_sign[j]))

    for _, signatures in pred_clus.items():
        if len(signatures) == 1:
            syspairs.add((signatures[0], signatures[0]))
            continue

        sort_sign = sorted(signatures)

        for i in range(len(sort_sign) - 1):
            for j in range(i + 1, len(sort_sign)):
                syspairs.add((sort_sign[i], sort_sign[j]))

    precision: float = len(goldpairs.intersection(syspairs)) / len(syspairs)
    recall: float = len(goldpairs.intersection(syspairs)) / len(goldpairs)

    return precision, recall, f1_score(precision, recall)


def pairwise_precision_recall_fscore(true_clus, pred_clus, test_block, strategy="cmacro"):
    """
    Compute the Pairwise precision, recall and F-score.

    Parameters
    ----------
    true_clusters: Dict
        dictionary with cluster id as keys and
        1d array containing the ground-truth signature id assignments as values.
    pred_clusters: Dict
        dictionary with cluster id as keys and
        1d array containing the predicted signature id assignments as values.
    test_block: Dict
        dictionary with block id as keys and 1d array
        containing signature ids as values (block assignment).
    strategy: string
        'clusters' is cluster-wise pairwise precision, recall
        and f1 scores. It is computed over all possible pairs in true and predicted
        clusters. 'cmacro' is computed over each block, and averaged finally.

    Returns
    -------
    float: calculated precision
    float: calculated recall
    float: calculated F1
    """

    true_clusters = true_clus.copy()
    pred_clusters = pred_clus.copy()

    tcset = set(reduce(lambda x, y: x + y, true_clusters.values()))
    pcset = set(reduce(lambda x, y: x + y, pred_clusters.values()))

    if tcset != pcset:
        raise ValueError("predictions do not cover all the signatures.")

    rev_true_clusters = {}
    for k, v in true_clusters.items():
        for vi in v:
            rev_true_clusters[vi] = k

    rev_pred_clusters = {}
    for k, v in pred_clusters.items():
        for vi in v:
            rev_pred_clusters[vi] = k

    if strategy == "clusters":
        precision, recall, f1 = cluster_precision_recall_fscore(true_clus, pred_clus)
        return np.round(precision, 3), np.round(recall, 3), np.round(f1, 3)

    elif strategy == "cmacro":
        mprecision = 0
        mrecall = 0
        mf1 = 0

        for _, signatures in test_block.items():
            gtruth_block = {}
            prediction_block = {}

            for sign in signatures:
                tclus = rev_true_clusters[sign]
                pclus = rev_pred_clusters[sign]
                if tclus not in gtruth_block:
                    gtruth_block[tclus] = list()
                gtruth_block[tclus].append(sign)
                if pclus not in prediction_block:
                    prediction_block[pclus] = list()
                prediction_block[pclus].append(sign)

            _mprecision, _mrecall, _mf1 = cluster_precision_recall_fscore(gtruth_block, prediction_block)

            mprecision += _mprecision
            mrecall += _mrecall
            mf1 += _mf1

        mprecision = mprecision / len(test_block)
        mrecall = mrecall / len(test_block)
        mf1 = mf1 / len(test_block)

        return np.round(mprecision, 3), np.round(mrecall, 3), np.round(mf1, 3)


def claims_eval(
    dataset: "ANDData",
    clusterer: "Clusterer",
    claims_pairs: List[Tuple[str, str, int, str, str]],
    directory_for_caching: Optional[str] = None,
    output_shap: bool = False,
    optional_name: Optional[str] = None,
) -> Dict[str, Union[int, float]]:
    """
    Evaluates predicted clusters on one block of the Semantic Scholar corrections data

    Parameters
    ----------
    dataset: ANDData
        a dataset of the block to evaluate
    clusterer: Clusterer
        the Clusterer to evaluate
    claims_pairs: List
        a list of the claims pairs data to check clusters against
        each pair is (sig id 1, sig id 2, label, block key 1, block key 2)
    directory_for_caching: string
        the directory to write output too, won't write if it is None
    output_shap: bool
        whether to output shaps for the incorrect pairs (slow)
    optional_name: str
        what name to use to write output instead of the featurizer version

    Returns
    -------
    Dict: dictionary of metrics for this block based on claims data
    """
    blocks = dataset.get_blocks()
    preds, dists = clusterer.predict(blocks, dataset)

    all_block_signatures = set()
    for signatures in blocks.values():
        all_block_signatures.update(signatures)

    all_pred_signatures = set()
    for signatures in preds.values():
        all_pred_signatures.update(signatures)

    assert all_block_signatures == all_pred_signatures, "Uh oh, blocks and preds have different signatures"

    signature_to_cluster = {}
    for cluster_key, signatures in preds.items():
        for signature in signatures:
            signature_to_cluster[signature] = cluster_key

    sig_pairs = []
    tp, fp, tn, fn = 0, 0, 0, 0
    for signature_id_1, signature_id_2, label, _, _ in claims_pairs:
        cluster_1 = signature_to_cluster.get(signature_id_1, None)
        cluster_2 = signature_to_cluster.get(signature_id_2, None)
        if cluster_1 is None or cluster_2 is None:
            continue

        same_cluster_pred = cluster_1 == cluster_2
        same_cluster_gold = label

        sig_pairs.append((signature_id_1, signature_id_2, same_cluster_pred, same_cluster_gold))

        if same_cluster_gold and same_cluster_pred:
            tp += 1
        elif same_cluster_gold and not same_cluster_pred:
            fn += 1
        elif not same_cluster_gold and same_cluster_pred:
            fp += 1
        elif not same_cluster_gold and not same_cluster_pred:
            tn += 1

    logger.info("Computing predictions (output_to_write)")
    output_to_write: Dict[str, Any] = {}
    for cluster_key, cluster_signatures in preds.items():
        cluster_output = []
        for signature in cluster_signatures:
            paper_id, _ = signature.split("___")
            paper = dataset.papers[paper_id]
            signature_info = dataset.signatures[signature]
            title = paper.title
            authors = [author.author_name for author in paper.authors]
            affiliations = signature_info.author_info_affiliations
            cluster_output.append((paper_id, signature, title, affiliations, authors))
        output_to_write[cluster_key] = cluster_output

    output_to_write["sig_pairs_wrong"] = []
    output_to_write["sig_pairs_right"] = []
    for id1, id2, pred_same, gold_same in tqdm(sig_pairs):
        paper_id1, _ = id1.split("___")
        paper1 = dataset.papers[paper_id1]
        title1 = paper1.title

        paper_id2, _ = id2.split("___")
        paper2 = dataset.papers[paper_id2]
        title2 = paper2.title

        logger.handlers[0].level = logging.ERROR

        output_to_write["sig_pairs_right" if pred_same == gold_same else "sig_pairs_wrong"].append(
            (id1, id2, title1, title2, pred_same, gold_same)
        )

        if output_shap and directory_for_caching is not None:
            features, _, nameless_features = many_pairs_featurize(
                [(id1, id2, np.nan)],
                dataset,
                clusterer.featurizer_info,
                n_jobs=10,
                use_cache=True,
                chunk_size=100,
                nameless_featurizer_info=clusterer.nameless_featurizer_info,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clusterer.classifier.booster_.params["objective"] = "binary"
                shap_output = shap.TreeExplainer(clusterer.classifier).shap_values(features)[1]
                clusterer.nameless_classifier.booster_.params["objective"] = "binary"  # type: ignore
                shap_output_nameless = shap.TreeExplainer(clusterer.nameless_classifier).shap_values(nameless_features)[
                    1
                ]

                title = f"{id1}-{id2}"
                plt.figure(1)
                shap.summary_plot(
                    shap_output,
                    features,
                    plot_type="dot",
                    feature_names=clusterer.featurizer_info.get_feature_names(),
                    show=False,
                    max_display=len(clusterer.featurizer_info.get_feature_names()),
                )
                plt.title(f"SHAP Values for {title}")
                plt.tight_layout()
                plt.savefig(join(directory_for_caching, f"{title}_shap.png"))
                plt.clf()
                plt.close()

                plt.figure(1)
                shap.summary_plot(
                    shap_output_nameless,
                    nameless_features,
                    plot_type="dot",
                    feature_names=clusterer.nameless_featurizer_info.get_feature_names(),  # type: ignore
                    show=False,
                    max_display=len(clusterer.nameless_featurizer_info.get_feature_names()),  # type: ignore
                )
                plt.title(f"SHAP Values for {title}")
                plt.tight_layout()
                plt.savefig(join(directory_for_caching, f"{title}_shap_nameless.png"))
                plt.clf()
                plt.close()

    logger.handlers[0].level = logging.INFO
    if directory_for_caching is not None:
        logger.info("Writing predictions to disk")
        suffix: Union[int, str]
        if optional_name is None:
            suffix = clusterer.featurizer_info.featurizer_version
        else:
            suffix = optional_name
        with open(join(directory_for_caching, f"preds_{suffix}.json"), "w") as _json_file:
            json.dump(output_to_write, _json_file)

        logger.info("Writing dists to disk")
        with open(join(directory_for_caching, f"dists_{suffix}.pkl"), "wb") as _pkl_file:
            pickle.dump(dists, _pkl_file)
        logger.info("Done dumping")

    precision = tp / (tp + fp) if tp + fp > 0 else np.nan
    recall = tp / (tp + fn) if tp + fn > 0 else np.nan
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0
        else np.nan
    )
    min_edit_score, min_edit_count, number_of_mistaken_ids = min_pair_edit(output_to_write)
    output = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": tp + tn + fp + fn,
        "min_edit_score": min_edit_score,
        "min_edit_count": min_edit_count,
        "number_of_mistaken_ids": number_of_mistaken_ids,
    }
    return output


def min_pair_edit(preds):
    """Find minimum number of cluster changes
    to fully correct a block with errors.

    Args:
        preds: Dictionary that has cluster assignments and claim pairs.

    Returns:
        min_edit_score: Minimum edit distance score from 0 to 1
        min_edit_count: Unnormalized count version of score above.
        number_of_mistaken_ids: Total number of signature ids that were part of wrong pairs
    """
    wrong = preds["sig_pairs_wrong"]
    right = preds["sig_pairs_right"]

    if len(wrong) == 0:
        return 0, 0, 0

    signature_to_cluster = dict()
    for key, value in preds.items():
        if not key.startswith("sig_pairs"):
            for v in value:
                signature_to_cluster[v[1]] = key

    all_clusters = set(list(signature_to_cluster.values()))
    all_clusters.update(["dummy"])

    tp_sigs = set()
    tn_sigs = set()
    for sig_id_1, sig_id_2, title_1, title_2, pred_same, gold_same in wrong + right:
        if gold_same:
            tp_sigs.add((sig_id_1, sig_id_2))
        else:
            tn_sigs.add((sig_id_1, sig_id_2))

    def eval_current_cluster(signature_to_cluster):
        tp, fp, tn, fn = 0, 0, 0, 0
        for s_id_1, s_id_2 in tp_sigs:
            same_cluster_pred = signature_to_cluster[s_id_1] == signature_to_cluster[s_id_2]
            if same_cluster_pred:
                tp += 1
            else:
                fn += 1

        for s_id_1, s_id_2 in tn_sigs:
            same_cluster_pred = signature_to_cluster[s_id_1] == signature_to_cluster[s_id_2]
            if same_cluster_pred:
                fp += 1
            else:
                tn += 1

        return -fp + -fn

    wrong_counts = Counter()
    for sig_id_1, sig_id_2, title_1, title_2, pred_same, gold_same in wrong:
        wrong_counts.update([sig_id_1, sig_id_2])

    worst_ids = [i[0] for i in wrong_counts.most_common(10000000)]

    steps = 0
    for worst_id in worst_ids:
        original_cluster_label = signature_to_cluster[worst_id]
        best_f1 = eval_current_cluster(signature_to_cluster)
        flip_tos = [i for i in all_clusters if i != original_cluster_label]
        best_flip_to = None
        for flip_to in flip_tos:
            signature_to_cluster[worst_id] = flip_to
            current_f1 = eval_current_cluster(signature_to_cluster)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_flip_to = flip_to

        if best_flip_to is not None:
            signature_to_cluster[worst_id] = best_flip_to

            # remake wrong and right
            wrong_new, right_new = [], []
            for sig_id_1, sig_id_2, title_1, title_2, _, gold_same in wrong + right:
                pred_same = signature_to_cluster[sig_id_1] == signature_to_cluster[sig_id_2]
                if pred_same == gold_same:
                    right_new.append([sig_id_1, sig_id_2, title_1, title_2, pred_same, gold_same])
                else:
                    wrong_new.append([sig_id_1, sig_id_2, title_1, title_2, pred_same, gold_same])

            wrong = wrong_new
            right = right_new
            steps += 1
        else:
            signature_to_cluster[worst_id] = original_cluster_label

        if len(wrong) == 0:
            break

    if len(wrong) != 0:
        print("something went wrong")

    return steps / (len(worst_ids) - 1), steps, len(worst_ids)
