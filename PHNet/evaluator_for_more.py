import pickle
import csv
import numpy as np
import os
import re
import  xml.dom.minidom
import xml.etree.ElementTree as ET
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster
import time
import networkx as nx
import community
from sklearn.metrics import mean_squared_log_error,accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.csgraph import connected_components
import sys
import community as community_louvain


# for Arnetminer dataset
with open (r"C:\Users\Jason Burne\Desktop\qiao_2019_bigdata\gene\PHNet.pkl",'rb') as file:
    PHNet = pickle.load(file)
with open(r'C:\Users\Jason Burne\Desktop\qiao_2019_bigdata\final_emb\pemb_final.pkl', "rb") as file_obj:  
    pembd = pickle.load(file_obj) 
 
# for citeseerx-kim dataset
# with open (r"C:\Users\Jason Burne\Desktop\qiao_2019_bigdata\gene-citeseerx-kim\PHNet.pkl",'rb') as file:
#     PHNet = pickle.load(file)
# with open(r'C:\Users\Jason Burne\Desktop\qiao_2019_bigdata\final_emb-citeseerx-kim\pemb_final.pkl', "rb") as file_obj:  
#     pembd = pickle.load(file_obj) 


def pairwise_evaluate(correct_labels,pred_labels):
    TP = 0.0  # Pairs Correctly Predicted To SameAuthor
    TP_FP = 0.0  # Total Pairs Predicted To SameAuthor
    TP_FN = 0.0  # Total Pairs To SameAuthor

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

def GHAC(mlist,papers, n_clusters=-1, linkage_method="average", affinity_metric="precomputed"):
    paper_weight = np.array(PHNet.loc[papers][papers])
        
    distance=[]
    graph=[]
    
    for i in range(len(mlist)):
        gtmp=[]
        for j in range(len(mlist)):
            if i<j and paper_weight[i][j]!=0:
                cosdis=np.dot(mlist[i],mlist[j])/(np.linalg.norm(mlist[i])*(np.linalg.norm(mlist[j])))                              
                gtmp.append(cosdis*paper_weight[i][j])
            elif i>j:
                gtmp.append(graph[j][i])
            else:
                gtmp.append(0)
        graph.append(gtmp)
    
    distance =np.multiply(graph,-1)

    
    if n_clusters==-1:
        best_m=-10000000
        graph=np.array(graph)
        n_components1, labels = connected_components(graph)
        
        graph[graph<=0.5]=0
        G=nx.from_numpy_matrix(graph)
         
        n_components, labels = connected_components(graph)
        
        for k in range(n_components,n_components1-1,-1):

            
            model_HAC = AgglomerativeClustering(linkage=linkage_method, metric=affinity_metric, n_clusters=k)
            model_HAC.fit(distance)
            labels = model_HAC.labels_
            
            part= {}
            for j in range (len(labels)):
                part[j]=labels[j]

            # mod = community.modularity(part,G)
            mod = community_louvain.modularity(part, G)

            if mod>best_m:
                best_m=mod
                best_labels=labels
        labels = best_labels
    else:
        model_HAC = AgglomerativeClustering(linkage=linkage_method, metric=affinity_metric, n_clusters=n_clusters)
        model_HAC.fit(distance)
        labels = model_HAC.labels_
    
    return labels

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
    metrics based on formal constraints." Information Retrieval 12.4
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

import math
from functools import reduce

def kmetric_evaluate(correct_labels, pred_labels):
    cluster_to_signatures_true = {}
    cluster_to_signatures_pred = {}
    
    for i, label in enumerate(correct_labels):
        if label not in cluster_to_signatures_true:
            cluster_to_signatures_true[label] = []
        cluster_to_signatures_true[label].append(str(i))
    
    for i, label in enumerate(pred_labels):
        if label not in cluster_to_signatures_pred:
            cluster_to_signatures_pred[label] = []
        cluster_to_signatures_pred[label].append(str(i))
    
    return kmetric_precision_recall_fscore(cluster_to_signatures_true, cluster_to_signatures_pred)

def b3_evaluate(correct_labels, pred_labels):
    true_clus = {}
    pred_clus = {}
    
    for i, label in enumerate(correct_labels):
        if label not in true_clus:
            true_clus[label] = []
        true_clus[label].append(str(i))
    
    for i, label in enumerate(pred_labels):
        if label not in pred_clus:
            pred_clus[label] = []
        pred_clus[label].append(str(i))
    
    return b3_precision_recall_fscore(true_clus, pred_clus, skip_signatures=None)

def f1_score(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def cluster_evaluate(path, method="GHAC", n_clusters=None, linkage_method="average", 
                    affinity_metric="precomputed", metrics=None, encoding='utf-8',
                    return_labels=False):
    """
    Enhanced cluster evaluation function with more flexibility
    
    Parameters:
    -----------
    path : str
        Path to the XML file containing publication data
    method : str, default="GHAC"
        Clustering method to use ("GHAC", "GHAC_nok", "HAC")
    n_clusters : int, optional
        Number of clusters. If None, will be determined from data
    linkage_method : str, default="average"
        Linkage method for hierarchical clustering
    affinity_metric : str, default="precomputed"
        Metric used for calculating distances
    metrics : list, optional
        List of metrics to evaluate. Options: ['pairwise', 'k_metric', 'b3']
    encoding : str, default='utf-8'
        Encoding of the input XML file
    return_labels : bool, default=False
        If True, return predicted labels along with metrics
        
    Returns:
    --------
    dict : Dictionary containing evaluation results and optional predicted labels
    """
    if metrics is None:
        metrics = ['pairwise', 'k_metric', 'b3']
    
    # Read and parse XML file
    try:
        with open(path, 'r', encoding=encoding) as f:
            text = re.sub(u"&", u" ", f.read())
        root = ET.fromstring(text)
    except Exception as e:
        raise Exception(f"Error reading file {path}: {str(e)}")

    # Extract data
    correct_labels = []
    papers = []
    mlist = []
    
    for i in root.findall('publication'):
        correct_labels.append(int(i.find('label').text))
        pid = "i" + i.find('id').text
        mlist.append(pembd[pid])
        papers.append(pid)
    
    # Determine number of clusters if not provided
    if n_clusters is None and method != "GHAC_nok":
        n_clusters = len(set(correct_labels))
    
    # Perform clustering
    t0 = time.perf_counter()
    
    if method == "GHAC_nok":
        labels = GHAC(mlist, papers, linkage_method=linkage_method, 
                     affinity_metric=affinity_metric)
    elif method == "GHAC":
        labels = GHAC(mlist, papers, n_clusters, linkage_method=linkage_method, 
                     affinity_metric=affinity_metric)
    elif method == "HAC":
        labels = HAC(mlist, papers, n_clusters)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    clustering_time = time.perf_counter() - t0
    
    # Calculate requested metrics
    results = {'time': clustering_time}
    
    if 'pairwise' in metrics:
        results['pairwise'] = pairwise_evaluate(correct_labels, labels)
    
    if 'k_metric' in metrics:
        results['k_metric'] = kmetric_evaluate(correct_labels, labels)
    
    if 'b3' in metrics:
        results['b3'] = b3_evaluate(correct_labels, labels)
    
    # Add clustering information
    results['clustering_info'] = {
        'n_clusters_true': len(set(correct_labels)),
        'n_clusters_pred': len(set(labels)),
        'n_samples': len(labels)
    }
    
    if return_labels:
        results['predicted_labels'] = labels
        results['true_labels'] = correct_labels
    
    return results

if __name__ == "__main__":
    path = r'raw-data-citeseerx-kim\ychen.xml'
    results = cluster_evaluate(path, method='GHAC')
    
    # Print results
    if 'pairwise' in results:
        print("Pairwise metrics - Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}".format(
            *results['pairwise']))
    if 'k_metric' in results:
        print("K-metric - Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}".format(
            *results['k_metric']))
    if 'b3' in results:
        print("B3 metrics - Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}".format(
            *results['b3']))
    
    print("\nClustering Information:")
    print(f"True clusters: {results['clustering_info']['n_clusters_true']}")
    print(f"Predicted clusters: {results['clustering_info']['n_clusters_pred']}")
    print(f"Number of samples: {results['clustering_info']['n_samples']}")
    print(f"Time taken: {results['time']:.3f}s")
    