# run in "tf-smac" env

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

def GHAC(mlist,papers,n_clusters=-1):
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

            
            model_HAC = AgglomerativeClustering(linkage="average",metric='precomputed',n_clusters=k)
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
        model_HAC = AgglomerativeClustering(linkage="average",metric='precomputed',n_clusters=n_clusters)
        model_HAC.fit(distance)
        labels = model_HAC.labels_
    
    return labels
    
def HAC(mlist,papers,n_clusters):
    distance=[]
    for i in range(len(mlist)):
        tmp=[]
        for j in range(len(mlist)):
            if i<j:
                cosdis=np.dot(mlist[i],mlist[j])/(np.linalg.norm(mlist[i])*(np.linalg.norm(mlist[j])))                              
                tmp.append(cosdis)
            elif i>j:
                tmp.append(distance[j][i])
            else:
                tmp.append(0)
        distance.append(tmp)
    
    distance =np.multiply(distance,-1)
    

    model_HAC = AgglomerativeClustering(linkage="average",metric='precomputed',n_clusters=n_clusters)
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
        precision,
        recall,
        f_score,
        per_signature_metrics,
        pred_bigger_ratios,
        true_bigger_ratios,
    )

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

def cluster_evaluate(method):
    times = 0
    result = []
    path = r'raw-data\\'      # or ‘raw-data-citeseerx-kim’
    file_names = os.listdir(path)
    ktrue = []
    kpre = []

    for fname in file_names:
        f = open(path + fname, 'r', encoding='utf-8').read()
        text = re.sub(u"&", u" ", f)
        root = ET.fromstring(text)
        correct_labels = []
        papers = []
        
        mlist = []
        for i in root.findall('publication'):
            correct_labels.append(int(i.find('label').text))
            pid = "i" + i.find('id').text
            mlist.append(pembd[pid])
            papers.append(pid)
            
        t0 = time.perf_counter()
        
        if method == "GHAC_nok":
            labels = GHAC(mlist, papers)
        elif method == "GHAC":
            labels = GHAC(mlist, papers, len(set(correct_labels)))
        elif method == "HAC":
            labels = HAC(mlist, papers, len(set(correct_labels)))
        
        time1 = time.perf_counter() - t0
        print(time1)
        times = times + time1
        
        correct_labels = np.array(correct_labels)
        labels = np.array(labels)
        print(correct_labels, len(set(correct_labels)))
        print(labels, len(set(labels)))
        ktrue.append(len(set(correct_labels)))
        kpre.append(len(set(labels)))

        # Calculate all metrics
        pairwise_precision, pairwise_recall, pairwise_f1 = pairwise_evaluate(correct_labels, labels)
        k_precision, k_recall, k_f1 = kmetric_evaluate(correct_labels, labels)
        b3_precision, b3_recall, b3_f1, _, _, _ = b3_evaluate(correct_labels, labels)

        print(f"{fname}: Pairwise: {pairwise_precision:.3f}/{pairwise_recall:.3f}/{pairwise_f1:.3f}")
        print(f"{fname}: K-metric: {k_precision:.3f}/{k_recall:.3f}/{k_f1:.3f}")
        print(f"{fname}: B3: {b3_precision:.3f}/{b3_recall:.3f}/{b3_f1:.3f}")

        result.append([fname, 
                      pairwise_precision, pairwise_recall, pairwise_f1,
                      k_precision, k_recall, k_f1,
                      b3_precision, b3_recall, b3_f1])


    
    with open(method + 'DATASET.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["name", 
                        "Pairwise_Prec", "Pairwise_Rec", "Pairwise_F1",
                        "K_Prec", "K_Rec", "K_F1",
                        "B3_Prec", "B3_Rec", "B3_F1",
                        "Actual_K", "Predicted_K"])


        # Write individual results
        for i in range(len(result)):
            row = list(result[i])
            row[0] = row[0][:-4]  # Remove .xml extension
            row.extend([ktrue[i], kpre[i]])
            writer.writerow(row)
            
    print(f"Cluster method: {method}")
    print(f"Average time: {times/len(result):.3f}")
    print(f"MSLE: {mean_squared_log_error(ktrue, kpre):.3f}")
    print(f"Accuracy: {accuracy_score(ktrue, kpre):.3f}")



# method = 'GHAC_nok'
method = 'GHAC'
#method = 'HAC'
    
def main():
    cluster_evaluate(method)
    
if __name__ == "__main__":
    main()