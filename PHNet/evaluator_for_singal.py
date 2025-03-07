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

    
def cluster_evaluate(path, method, linkage_method="average", affinity_metric="precomputed"):
    times=0
    result = []
    fname = os.path.basename(path)

    ktrue=[]
    kpre=[]

    f = open(path,'r',encoding = 'utf-8').read()
    text=re.sub(u"&",u" ",f)
    root = ET.fromstring(text)
    correct_labels = []
    papers=[]
    
    mlist = []
    for i in root.findall('publication'):
        correct_labels.append(int(i.find('label').text))
        pid = "i" + i.find('id').text
        mlist.append(pembd[pid])
        papers.append(pid)
        
    t0 = time.perf_counter()
    
    if method=="GHAC_nok": #k is unknown
        labels = GHAC(mlist,papers, linkage_method=linkage_method, affinity_metric=affinity_metric)
    elif method=="GHAC": #k is known
        labels = GHAC(mlist,papers, len(set(correct_labels)),linkage_method=linkage_method, affinity_metric=affinity_metric)
    elif method=="HAC": 
        labels = HAC(mlist,papers, len(set(correct_labels)))
    
    time1 = time.perf_counter()-t0


    pairwise_precision, pairwise_recall, pairwise_f1 = pairwise_evaluate(correct_labels,labels)
    # print (fname,pairwise_precision, pairwise_recall, pairwise_f1)

       
    return pairwise_precision, pairwise_recall, pairwise_f1

if __name__ == "__main__":
    # method = 'GHAC_nok'
    method = 'GHAC'
    #method = 'HAC'
    path = r'raw-data-citeseerx-kim\ychen.xml'
    precision, recall, f1 = cluster_evaluate(path, method)
    print("Final Results - Precision: {}, Recall: {}, F1: {}".format(precision, recall, f1))