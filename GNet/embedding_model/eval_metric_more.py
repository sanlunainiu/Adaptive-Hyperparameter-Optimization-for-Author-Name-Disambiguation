from enum import Enum
import numpy as np
from sklearn.cluster import *
import hdbscan
from xmeans import XMeans
from utility import construct_doc_matrix

class ClusterMethod(Enum):
    HDBSCAN = 'hdbscan'
    DBSCAN = 'dbscan'
    AP = 'ap'
    XMEANS = 'xmeans'
    MEANSHIFT = 'meanshift'
    KMEANS = 'kmeans'
    HAC = 'hac'

class EvalMetric(Enum):
    CLUSTER = 'cluster'
    KMETRIC = 'kmetric'
    B3 = 'b3'
    ALL = 'all'

class ClusteringMethods:
    @staticmethod
    def hdbscan_cluster(D_matrix, **kwargs):
        return hdbscan.HDBSCAN(min_cluster_size=1).fit_predict(D_matrix)
    
    @staticmethod
    def dbscan_cluster(D_matrix, **kwargs):
        return DBSCAN(eps=1.5, min_samples=2).fit_predict(D_matrix)
    
    @staticmethod
    def ap_cluster(D_matrix, **kwargs):
        return AffinityPropagation(damping=0.6).fit_predict(D_matrix)
    
    @staticmethod
    def xmeans_cluster(D_matrix, **kwargs):
        return XMeans(random_state=1).fit_predict(D_matrix)
    
    @staticmethod
    def meanshift_cluster(D_matrix, **kwargs):
        return MeanShift().fit_predict(D_matrix)
    
    @staticmethod
    def kmeans_cluster(D_matrix, **kwargs):
        true_cluster_size = kwargs.get('true_cluster_size')
        return KMeans(n_clusters=true_cluster_size, init="k-means++").fit_predict(D_matrix)
    
    @staticmethod
    def hac_cluster(D_matrix, **kwargs):
        true_cluster_size = kwargs.get('true_cluster_size')
        return AgglomerativeClustering(
            n_clusters=true_cluster_size,
            linkage="average",
            affinity="cosine"
        ).fit_predict(D_matrix)

class Evaluator:
    @staticmethod
    def compute_f1(dataset, bpr_optimizer, cluster_method='dbscan', eval_metric='all'):

        try:
            cluster_method = ClusterMethod(cluster_method.lower())
            eval_metric = EvalMetric(eval_metric.lower())
        except ValueError:
            raise ValueError(
                f"Invalid method. cluster_method must be one of {[e.value for e in ClusterMethod]}, "
                f"eval_metric must be one of {[e.value for e in EvalMetric]}"
            )
        
        D_matrix = construct_doc_matrix(bpr_optimizer.paper_latent_matrix, dataset.paper_list)
        true_cluster_size = len(set(dataset.label_list))
        
        clustering_methods = {
            ClusterMethod.HDBSCAN: ClusteringMethods.hdbscan_cluster,
            ClusterMethod.DBSCAN: ClusteringMethods.dbscan_cluster,
            ClusterMethod.AP: ClusteringMethods.ap_cluster,
            ClusterMethod.XMEANS: ClusteringMethods.xmeans_cluster,
            ClusterMethod.MEANSHIFT: ClusteringMethods.meanshift_cluster,
            ClusterMethod.KMEANS: ClusteringMethods.kmeans_cluster,
            ClusterMethod.HAC: ClusteringMethods.hac_cluster
        }
        
        cluster_func = clustering_methods[cluster_method]
        y_pred = cluster_func(D_matrix, true_cluster_size=true_cluster_size)
        
        true_label_dict = {}
        predict_label_dict = {}
        
        for idx, true_lbl in enumerate(dataset.label_list):
            if true_lbl not in true_label_dict:
                true_label_dict[true_lbl] = [idx]
            else:
                true_label_dict[true_lbl].append(idx)
                
        for idx, pred_lbl in enumerate(y_pred):
            if pred_lbl not in predict_label_dict:
                predict_label_dict[pred_lbl] = [idx]
            else:
                predict_label_dict[pred_lbl].append(idx)


        if eval_metric == EvalMetric.ALL:
            results = {}
            results['cluster'] = Evaluator._compute_cluster_metrics(true_label_dict, predict_label_dict)
            results['kmetric'] = Evaluator._compute_kmetric(true_label_dict, predict_label_dict)
            results['b3'] = Evaluator._compute_b3(true_label_dict, predict_label_dict)
            return results
        else:
            if eval_metric == EvalMetric.CLUSTER:
                return Evaluator._compute_cluster_metrics(true_label_dict, predict_label_dict)
            elif eval_metric == EvalMetric.KMETRIC:
                return Evaluator._compute_kmetric(true_label_dict, predict_label_dict)
            elif eval_metric == EvalMetric.B3:
                return Evaluator._compute_b3(true_label_dict, predict_label_dict)
            
    @staticmethod
    def _compute_cluster_metrics(true_label_dict, predict_label_dict):

        r_k_table = []
        for v1 in predict_label_dict.values():
            k_list = []
            for v2 in true_label_dict.values():
                N_ij = len(set(v1).intersection(v2))
                k_list.append(N_ij)
            r_k_table.append(k_list)
            
        r_k_matrix = np.array(r_k_table)
        r_num = int(r_k_matrix.shape[0])
        
        sum_f1 = sum_pre = sum_rec = 0.0
        for row in range(r_num):
            row_sum = np.sum(r_k_matrix[row,:])
            if row_sum != 0:
                max_col_index = np.argmax(r_k_matrix[row,:])
                row_max_value = r_k_matrix[row, max_col_index]
                prec = float(row_max_value) / row_sum
                col_sum = np.sum(r_k_matrix[:, max_col_index])
                rec = float(row_max_value) / col_sum
                row_f1 = float(2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
                sum_f1 += row_f1
                sum_pre += prec
                sum_rec += rec
        
        return float(sum_f1) / r_num, float(sum_pre) / r_num, float(sum_rec) / r_num


    @staticmethod
    def _compute_kmetric(true_label_dict, predict_label_dict):

        signature_to_pred_cluster = {}
        pred_cluster_sizes = {}
        
        for cluster_id, signatures in predict_label_dict.items():
            pred_cluster_sizes[cluster_id] = len(signatures)
            for signature in signatures:
                signature_to_pred_cluster[signature] = cluster_id
                
        aap_sum = acp_sum = 0  
        total_signatures = 0
        
        for true_cluster_id, true_signatures in true_label_dict.items():
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
        f_score = np.sqrt(recall * precision) if recall * precision > 0 else 0
        
        return f_score, precision, recall


    @staticmethod
    def _compute_b3(true_label_dict, predict_label_dict):

        true_clusters = {k: frozenset(v) for k, v in true_label_dict.items()}
        pred_clusters = {k: frozenset(v) for k, v in predict_label_dict.items()}
        
        tcset = set([item for sublist in true_label_dict.values() for item in sublist])
        
        rev_true_clusters = {}
        for k, v in true_clusters.items():
            for vi in v:
                rev_true_clusters[vi] = k

        rev_pred_clusters = {}
        for k, v in pred_clusters.items():
            for vi in v:
                rev_pred_clusters[vi] = k

        precision = recall = 0.0
        intersections = {}
        n_samples = len(tcset)
        
        for item in tcset:
            pred_cluster_i = pred_clusters[rev_pred_clusters[item]]
            true_cluster_i = true_clusters[rev_true_clusters[item]]
            
            if (pred_cluster_i, true_cluster_i) in intersections:
                intersection = intersections[(pred_cluster_i, true_cluster_i)]
            else:
                intersection = pred_cluster_i.intersection(true_cluster_i)
                intersections[(pred_cluster_i, true_cluster_i)] = intersection
                
            _precision = len(intersection) / len(pred_cluster_i)
            _recall = len(intersection) / len(true_cluster_i)
            precision += _precision
            recall += _recall
        
        precision /= n_samples
        recall /= n_samples
        f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return f_score, precision, recall

