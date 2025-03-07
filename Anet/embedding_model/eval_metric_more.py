import numpy as np
from sklearn.cluster import AgglomerativeClustering
import math
from functools import reduce
from enum import Enum
from utility import construct_doc_matrix

class EvalMode(Enum):
    CLUSTER = 'cluster'
    KMETRIC = 'kmetric'
    B3 = 'b3'
    ALL = 'all'

class Evaluator():
    @staticmethod 
    def compute_f1(dataset, bpr_optimizer, value_mode='all'):
        
        try:
            mode = EvalMode(value_mode.lower())
        except ValueError:
            raise ValueError(f"Invalid value_mode: {value_mode}. Must be one of {[e.value for e in EvalMode]}")

        D_matrix = construct_doc_matrix(bpr_optimizer.paper_latent_matrix,
                                      dataset.paper_list)
        true_cluster_size = len(set(dataset.label_list))
        y_pred = AgglomerativeClustering(n_clusters=true_cluster_size,
                                       linkage="average",
                                       metric="cosine").fit_predict(D_matrix)
        
        true_clusters = {}
        pred_clusters = {}
        
        for idx, true_lbl in enumerate(dataset.label_list):
            if true_lbl not in true_clusters:
                true_clusters[true_lbl] = [str(idx)]
            else:
                true_clusters[true_lbl].append(str(idx))
                
        for idx, pred_lbl in enumerate(y_pred):
            if pred_lbl not in pred_clusters:
                pred_clusters[pred_lbl] = [str(idx)]
            else:
                pred_clusters[pred_lbl].append(str(idx))

        if mode == EvalMode.CLUSTER:
            return Evaluator._compute_cluster_f1(true_clusters, pred_clusters)
        elif mode == EvalMode.KMETRIC:
            return Evaluator._compute_kmetric(true_clusters, pred_clusters)
        elif mode == EvalMode.B3:
            return Evaluator._compute_b3(true_clusters, pred_clusters)
        else:  # ALL
            return {
                "cluster": Evaluator._compute_cluster_f1(true_clusters, pred_clusters),
                "kmetric": Evaluator._compute_kmetric(true_clusters, pred_clusters),
                "b3": Evaluator._compute_b3(true_clusters, pred_clusters)
            }

    @staticmethod
    def _compute_cluster_f1(true_clusters, pred_clusters):
        r_k_table = []
        for v1 in pred_clusters.values():
            k_list = []
            for v2 in true_clusters.values():
                N_ij = len(set(v1).intersection(v2))
                k_list.append(N_ij)
            r_k_table.append(k_list)
            
        r_k_matrix = np.array(r_k_table)
        r_num = int(r_k_matrix.shape[0])
        
        sum_f1 = sum_prec = sum_rec = 0.0
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
                sum_prec += prec
                sum_rec += rec
        
        return (float(sum_f1) / r_num, 
                float(sum_prec) / r_num,
                float(sum_rec) / r_num)

    @staticmethod
    def _compute_kmetric(true_clusters, pred_clusters):
        signature_to_pred_cluster = {}
        pred_cluster_sizes = {}
        
        for cluster_id, signatures in pred_clusters.items():
            pred_cluster_sizes[cluster_id] = len(signatures)
            for signature in signatures:
                signature_to_pred_cluster[signature] = cluster_id
                
        aap_sum = acp_sum = 0  
        total_signatures = 0
        
        for true_cluster_id, true_signatures in true_clusters.items():
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
        f_score = math.sqrt(recall * precision) if recall * precision > 0 else 0
        
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