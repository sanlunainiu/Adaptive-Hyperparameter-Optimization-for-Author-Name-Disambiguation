from utils.cache import LMDBClient
from utils import data_utils
from utils import settings
import numpy as np
from local.gae.input_data_gpu import load_local_data
import networkx as nx
from local.gae.preprocessing_gpu import normalize_vectors, gen_train_edges,preprocess_graph
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist

def calculate_norm_ratio(embs_input, high_percentile=90, low_percentile=50):
    norms = [np.linalg.norm(vec) for vec in embs_input]
    high_value = np.percentile(norms, high_percentile)
    low_value = np.percentile(norms, low_percentile)
    return high_value / low_value if low_value > 0 else np.nan

def calculate_tail_integral(data, percentile=90):
    high_percentile_value = np.percentile(data, percentile)
    tail_data = data[data > high_percentile_value]
    tail_integral = np.sum(tail_data) - high_percentile_value * len(tail_data)
    return tail_integral

def calculate_tail_mass(data, percentile_num=99):
    percentile = np.percentile(data, percentile_num)
    tail_mass = np.mean(data > percentile)
    return  tail_mass

def calculate_tail_weight(data, threshold_multiplier=3):
    mean_value = np.mean(data)
    return np.sum(data > threshold_multiplier * mean_value) / len(data)

def generate_training_features():
    INTER_LMDB_NAME = 'pub_authors.feature'
    lc_input = LMDBClient(INTER_LMDB_NAME)
    
    LMDB_NAME_EMB = "author_triplets.emb"
    lc_emb = LMDBClient(LMDB_NAME_EMB)
    
    idf = data_utils.load_data(settings.GLOBAL_DATA_DIR, 'feature_idf.pkl')
    
    name_to_pubs_test = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_test_100.json')
    
    author_features = {}
    
    for name, name_data in name_to_pubs_test.items():
        # print('Processing:', name)
        embs_input = []
        feature_counts = []
        feature_idf_values = []
        valid_cluster_count = 0
        valid_total_patent_count = 0 
        
        for aid, pids in name_data.items():
            if len(pids) < 5:  
                continue
            valid_cluster_count += 1
            valid_total_patent_count += len(pids)
            for pid in pids:
                cur_emb = lc_input.get(pid)
                if cur_emb is None:
                    continue
                cur_feature_idf_values = [idf.get(feature, 0) for feature in cur_emb]
                
                embs_input.append(lc_emb.get(pid))
                feature_counts.append(len(cur_emb))
                feature_idf_values.extend(cur_feature_idf_values)
                
                
        if embs_input:  
            # for Pairwise F1
            # norm_ratio = calculate_norm_ratio(embs_input, 90, 50)
            # for k_metric
            # norm_ratio = calculate_norm_ratio(embs_input, 85, 50)
            # for b3
            norm_ratio = calculate_norm_ratio(embs_input, 90, 50)

            avg_feature_count = np.mean(feature_counts) 
            avg_feature_idf = np.mean(feature_idf_values)  
            
            median_feature_count = np.median(feature_counts)
            
            # for Pairwise F1
            # feature_count_percentile_70 = np.percentile(feature_counts, 70)
            # for k_metric
            # feature_count_percentile_70 = np.percentile(feature_counts, 85)
            # for b3
            feature_count_percentile_70 = np.percentile(feature_counts, 90)
            
            median_idf = np.median(feature_idf_values)
            
            # for Pairwise F1
            # idf_percentile_70 = np.percentile(feature_idf_values, 70)
            # for k_metric
            # idf_percentile_70 = np.percentile(feature_idf_values, 80)
            # for b3
            idf_percentile_70 = np.percentile(feature_idf_values, 70)
            
            feature_counts_tail_weight = calculate_tail_weight(feature_counts, 3)
            idf_tail_weight = calculate_tail_weight(feature_idf_values, 3)

            tail_mass_99_counts = calculate_tail_mass(feature_counts)
            tail_mass_99_idf = calculate_tail_mass(feature_idf_values)
    
            feature_counts_array = np.array(feature_counts)
            
            # for Pairwise F1
            # tail_integral_feature_counts = calculate_tail_integral(feature_counts_array, 90)
            # for k_metric
            # tail_integral_feature_counts = calculate_tail_integral(feature_counts_array, 85)
            # for b3
            tail_integral_feature_counts = calculate_tail_integral(feature_counts_array, 90)

            feature_idf_values_array = np.array(feature_idf_values)
            
            # for Pairwise F1
            # tail_integral_feature_idf_values = calculate_tail_integral(feature_idf_values_array, 90)           
            # for k_metric
            tail_integral_feature_idf_values = calculate_tail_integral(feature_idf_values_array, 90) 
            # for b3
            # tail_integral_feature_idf_values = calculate_tail_integral(feature_idf_values_array, 60) 
            
            ratio_patents_per_cluster = valid_total_patent_count / valid_cluster_count if valid_cluster_count else 0
            ratio_clusters_per_patent = valid_cluster_count / valid_total_patent_count if valid_total_patent_count else 0
            ratio_excess_patents_per_cluster = (valid_total_patent_count - valid_cluster_count) / valid_cluster_count if valid_cluster_count else 0
            ratio_avg_idf_per_cluster = avg_feature_idf / valid_cluster_count if valid_cluster_count else 0
            ratio_avg_features_per_cluster = avg_feature_count / valid_cluster_count if valid_cluster_count else 0

            author_features[name] = {
                'norm_ratio': norm_ratio,
                'avg_feature_count': avg_feature_count,
                'avg_feature_idf': avg_feature_idf,
                'valid_cluster_count': valid_cluster_count,
                'median_feature_count': median_feature_count,
                'feature_count_percentile_70': feature_count_percentile_70,
                'median_idf': median_idf,
                'idf_percentile_70': idf_percentile_70,
                'feature_counts_tail_weight': feature_counts_tail_weight,   
                'tail_mass_99_idf': tail_mass_99_idf,   
                'tail_integral_feature_counts': tail_integral_feature_counts,   
                'tail_integral_feature_idf_values': tail_integral_feature_idf_values,   
                'ratio_clusters_per_patent': ratio_clusters_per_patent,
                'ratio_excess_patents_per_cluster': ratio_excess_patents_per_cluster,
                'ratio_avg_idf_per_cluster': ratio_avg_idf_per_cluster,
                'ratio_avg_features_per_cluster': ratio_avg_features_per_cluster
            }


    return author_features
        
if __name__ == '__main__':
    author_features = generate_training_features()
