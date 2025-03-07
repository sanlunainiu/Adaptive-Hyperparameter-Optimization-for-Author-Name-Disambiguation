import json
import copy
import argparse
import logging
import pickle
from typing import Dict, Any, Optional, List
from collections import defaultdict
import os
import numpy as np
import pandas as pd

from s2and.data import ANDData
from s2and.featurizer import featurize, FeaturizationInfo
from s2and.model import PairwiseModeler, Clusterer, FastCluster
from s2and.eval import pairwise_eval, cluster_eval, facet_eval
from s2and.eval import cluster_eval_all, cluster_eps_optimize_b3, cluster_eps_optimize_pairwise, cluster_eps_optimize_kmetric 
from s2and.consts import FEATURIZER_VERSION, DEFAULT_CHUNK_SIZE, PROJECT_ROOT_PATH
from s2and.file_cache import cached_path
from s2and.plotting_utils import plot_facets
from hyperopt import hp
from datetime import datetime


# this is the random seed we used for the ablations table
random_seed = 42
# number of cpus to use
n_jobs = 4

# we're going to load the arnetminer dataset
# and assume that you have it already downloaded to the `S2AND/data/` directory
dataset_name = 'arnetminer' # or 'aminer', 'inspire', 'kisti', 'pubmed', 'qian', 'zbmath'

parent_dir = os.getcwd()  # set file path
DATA_DIR = os.path.join(parent_dir, 'data', dataset_name)

anddata = ANDData(
    signatures=os.path.join(DATA_DIR, dataset_name + "_signatures.json"),
    papers=os.path.join(DATA_DIR, dataset_name + "_papers.json"),
    name=dataset_name,
    mode="train",  # can also be 'inference' if just predicting
    specter_embeddings=os.path.join(DATA_DIR, dataset_name + "_specter.pickle"),
    clusters=os.path.join(DATA_DIR, dataset_name + "_clusters.json"),
    block_type="s2",  # can also be 'original'
    train_pairs=None,  # in case you have predefined splits for the pairwise models
    val_pairs=None,
    test_pairs=None,
    train_pairs_size=100000,  # how many training pairs for the pairwise models?
    val_pairs_size=10000,
    test_pairs_size=10000,
    n_jobs=n_jobs,
    load_name_counts=True,  # the name counts derived from the entire S2 corpus need to be loaded separately
    preprocess=True,
    random_seed=random_seed,
)

# to train the pairwise model, we define which feature categories to use
# here it is all of them
features_to_use = [
    "name_similarity",
    "affiliation_similarity",
    "email_similarity",
    "coauthor_similarity",
    "venue_similarity",
    "year_diff",
    "title_similarity",
    "reference_features",
    "misc_features",
    "name_counts",
    "embedding_similarity",
    "journal_similarity",
    "advanced_name_similarity",
]

# we also have this special second "nameless" model that doesn't use any name-based features
# it helps to improve clustering performance by preventing model overreliance on names
nameless_features_to_use = [
    feature_name
    for feature_name in features_to_use
    if feature_name not in {"name_similarity", "advanced_name_similarity", "name_counts"}
]

# we store all the information about the features in this convenient wrapper
featurization_info = FeaturizationInfo(features_to_use=features_to_use, featurizer_version=FEATURIZER_VERSION)
nameless_featurization_info = FeaturizationInfo(features_to_use=nameless_features_to_use, featurizer_version=FEATURIZER_VERSION)

# now we can actually go and get the pairwise training, val and test data
train, val, test = featurize(anddata, featurization_info, n_jobs=1, use_cache=False, chunk_size=DEFAULT_CHUNK_SIZE, nameless_featurizer_info=nameless_featurization_info, nan_value=np.nan)  # type: ignore
X_train, y_train, nameless_X_train = train
X_val, y_val, nameless_X_val = val
X_test, y_test, nameless_X_test = test

# now we define and fit the pairwise modelers
pairwise_modeler = PairwiseModeler(
    n_iter=25,  # number of hyperparameter search iterations
    estimator=None,  # this will use the default LightGBM classifier
    search_space=None,  # this will use the default LightGBM search space
    monotone_constraints=featurization_info.lightgbm_monotone_constraints,  # we use monotonicity constraints to make the model more sensible
    random_state=random_seed,
)
pairwise_modeler.fit(X_train, y_train, X_val, y_val)

# as mentioned above, there are 2: one with all features and a nameless one
nameless_pairwise_modeler = PairwiseModeler(
    n_iter=25,
    estimator=None,
    search_space=None,
    monotone_constraints=nameless_featurization_info.lightgbm_monotone_constraints,
    random_state=random_seed,
)
nameless_pairwise_modeler.fit(nameless_X_train, y_train, nameless_X_val, y_val)

# now we can fit the clusterer itself
clusterer = Clusterer(
    featurization_info,
    pairwise_modeler.classifier,  # the actual pairwise classifier
    cluster_model=FastCluster(linkage='average'),  # average linkage agglomerative clustering
    search_space={"eps": hp.uniform("choice", 0, 1)},  # the hyperparemetrs for the clustering algorithm
    n_jobs=1,
    use_cache=False,
    nameless_classifier=nameless_pairwise_modeler.classifier,  # the nameless pairwise classifier
    nameless_featurizer_info=nameless_featurization_info,
    random_state=random_seed,
    use_default_constraints_as_supervision=False,  # this is an option used by the S2 production system but not in the S2AND paper
)
clusterer.fit(anddata)

# but how good are our models? 
# first, let's look at the quality of the pairwise evaluation
pairwise_metrics = pairwise_eval(
    X_test,
    y_test,
    pairwise_modeler,
    os.path.join(PROJECT_ROOT_PATH, "data", "tutorial_figures"),  # where to put the figures
    "tutorial_figures",  # what to call the figures
    featurization_info.get_feature_names(),
    nameless_classifier=nameless_pairwise_modeler,
    nameless_X=nameless_X_test,
    nameless_feature_names=nameless_featurization_info.get_feature_names(),
    skip_shap=False,  # if your model isn't a tree-based model, you should put True here and it will not make SHAP figures
)
print(pairwise_metrics)

def optimize_and_save_results(optimization_type, anddata, clusterer, dataset_name, parent_dir, n_iter=200):
    """
    Performs optimization and saves the results.
    
    Parameters
    ----------
    optimization_type : str
        Type of optimization ('b3', 'pairwise_cmacro', 'pairwise_clusters', 'k_metric')
    anddata : ANDData
        Dataset object
    clusterer : Clusterer
        Clusterer object
    dataset_name : str
        Name of the dataset
    parent_dir : str
        Parent directory for the output
    n_iter : int, optional
        Number of optimization iterations (default: 200)
    """
    
    if optimization_type == 'b3':
        metrics = cluster_eps_optimize_b3(
            dataset=anddata,
            clusterer=clusterer,
            split="test",
            n_iter=n_iter
        )
        metric_prefix = "B3"
    elif optimization_type == 'pairwise_cmacro':
        metrics = cluster_eps_optimize_pairwise(
            dataset=anddata,
            clusterer=clusterer,
            split="test",
            n_iter=n_iter,
            optimize_method=optimization_type
        )
        metric_prefix = "PCM"
    elif optimization_type == 'pairwise_clusters':
        metrics = cluster_eps_optimize_pairwise(
            dataset=anddata,
            clusterer=clusterer,
            split="test",
            n_iter=n_iter,
            optimize_method=optimization_type
        )
        metric_prefix = "PC"
    elif optimization_type == 'k_metric':
        metrics = cluster_eps_optimize_kmetric(
            dataset=anddata,
            clusterer=clusterer,
            split="test",
            n_iter=n_iter,
            optimize_method="k_metric"
        )
        metric_prefix = "KM"
    else:
        raise ValueError(f"Unknown optimization type: {optimization_type}")


    output_dir = os.path.join(parent_dir, "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)

    # get block dict
    train_block_dict, val_block_dict, test_block_dict = anddata.split_blocks_helper(anddata.get_blocks())
    block_dict = {**train_block_dict, **val_block_dict, **test_block_dict}


    data_rows = []
    for block_name, block_metrics in metrics.items():
        # get the true number of clusters
        current_block_dict = {block_name: block_dict[block_name]}
        true_clusters = anddata.construct_cluster_to_signatures(current_block_dict)
        
        row = {
            'Block_Name': block_name,
            'Signatures_Count': len(block_dict[block_name]),
            'True_Clusters': len(true_clusters) if true_clusters else 0,
            'Optimal_Eps': round(block_metrics['eps'], 4),
        }
        
        # add specific indicators based on optimization type
        if optimization_type == 'k_metric':
            row.update({
                f'{metric_prefix}_Best_K': round(block_metrics['k'], 4),
                f'{metric_prefix}_Best_ACP': round(block_metrics['acp'], 4),
                f'{metric_prefix}_Best_AAP': round(block_metrics['aap'], 4)
            })
        else:
            row.update({
                f'{metric_prefix}_Best_Precision': round(block_metrics['precision'], 4),
                f'{metric_prefix}_Best_Recall': round(block_metrics['recall'], 4),
                f'{metric_prefix}_Best_F1': round(block_metrics['f1'], 4)
            })
        
        data_rows.append(row)


    df = pd.DataFrame(data_rows)


    if optimization_type == 'k_metric':
        columns_order = ['Block_Name', 'Signatures_Count', 'True_Clusters', 
                        'Optimal_Eps', f'{metric_prefix}_Best_K', 
                        f'{metric_prefix}_Best_ACP', f'{metric_prefix}_Best_AAP']
    else:
        columns_order = ['Block_Name', 'Signatures_Count', 'True_Clusters',
                        'Optimal_Eps', f'{metric_prefix}_Best_Precision', 
                        f'{metric_prefix}_Best_Recall', f'{metric_prefix}_Best_F1']
    df = df[columns_order]

    # save result
    output_file = os.path.join(output_dir, f"gggblock_metrics_opt_{dataset_name}_{optimization_type}.csv")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    # print(f"\nResults saved to {output_file}")
    # print("\nSummary statistics:")
    # print(f"Total number of blocks: {len(df)}")
    # print(f"Average Signatures per Block: {df['Signatures_Count'].mean():.2f}")
    # print(f"Average True Clusters per Block: {df['True_Clusters'].mean():.2f}")
    # print(f"Average Optimal Eps: {df['Optimal_Eps'].mean():.4f}")

    # if optimization_type == 'k_metric':
    #     print(f"Average Best K: {df[f'{metric_prefix}_Best_K'].mean():.4f}")
    #     print(f"Average Best ACP: {df[f'{metric_prefix}_Best_ACP'].mean():.4f}")
    #     print(f"Average Best AAP: {df[f'{metric_prefix}_Best_AAP'].mean():.4f}")
    # else:
    #     print(f"Average Precision: {df[f'{metric_prefix}_Best_Precision'].mean():.4f}")
    #     print(f"Average Recall: {df[f'{metric_prefix}_Best_Recall'].mean():.4f}")
    #     print(f"Average F1: {df[f'{metric_prefix}_Best_F1'].mean():.4f}")

    return df, metrics


# B3 optimization
b3_df, b3_metrics = optimize_and_save_results('b3', anddata, clusterer,dataset_name, parent_dir)

# Pairwise CMacro optimization
# pcm_df, pcm_metrics = optimize_and_save_results('pairwise_cmacro', anddata, clusterer,dataset_name, parent_dir)

# Pairwise Clusters optimization
# pc_df, pc_metrics = optimize_and_save_results('pairwise_clusters', anddata, clusterer,dataset_name, parent_dir)

# K-Metric optimization
# km_df, km_metrics = optimize_and_save_results('k_metric', anddata, clusterer,dataset_name, parent_dir)


