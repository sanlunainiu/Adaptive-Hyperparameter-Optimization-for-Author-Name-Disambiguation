import numpy as np
import xml.etree.ElementTree as ET
import os
import csv
from hyperopt import fmin, atpe, hp, STATUS_OK, Trials, tpe, rand
# from evaluator_for_more import cluster_evaluate
from evaluator_for_singal import cluster_evaluate


space = {
    'affinity_metric': hp.choice('affinity_metric', ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed']),
}

def load_data_from_path(path):
    tree = ET.parse(path)
    root = tree.getroot()
    data = []

    for vector in root.findall('.//vector'):
        values = list(map(float, vector.text.split()))
        data.append(values)
    return np.array(data)

def objective(params, path):
    affinity_metric = params['affinity_metric']
    try:
        # for Pairwise F1
        precision, recall, f1 = cluster_evaluate(path, method='GHAC', linkage_method='average', affinity_metric=affinity_metric)
        
        # for multiple evaluation metrics: Pairwise F1, B-Cubed F1, and K metrics
        # res = cluster_evaluate(path, method='GHAC', linkage_method='average', affinity_metric=affinity_metric, metrics=['pairwise'])
        # precision, recall, f1 = res['pairwise']
        
    except ValueError as e:
        if 'Cosine affinity cannot be used when X contains zero vectors' in str(e):

            return {'loss': float('inf'), 'status': STATUS_OK, 'precision': 0, 'recall': 0, 'f1': 0}
        else:
            raise e
    return {'loss': -f1, 'status': STATUS_OK, 'precision': precision, 'recall': recall, 'f1': f1}

def optimize_xml(path):
    try:
        trials = Trials()
        best = fmin(fn=lambda params: objective(params, path), 
                    space=space, 
                    algo=atpe.suggest, 
                    max_evals=200, 
                    trials=trials)
        affinity_metrics = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed']
        best_affinity_metric = affinity_metrics[best['affinity_metric']]
        best_trial = trials.best_trial['result']
        precision, recall, f1 = best_trial['precision'], best_trial['recall'], best_trial['f1']
    except ValueError as e:
        if 'Cosine affinity cannot be used when X contains zero vectors' in str(e):

            best_affinity_metric = 'precomputed'
            
            # for Pairwise F1
            precision, recall, f1 = cluster_evaluate(path, method='GHAC', linkage_method='average', affinity_metric='precomputed')
            
            # for multiple evaluation metrics: Pairwise F1, B-Cubed F1, and K metrics
            # res= cluster_evaluate(path, method='GHAC', linkage_method='average', affinity_metric='precomputed',metrics=['k_metric'])
            # precision, recall, f1 = res['k_metric']
        else:
            raise e
    return best_affinity_metric, precision, recall, f1

def main():
    directory = r"raw-data"  # or 'raw-data-citeseerx-kim'
    results_file = r"DATASET_hyperparameters_result.csv"
    with open(results_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Best Hyperparameters", "Precision", "Recall", "F1"])
        for filename in os.listdir(directory):
            if filename.endswith('.xml'):
                file_path = os.path.join(directory, filename)
                best_hyperparameters, precision, recall, f1 = optimize_xml(file_path)
                writer.writerow([filename, best_hyperparameters, precision, recall, f1])
                print(f"Processed {filename}")

if __name__ == "__main__":
    main()
