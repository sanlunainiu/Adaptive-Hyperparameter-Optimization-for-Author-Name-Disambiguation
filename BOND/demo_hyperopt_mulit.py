import os
import numpy as np
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, atpe
import csv
from training.autotrain_bond_sig import BONDTrainer
from training.autotrain_bond_ensemble import ESBTrainer
from dataset.preprocess_SND import dump_name_pubs, dump_features_relations_to_file, build_graph
from params import set_params
from evaluation.evaluation_sig_more import evaluate
# from evaluation.evaluation_sig import evaluate
from dataset.load_data import load_json
import warnings
np.warnings = warnings

args = set_params()


names = load_json(r'dataset\data\names_list\valid_names_list.json')
results = []

def pipeline(model, name, lr):
    if model == 'bond':
        trainer = BONDTrainer()
        result_path = trainer.fit(datatype=args.mode, name=name, lr=lr)
    elif model == 'bond+':
        trainer = ESBTrainer()
        result_path = trainer.fit(datatype=args.mode, name=name)
    return result_path

predict = os.getcwd()
ground_truth = r'dataset\data\src\sna-valid\sna_valid_ground_truth.json'

for name in names:
    print(name)
    def objective(lr):
        result_path = pipeline(model="bond", name=name, lr=lr)
        predict_path = os.path.join(predict, result_path)
        pairwise_precision, pairwise_recall, pairwise_f1 = evaluate(predict_path, ground_truth, metric="b3") # 'k' or 'pairwise'
        return {
            'loss': -pairwise_f1,
            'status': STATUS_OK,
            'precision': pairwise_precision,
            'recall': pairwise_recall,
            'f1': pairwise_f1
        }

    space = hp.loguniform('lr', np.log(1e-7), np.log(1e-1))
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=atpe.suggest,
        max_evals=200,
        trials=trials
    )
    best_trial = trials.best_trial
    best_lr =best['lr']
    results.append({
        'name': name,
        'lr': best_lr,
        'precision': best_trial['result']['precision'],
        'recall': best_trial['result']['recall'],
        'f1': best_trial['result']['f1']
    })

# Write results to CSV file and compute averages
with open(r'out\Evaluation_metric_hyperopt_results.csv', 'w', newline='') as file:
    fieldnames = ['name', 'lr', 'precision', 'recall', 'f1']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        writer.writerow({
            'name': result['name'],
            'lr': result['lr'],
            'precision': result['precision'],
            'recall': result['recall'], 
            'f1': result['f1']
        })

    # Compute averages
    avg_precision = np.mean([r['precision'] for r in results])
    avg_recall = np.mean([r['recall'] for r in results])
    avg_f1 = avg_precision*avg_recall*2/(avg_precision + avg_recall)

    writer.writerow({'name': 'Average', 'lr': '', 'precision': avg_precision, 'recall': avg_recall, 'f1': avg_f1})

