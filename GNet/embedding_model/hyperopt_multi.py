import os
import time
import numpy as np
import pandas as pd
import data_parser
import embedding
import train_helper
import sampler
import eval_metric_more
from hyperopt import fmin, atpe, hp, Trials, STATUS_OK, tpe, rand

def get_file_list(file_dir):
    file_list = []
    for root, dirs, files in os.walk(file_dir):
        file_list.extend(files)
    return file_list

def main(filename, alpha):
    latent_dimen = 40
    matrix_reg = 0.05
    num_epoch = 10
    sampler_method = 'uniform'

    dataset = data_parser.DataSet(filename)
    dataset.reader_arnetminer()
    bpr_optimizer = embedding.BprOptimizer(latent_dimen, alpha, matrix_reg)
    dd_sampler = sampler.LinkedDocGraphSampler()
    dt_sampler = sampler.DocumentTitleSampler()
    dabstract_sampler = sampler.DocumentAbstractSampler()
    eval_f1 = eval_metric_more.Evaluator()

    run_helper = train_helper.TrainHelper()
    avg_f1, avg_pre, avg_rec = run_helper.helper(num_epoch, dataset, bpr_optimizer,
                                                 dd_sampler, dt_sampler, dabstract_sampler,
                                                 eval_f1, sampler_method, filename)
    return avg_f1

def optimize_alpha(filename):
    f1_scores = []

    def objective(alpha):
        f1_score = main(filename, alpha)
        f1_scores.append(f1_score) 
        return -f1_score

    space = hp.uniform('alpha', 0.0001, 0.1)
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=atpe.suggest,
        max_evals=200,
        trials=trials
    )
    best_alpha = best['alpha']
    best_f1 = max(f1_scores)  
    return best_alpha, best_f1

if __name__ == "__main__":
    file_dir = r"sampled_data"
    file_list = get_file_list(file_dir)
    file_list = sorted(file_list)
    
    results = []
    for filename in file_list:
        best_alpha, best_f1 = optimize_alpha(os.path.join(file_dir, filename))
        results.append({
            "filename": filename,
            "alpha": best_alpha,
            "f1": best_f1
        })
        print(f"Processed {filename}: Best alpha = {best_alpha}, Best F1 = {best_f1}")
    

    results_df = pd.DataFrame(results)
    results_df.to_csv(r'Evaluation_metrics_DATASET_AND_results.csv', index=False)
