# Single threaded version
import os
import sys
import argparse
import csv
from hyperopt import fmin, tpe, hp, Trials, atpe, STATUS_OK

import data_parser
import embedding
import train_helper
import sampler
import eval_metric
import eval_metric_more

def parse_args(file_path):
    """
    Parse the embedding model arguments for a given file path
    """
    parser = argparse.ArgumentParser(description="Run embedding for name disambiguation")
    parser.add_argument("--file_path", type=str, default=file_path,
                        help='input file name')
    parser.add_argument("--latent_dimen", type=int, default=20,
                        help='number of dimensions in embedding')
    parser.add_argument("--alpha", type=float, default=0.02,
                        help='learning rate')
    parser.add_argument("--matrix_reg", type=float, default=0.005,
                        help='matrix regularization parameter')
    parser.add_argument("--num_epoch", type=int, default=50,
                        help="number of epochs for SGD inference")
    parser.add_argument("--sampler_method", type=str, default="uniform",
                        help="sampling approach")
    return parser.parse_args([])

def objective(params, file_path):
    args = parse_args(file_path)
    args.alpha = params['alpha']

    dataset = data_parser.DataSet(args.file_path)
    dataset.reader_arnetminer()

    bpr_optimizer = embedding.BprOptimizer(args.latent_dimen, args.alpha, args.matrix_reg)
    pp_sampler = sampler.CoauthorGraphSampler()
    pd_sampler = sampler.BipartiteGraphSampler()
    dd_sampler = sampler.LinkedDocGraphSampler()
    eval_f1 = eval_metric_more.Evaluator()

    run_helper = train_helper.TrainHelper()
    average_f1, average_prec, average_rec = run_helper.helper(args.num_epoch, dataset, 
                                                              bpr_optimizer, pp_sampler, 
                                                              pd_sampler, dd_sampler, 
                                                              eval_f1, args.sampler_method)

    return {
        'loss': -average_f1, 
        'status': STATUS_OK, 
        'average_prec': average_prec, 
        'average_rec': average_rec,
        'average_f1': average_f1  # 确保此键被设置
    }

def run_hyperopt(file_path):
    space = {
        'alpha': hp.uniform('alpha', 0.0001, 0.1)
    }

    trials = Trials()
    best = fmin(
        fn=lambda params: objective(params, file_path),
        space=space,
        algo=atpe.suggest,
        max_evals=200,
        trials=trials
    )

    optimization_results = []
    for trial in trials.trials:
        params = {k: v[0] for k, v in trial['misc']['vals'].items()}
        result = trial['result']
        if result['status'] == STATUS_OK:
            optimization_results.append({
                'params': params,
                'loss': result['loss'],
                'average_prec': result.get('average_prec'),
                'average_rec': result.get('average_rec'),
                'average_f1': result.get('average_f1')
            })

    best_result = min(optimization_results, key=lambda x: x['loss'], default=None)
    return best, best_result

if __name__ == "__main__":
    data_folder = r"sampled_data" # or 'sampled_data_citeseerx'
    with open(os.path.join('Evaluation_metrics_DATASET_hyperopt_results.csv'), 'w', newline='') as csvfile:
        fieldnames = ['Author', 'Best Alpha', 'Loss', 'Average F1', 'Average Precision', 'Average Recall']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for file_name in os.listdir(data_folder):
            if file_name.endswith('.xml'):
                print(f'Looking at {file_name} to find the best alpha value.')
                file_path = os.path.join(data_folder, file_name)
                best_params, best_result = run_hyperopt(file_path)
                if best_result:
                    writer.writerow({
                        'Author': os.path.splitext(file_name)[0],
                        'Best Alpha': best_params['alpha'],
                        'Loss': best_result['loss'],
                        'Average F1': best_result['average_f1'],
                        'Average Precision': best_result['average_prec'],
                        'Average Recall': best_result['average_rec']
                    })
