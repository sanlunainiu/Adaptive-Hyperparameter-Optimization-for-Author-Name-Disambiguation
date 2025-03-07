import os
import argparse
import csv
import data_parser
import embedding
import train_helper
import sampler
import eval_metric
import eval_metric_more

def parse_args():
    """
    Parse the embedding model arguments
    """
    parser = argparse.ArgumentParser(description="Run embedding for name disambiguation")
    parser.add_argument("--data_folder", type=str, default=r"sampled_data", # or sampled_data_citeseerx
                        help='path to the folder containing the data files')
    parser.add_argument("--latent_dimen", type=int, default=20,
                        help='number of dimensions in embedding')
    parser.add_argument("--alpha", type=float, default=0.02,
                        help='learning rate')
    parser.add_argument("--matrix_reg", type=float, default=0.05,
                        help='matrix regularization parameter')
    parser.add_argument("--num_epoch", type=int, default=50,
                        help="number of epochs for SGD inference")
    parser.add_argument("--sampler_method", type=str, default="uniform", 
                        help="sampling approach")
    
    return parser.parse_args()

def main(file_path):
    """
    Process a single file for name disambiguation
    """
    dataset = data_parser.DataSet(file_path)
    dataset.reader_arnetminer()
    bpr_optimizer = embedding.BprOptimizer(args.latent_dimen, args.alpha, args.matrix_reg)
    pp_sampler = sampler.CoauthorGraphSampler()
    pd_sampler = sampler.BipartiteGraphSampler()
    dd_sampler = sampler.LinkedDocGraphSampler()
    # eval_f1 = eval_metric.Evaluator()
    eval_f1 = eval_metric_more.Evaluator()

    run_helper = train_helper.TrainHelper()
    average_f1, average_prec, average_rec = run_helper.helper(args.num_epoch, dataset, bpr_optimizer, pp_sampler, pd_sampler, dd_sampler, eval_f1, args.sampler_method)
    return average_f1, average_prec, average_rec

if __name__ == "__main__":
    args = parse_args()
    # Create and open a CSV file to store the results
    with open(os.path.join(args.data_folder, r'Evaluation_metrics_DATASET_AND_results.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['File Name', 'Average F1', 'Average Precision', 'Average Recall'])
        # Iterate over all XML files in the specified directory
        for file_name in os.listdir(args.data_folder):
            print(f'disambiguating {file_name}')
            if file_name.endswith('.xml'):
                file_path = os.path.join(args.data_folder, file_name)
                average_f1, average_prec, average_rec = main(file_path)
                # Write the results for each file to the CSV
                writer.writerow([os.path.splitext(file_name)[0], average_f1, average_prec, average_rec])
