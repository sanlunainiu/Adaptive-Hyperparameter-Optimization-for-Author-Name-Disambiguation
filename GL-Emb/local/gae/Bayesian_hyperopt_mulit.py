from local.gae.train_gpu_new import gae_for_na
from utils import settings
from utils import data_utils
from hyperopt import hp, fmin, Trials, STATUS_OK, atpe, space_eval
import tensorflow.compat.v1 as tf
from os.path import join
import csv

tf.disable_v2_behavior()

class Dummy:
    pass

FLAGS = Dummy()

space = {
    'learning_rate': hp.uniform('learning_rate', 0.0001, 0.1),
}

names_list = data_utils.load_json(settings.DATA_DIR, 'test_name_list.json') 

best_params_and_scores = {}

def objective(space, name):
    tf.reset_default_graph()  
    FLAGS.learning_rate = space['learning_rate']
    results = gae_for_na(name)
    [prec, rec, f1], num_nodes, n_clusters = results
    return {'loss': -f1, 'status': STATUS_OK, 'prec': prec, 'rec': rec, 'f1': f1}

# run hyperparameter optimization for each name
for name in names_list:
    trials = Trials()
    best = fmin(fn=lambda space: objective(space, name),
                space=space,
                algo=atpe.suggest,
                max_evals=200,
                trials=trials)

    # find the best evaluation metric
    best_trial = min(trials.trials, key=lambda x: x['result']['loss'])
    best_score = {
        'params': best,
        'prec': best_trial['result']['prec'],
        'rec': best_trial['result']['rec'],
        'f1': best_trial['result']['f1']
    }
    best_params_and_scores[name] = best_score

avg_prec_new = sum(score['prec'] for score in best_params_and_scores.values()) / len(best_params_and_scores)
avg_rec_new = sum(score['rec'] for score in best_params_and_scores.values()) / len(best_params_and_scores)
avg_f1_new = sum(score['f1'] for score in best_params_and_scores.values()) / len(best_params_and_scores)

csv_file = join(settings.OUT_DIR, 'metric_hyperparameter_optimization_result.csv')
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Name', 'learning_rate', 'prec', 'rec', 'f1'])
    
    for name, metrics in best_params_and_scores.items():
        writer.writerow([
            name,
            metrics['params'].get('learning_rate', 'N/A'),
            metrics['prec'],
            metrics['rec'],
            metrics['f1']
        ])
