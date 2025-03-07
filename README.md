Adaptive Hyperparameter Optimization for Author Name Disambiguation
============
This is implementation of our paper:

Lu, S., Zhou, Y. (in press). Adaptive Hyperparameter Optimization for Author Name Disambiguation. Journal of the Association for Information Science and Technology. DOI: 10.1002/asi.24996

## summary

In this paper, we propose a block-based hyperparameter optimization algorithm. Experiments on 6 state-of-the-art algorithms and 11 public datasets show that our method can significantly improve the performance of name disambiguation, namely **Cluster F1/Pairwise F1, B-Cubed F1, and K metrics**. On this basis, we train a random forest regression model that uses block features to fit the optimal hyperparameters of each block to predict the optimal hyperparameters of new blocks. The results show that the selected 16 features can better predict the optimal hyperparameters of blocks. 6 state-of-the-art name disambiguation algorithms and 11 public datasets are shown in Table 1.

<center>Table Algorithms and Datasets</center>


| Algorithm                                                    | Dataset                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [@ANet (Zhang et al., 2017)](https://github.com/baichuan/disambiguation_embedding) | [Arnetminer](https://www.aminer.org/disambiguation), [CiteSeerX](https://figshare.com/articles/dataset/DBLP-derived_labeled_data_for_author_name_disambiguation/6840281/2?file=12454577) |
| [GNet (Xu et al., 2018)](https://github.com/xujunrt/Author-Disambiguation) | [Arnetminer](https://www.aminer.org/disambiguation), [CiteSeerX](https://figshare.com/articles/dataset/DBLP-derived_labeled_data_for_author_name_disambiguation/6840281/2?file=12454577) |
| [PHNet (Qiao et al., 2019)](https://github.com/joe817/name-disambiguation) | [Arnetminer](https://www.aminer.org/disambiguation), [CiteSeerX](https://figshare.com/articles/dataset/DBLP-derived_labeled_data_for_author_name_disambiguation/6840281/2?file=12454577) |
| [G/L-Emb (Zhang et al., 2018)](https://github.com/neozhangthe1/disambiguation) | [Aminer](https://static.aminer.cn/misc/na-data-kdd18.zip)    |
| [BOND (Cheng et al., 2024)](https://github.com/THUDM/WhoIsWho/tree/main/bond) | [WhoIsWho-v3](http://whoiswho.biendata.xyz/#/data)           |
| [S2AND (Subramanian et al., 2021)](https://github.com/allenai/S2AND) |                                                              |

We strongly recommend familiarizing yourself with these six state-of-the-art algorithms. As our work builds upon these state-of-the-art algorithms, we will provide detailed implementations demonstrating how our approach improves performance across each of these six algorithms. Additionally, we present the implementation of a Random Forest regression model that fits block characteristics to their optimal hyperparameters.

reference

<font size="3">[Cheng, Y., Chen, B., Zhang, F., & Tang, J. (2024). BOND: Bootstrapping From-Scratch Name Disambiguation with Multi-task Promoting. In T.-S. Chua, C.-W. Ngo, R. Kumar, H. W. Lauw, & R. K.-W. Lee (Eds.), Proceedings of the ACM on Web Conference 2024, WWW’24, Singapore—May 13-17, 2024 (pp. 4216-4226). ACM.](https://doi.org/10.1145/3589334.3645580) </font>

<font size="3">[Qiao, Z., Du, Y., Fu, Y., Wang, P., & Zhou, Y. (2019). Unsupervised author disambiguation using heterogeneous graph convolutional network embedding. In 2019 IEEE International Conference on Big Data (Big Data), Los Angeles, CA, USA—December 9-12, 2019 (pp. 910-919). IEEE.](https://doi.org/10.1109/BigData47090.2019.9005458) </font>

<font size="3">[Subramanian, S., King, D., Downey, D., & Feldman, S. (2021). S2and: A benchmark and evaluation system for author name disambiguation. In 2021 ACM/IEEE Joint Conference on Digital Libraries, JCDL’21, Virtual Event—September 27-30, 2021 (pp. 170-179). IEEE.](https://doi.org/10.1109/JCDL52503.2021.00029) </font>

<font size="3">[Xu, J., Shen, S., Li, D., & Fu, Y. (2018). A network-embedding based method for author disambiguation. In A. Cuzzocrea, D. Srivastava, R. Agrawal, A. Broder, M. Zaki, S. Candan, A. Labrinidis, A. Schuster, & H. Wang (Eds.), Proceedings of the 27th ACM International Conference on Information and Knowledge Management, CIKM’18, Torino, Italy—October 22-26, 2018 (pp. 1735-1738). ACM.](https://doi.org/10.1145/3269206.3269272)</font>

<font size="3">[Zhang, B., & Al Hasan, M. (2017). Name disambiguation in anonymized graphs using network embedding. In E.-P. Lim, M. Winslett, M. Sanderson, A. W.-C. Fu, J. Sun, J. S. Culpepper, E. Lo, J. C. Ho, D. Donato, R. Agrawal, Y. Zheng, C. Castillo, A. Sun, V. S. Tseng, & C. Li (Eds.), Proceedings of the 2017 ACM on Conference on Information and Knowledge Management, CIKM’17, Singapore—November 6-10, 2017 (pp. 1239-1248). ACM.](https://doi.org/10.1145/3132847.3132873)</font>

<font size="3">[Zhang, Y., Zhang, F., Yao, P., & Tang, J. (2018). Name disambiguation in AMiner: Clustering, maintenance, and human in the loop. In Y. Guo & F. Farooq (Eds.), Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD’18, London, UK—August 19-23, 2018 (pp. 1002–1011). ACM.](https://doi.org/10.1145/3219819.3219859)</font>

## How to run: hyperparameter optimization

<h3 align="center">ANet  (Zhang et al., 2017)</h3>

#### Requirements

* python 3.8.x (3.8.19 recommended). Install requirements via ```
  pip install -r ANet/requirements.txt``` 

#### Usage

<div style="text-align: center; font-size: 20px; font-weight: bold;">1. Performance of ANet Algorithm</div>

* To compute Cluster F1:

  ```python
  # In Anet/embedding_model/all_main.py
  import eval_metric
  
  # In 'main' function
  eval_f1 = eval_metric.Evaluator()
  
  # In Anet\embedding_model\train_helper.py
  average_f1, average_pre, average_rec = eval_f1.compute_f1(dataset, bpr_optimizer)
  ```

* For additional evaluation metrics: B3 F1, and K.

  ```python
  # In Anet/embedding_model/all_main.py
  import eval_metric_more
  
  # In 'main' function
  eval_f1 = eval_metric_more.Evaluator()
  
  # Computing B3 F1, K-metric, or Cluster F1
  # In Anet\embedding_model\train_helper.py
  average_f1, average_prec, average_rec = eval_f1.compute_f1(
      dataset, 
      bpr_optimizer, 
      value_mode='b3' # 'kmetric', 'cluster'
  )
  ```

* For individual block evaluation:

  ```python
  # Use net\embedding_model\main.py for single block Cluster F1 computation
  ```

  <div style="text-align: center; font-size: 20px; font-weight: bold;">2. Performance with Block-based Hyperparameter Optimization of ANet</div>

* To compute Cluster F1:

  ```python
  # In Anet/embedding_model/hyperopt_multi.py
  import eval_metric
  
  # In 'main' function
  eval_f1 = eval_metric.Evaluator()
  
  # In Anet\embedding_model\train_helper.py
  average_f1, average_pre, average_rec = eval_f1.compute_f1(dataset, bpr_optimizer)
  ```

* For additional evaluation metrics: B3 F1, and K.

  ```python
  # In Anet/embedding_model/hyperopt_multi.py
  import eval_metric_more
  
  # In 'main' function
  eval_f1 = eval_metric_more.Evaluator()
  
  # Computing B3 F1, K-metric, Cluster F1
  # In Anet\embedding_model\train_helper.py
  average_f1, average_prec, average_rec = eval_f1.compute_f1(
      dataset, 
      bpr_optimizer, 
      value_mode='b3' # 'kmetric', 'cluster'
  )
  ```

We also developed a block-based hyperparameter optimization code that can use a multithreaded, see ```Anet/embedding_model/hyperopt_multi_mp.py```.

<h3 align="center">GNet  (Xu et al., 2018)</h3>

#### Requirements

- python 3.8.x (3.8.19 recommended). Install requirements via ```
  pip install -r GNet/requirements.txt``` 

#### Uasge

<div style="text-align: center; font-size: 20px; font-weight: bold;">1. Performance of GNet Algorithm</div>

- To compute Cluster F1:

  ```python
  # In Gnet/embedding_model/mian.py
  import eval_metric
  
  # In 'main' function
  eval_f1 = eval_metric.Evaluator()
  
  # In Gnet\embedding_model\train_helper.py
  average_f1, average_pre, average_rec = eval_f1.compute_f1(dataset, bpr_optimizer)
  ```

- For additional evaluation metrics: B3 F1, and K.

  ```python
  # In Gnet/embedding_model/mian.py
  import eval_metric_more
  
  # In 'main' function
  eval_f1 = eval_metric_more.Evaluator()
  
  # Computing B3 F1, K-metric, or Cluster F1
  # In Gnet\embedding_model\train_helper.py
  average_f1, average_pre, average_rec = eval_f1.compute_f1(
      dataset, 
      bpr_optimizer, 
      cluster_method='dbscan', 
      eval_metric='b3' # 'kmetric', 'cluster'
  ) 
  ```

<div style="text-align: center; font-size: 20px; font-weight: bold;">2. Performance with Block-based Hyperparameter Optimization of GNet</div>

- To compute Cluster F1:

  ```python
  # In Gnet/embedding_model/hyperopt_multi.py
  import eval_metric
  
  # In 'main' function
  eval_f1 = eval_metric.Evaluator()
  
  # In Gnet\embedding_model\train_helper.py
  average_f1, average_pre, average_rec = eval_f1.compute_f1(dataset, bpr_optimizer)
  ```

- For additional evaluation metrics: B3 F1, and K.

  ```python
  # In Gnet/embedding_model/hyperopt_multi.py
  import eval_metric_more
  
  # In 'main' function
  eval_f1 = eval_metric_more.Evaluator()
  
  # Computing B3 F1, K-metric, or Cluster F1
  # In Gnet\embedding_model\train_helper.py
  average_f1, average_pre, average_rec = eval_f1.compute_f1(
      dataset, 
      bpr_optimizer, 
      cluster_method='dbscan', 
      eval_metric='b3' # 'kmetric', 'cluster'
  ) 
  ```

<h3 align="center">PHNet  (Qiao et al., 2019)</h3>

#### Requirements

You need to install two environments to execute the code.

Environment 1:

- python 3.8.x (3.8.19 recommended). Install requirements via ```
  pip install -r PHNet/requirements1.txt``` 

Environment 2:

- python 3.6.x (3.6.13 recommended). Install requirements via ```
  pip install -r PHNet/requirements2.txt``` 

#### Uasge

<div style="text-align: center; font-size: 20px; font-weight: bold;">0. Basic PHNet Algorithm</div>

```python
# step 1: preprocess the data. Run in Environment 1.
python PHNet/data_processing.py

# step 2: train the GRU based encoder to learn deep semantic representations. Run in Environment 2.
python PHNet/DRLgru.py 

# step 3: construct a PHNet and generate random walks. Run in Environment 1.
python PHNet/walks.py

# step 4: weighted heterogeneous network embedding. Run in Environment 2.
python PHNet/WHNE.py
```

<div style="text-align: center; font-size: 20px; font-weight: bold;">1. Performance of PHNet Algorithm</div>

The following code runs in Environment 1. 

- To compute Pairwise F1:

  ```python
  # Note that for different datasets, namely CiteseerX-kim and Arnetminer datasets, the corresponding data and trained models need to be loaded
  python PHNet/evaluator.py
  ```

- For additional evaluation metrics: B3 F1, and K.

  ```python
  # You will get results for all three metrics, Pairwise F1, B3 F1, and k-metric. For different datasets, namely CiteseerX-kim and Arnetminer, you need to load the corresponding data and trained model
  python PHNet/evaluator_more.py
  ```

<div style="text-align: center; font-size: 20px; font-weight: bold;">2. Performance with Block-based Hyperparameter Optimization of PHNet</div>

- To compute Pairwise F1:

  ```python
  # In PHNet/hyperopt_multi.py
  from evaluator_for_singal import cluster_evaluate
  
  # In 'objective' function
  precision, recall, f1 = cluster_evaluate(
      path, 
      method='GHAC', 
      linkage_method='average', 
      affinity_metric=affinity_metric
  )
  
  # In 'optimize_xml' function
  precision, recall, f1 = cluster_evaluate(
      path, 
      method='GHAC', 
      linkage_method='average', 
      affinity_metric='precomputed')
  ```

- For additional evaluation metrics: B3 F1, and K.

  ```python
  # In PHNet/hyperopt_multi.py
  from evaluator_for_more import cluster_evaluate
  
  # Computing B3 F1, K-metric, Pairwise F1
  # In 'objective' function
  res = cluster_evaluate(
      path, 
      method='GHAC', 
      linkage_method='average',   
      affinity_metric=affinity_metric, 
      metrics=['b3'] # 'k_metric', 'pairwise'
  )
  precision, recall, f1 = res['b3'] # 'k_metric', 'pairwise'
  
  # In 'optimize_xml' function
  res = cluster_evaluate(
      path, 
      method='GHAC', 
      linkage_method='average', 
      affinity_metric='precomputed', 
      metrics=['b3'] # 'k_metric', 'pairwise'
  )
  precision, recall, f1 = res['b3'] # 'k_metric', 'pairwise'
  ```

<h3 align="center">G/L-Emb  (Zhang et al., 2018)</h3>

#### Requirements

You need to install two environments to execute the code.

Environment 1:

- python 3.6.x (3.6.13 recommended). Install requirements via ```
  pip install -r GL-Emb/requirements1.txt``` 

Environment 2:

- python 3.8.x (3.8.19 recommended). Install requirements via ```
  pip install -r GL-Emb/requirements2.txt``` 

#### Uasge

<div style="text-align: center; font-size: 20px; font-weight: bold;">0. Basic G/L-Emb Algorithm</div>

Download raw data from https://static.aminer.cn/misc/na-data-kdd18.zip. Unzip the file and put the _data_ directory into GL-Emb directory.

```python
# step 1: preprocess the data. Run in Environment 1.
python scripts/preprocessing.py

# step 2: global model. Run in Environment 1.
python global_/gen_train_data.py
python global_/global_model.py
python global_/prepare_local_data.py
```

<div style="text-align: center; font-size: 20px; font-weight: bold;">1. Performance of G/L-Emb Algorithm</div>

The following code runs in Environment 2. 

- To compute Pairwise F1:

  ```python
  # In local/gae/train_gpu_new.py
  # In 'gae_for_name' function
  prec, rec, f1 =  pairwise_precision_recall_f1(clusters_pred, labels)
  ```

- For additional evaluation metrics: B3 F1, and K.

  ```python
  # In local/gae/train_gpu_new.py
  # In 'gae_for_name' function
  # Computing B3 F1
  prec, rec, f1 =  b3_metric(clusters_pred, labels) 
  
  # Computing K-metric
  prec, rec, f1 =  k_metric(clusters_pred, labels) 
  ```

<div style="text-align: center; font-size: 20px; font-weight: bold;">2. Performance with Block-based Hyperparameter Optimization of G/L-Emb</div>

- To compute Pairwise F1:

  ```python
  # In local/gae/Bayesian_hyperopt_mulit.py
  # In 'gae_for_name' function
  prec, rec, f1 =  pairwise_precision_recall_f1(clusters_pred, labels)
  ```

- For additional evaluation metrics: B3 F1, and K.

  ```python
  # In local/gae/Bayesian_hyperopt_mulit.py
  # In 'gae_for_name' function
  # Computing B3 F1
  prec, rec, f1 =  b3_metric(clusters_pred, labels) 
  
  # Computing K-metric
  prec, rec, f1 =  k_metric(clusters_pred, labels) 
  ```

<h3 align="center">BOND  (Chen et al., 2024)</h3>

#### Requirements

- python 3.8.x (3.8.19 recommended). Install requirements via ```
  pip install -r BOND/requirements.txt``` 

#### Uasge

<div style="text-align: center; font-size: 20px; font-weight: bold;">0. Basic BOND Algorithm</div>

Download raw data from <http://whoiswho.biendata.xyz/#/data>. Paper embedding can be downloaded from: [Embedding](https://pan.baidu.com/s/1A5XA9SCxvENM2kKPUv6X4Q?pwd=c9kk) Password: c9kk. Make sure that the structure of `data` directory in `BOND/dataset` is as follows.

```
BOND
├── dataset
    ├── data
        ├── paper_emb
        └── src
            ├── sna_test
            │   ├── sna_test_pub.json
            │   └── sna_test_raw.json
            ├── sna_valid
            │   ├── sna_valid_example.json
            │   ├── sna_valid_ground_truth.json
            │   ├── sna_valid_pub.json
            │   └── sna_valid_raw.json
            └── train
                ├── train_author.json
                └── train_pub.json
```

```python
# step 1: preprocess the data and train model
python BOND/demo.py
```

<div style="text-align: center; font-size: 20px; font-weight: bold;">1. Performance of BOND Algorithm</div>

- To compute Pairwise F1:

  ```python
  python BOND/evaluation/evaluation.py
  ```

- For additional evaluation metrics: B3 F1, and K.

  ```python
  # In BOND/evaluation/evaluation_more.py
  
  # Computing B3 F1
  evaluate(predict, ground_truth, metric_type='b3')
  
  # Computing K-metric
  evaluate(predict, ground_truth, metric_type='kmetric')
  
  # Pairwise F1 is also available
  evaluate(predict, ground_truth, metric_type='pairwise')
  ```

<div style="text-align: center; font-size: 20px; font-weight: bold;">2. Performance with Block-based Hyperparameter Optimization of BOND</div>

- To compute Pairwise F1:

  ```python
  # In BOND/demo_hyperopt_mulit.py
  from evaluation.evaluation_sig import evaluate
  
  # run BOND/demo_hyperopt_mulit.py
  python demo_hyperopt_mulit.py
  ```

- For additional evaluation metrics: B3 F1, and K.

  ```python
  # In BOND/demo_hyperopt_mulit.py
  from evaluation.evaluation_sig_more import evaluate
  
  # In 'objective' function
  # Computing B3 F1
  pairwise_precision, pairwise_recall, pairwise_f1 = evaluate(
      predict, 
      ground_truth, 
      metric='b3'
  )
  
  # Computing K-metric
  pairwise_precision, pairwise_recall, pairwise_f1 = evaluate(
      predict, 
      ground_truth, 
      metric='k'
  )
  
  # Pairwise F1 is also available
  pairwise_precision, pairwise_recall, pairwise_f1 = evaluate(
      predict, 
      ground_truth, 
      metric='pairwise'
  )
  ```

<h3 align="center">S2AND  (Subramanian et al., 2021)</h3>

#### Requirements

- python 3.8.x (3.8.15 recommended). Install requirements via ```
  pip install -r S2AND/requirements.txt``` 

#### Uasge

<div style="text-align: center; font-size: 20px; font-weight: bold;">0. Basic BOND Algorithm</div>

To obtain the S2AND dataset, run the following command after the package is installed (from inside the `S2AND` directory):
`[Expected download size is: 50.4 GiB]`

```
aws s3 sync --no-sign-request s3://ai2-s2-research-public/s2and-release data/
```

Note that this software package comes with tools specifically designed to access and model the dataset.

Modify the config file at `data/path_config.json`. This file should look like this

```
{
    "main_data_dir": "absolute path to wherever you downloaded the data to",
    "internal_data_dir": "ignore this one unless you work at AI2"
}
```

<div style="text-align: center; font-size: 20px; font-weight: bold;">1. Performance of BOND Algorithm</div>

- To compute Cluster F1, Pairwise F1, B3 F1, and K:

  ```python
  # This code will calculate four metrics simultaneously and output them together. We only report Pairwise F1, B3 F1, and K in our paper.
  python S2AND/S2AND_original.py
  ```

<div style="text-align: center; font-size: 20px; font-weight: bold;">2. Performance with Block-based Hyperparameter Optimization of S2AND</div>

- To compute Cluster F1, Pairwise F1, B3 F1, and K:

  ```python
  # In S2AND/S2AND_opt_for_blocks.py. Again We only report Pairwise F1, B3 F1, and K in our paper.
  
  # Computing Cluster F1
  b3_df, b3_metrics = optimize_and_save_results(
      'pairwise_clusters', 
      anddata, 
      clusterer,
      dataset_name, 
      parent_dir
  ) 
  
  # Computing Pairwise F1
  b3_df, b3_metrics = optimize_and_save_results(
      'pairwise_cmacro', 
      anddata, 
      clusterer,
      dataset_name, 
      parent_dir
  ) 
  
  # Computing B3 F1
  b3_df, b3_metrics = optimize_and_save_results(
      'b3', 
      anddata, 
      clusterer,
      dataset_name, 
      parent_dir
  ) 
  
  # Computing B3 F1
  b3_df, b3_metrics = optimize_and_save_results(
      'k_metric', 
      anddata, 
      clusterer,
      dataset_name, 
      parent_dir
  ) 
  
  # run S2AND/S2AND_opt_for_blocks.py
  python S2AND/S2AND_opt_for_blocks.py
  ```

## How to run: Random Forest regression

As mentioned in our paper, we fit the block features and optimal hyperparameters of the test set only on the G/L-Emb algorithm and the Aminer dataset. 

#### Uasge

Run the following code in environment 2.

```
python GL-Emb/random_forest.py
```

Change the hyperparameter optimization result data of Pairwise F1, B3 F1, and K. You can obtain the fitting results in Table 6 and the importance ranking in Figure 4 in our paper. Since the results of each disambiguation may be different, your results may have a slight difference from the results reported in our paper (Table 6 and Figure 4).

## Citation

If you find this work useful for your research, please consider citing the following paper:

```
Lu, S., Zhou, Y. (in press). Adaptive Hyperparameter Optimization for Author Name Disambiguation. Journal of the Association for Information Science and Technology. DOI: 10.1002/asi.24996
```

