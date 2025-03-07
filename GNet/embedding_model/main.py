import embedding
import train_helper
import sampler
import eval_metric_more
import eval_metric
import os
import time
import pandas as pd
import data_parser

F1_List = []
F1_Max_List = []
F1_Max_List_pre =[]
F1_Max_List_rec = []
f1_name = {}


def get_file_list(file_dir):
    file_list = []
    for root, dirs, files in os.walk(file_dir):
        file_list.append(files)
    return file_list[0]

def get_median(data):
        data = sorted(data)
        size = len(data)
        if size % 2 == 0:   # 判断列表长度为偶数
            median = (data[size//2]+data[size//2-1])/2
            data[0] = median
        if size % 2 == 1:   # 判断列表长度为奇数
            median = data[(size-1)//2]
            data[0] = median
        return data[0]

def main(filename):
    """
    pipeline for representation learning for all papers for a given name reference
    """
    latent_dimen = 40
    alpha = 0.02
    matrix_reg = 0.05
    num_epoch = 10
    sampler_method = 'uniform'
    
    dataset = data_parser.DataSet(filename)
    dataset.reader_arnetminer()
       
    bpr_optimizer = embedding.BprOptimizer(latent_dimen, alpha, matrix_reg)
    pp_sampler = sampler.CoauthorGraphSampler() # C_Graph
    pd_sampler = sampler.BipartiteGraphSampler()
    dd_sampler = sampler.LinkedDocGraphSampler() # D_Graph
    dt_sampler = sampler.DocumentTitleSampler() # T_Graph
    djconf_sampler = sampler.DocumentJConfSampler() # Jconf_Graph
    dyear_sampler = sampler.DocumentYearSampler() # Year_Graph
    dorg_sampler = sampler.DocumentOrgSampler() # Org_Graph
    dabstract_sampler = sampler.DocumentAbstractSampler() # Abstract_Graph
    eval_f1 = eval_metric.Evaluator()
    # eval_f1 = eval_metric_more.Evaluator()

    run_helper = train_helper.TrainHelper()


    # 基本的，不加任何额外的东西
    avg_f1,avg_pre,avg_rec = run_helper.helper(num_epoch, dataset, bpr_optimizer,
                                                # pp_sampler,
                                                # pd_sampler,
                                                dd_sampler,
                                                dt_sampler,
                                                # djconf_sampler,
                                                # dorg_sampler,
                                                # dyear_sampler,
                                                dabstract_sampler,
                                                eval_f1, 
                                                sampler_method,
                                                filename)

    F1_Max_List.append(avg_f1)
    F1_Max_List_pre.append(avg_pre)
    F1_Max_List_rec.append(avg_rec)

    print(avg_f1)
    return avg_f1,avg_pre,avg_rec

if __name__ == "__main__":
    file_list = get_file_list(r"sampled_data")  # or 'sampled_data_citeseerx'
    file_list = sorted(file_list)
    file_list = file_list[:]
    cnt = 0
    copy_f1_list = []
    for x in file_list:
        cnt += 1
        filename = r'sampled_data\\' + str(x)   # or 'sampled_data_citeseerx'
        print(filename)
        print("count:" + str(cnt))
        print(time.strftime('%H:%M:%S', time.localtime(time.time())))
        F1_Max_List = []
        for i in range(1):
            avg_f1,avg_pre,avg_rec = main(filename)
            print(F1_Max_List)
            print(avg_f1,avg_pre,avg_rec)
        f1_name[x]=[]
        f1_name[x].append(get_median(F1_Max_List))
        f1_name[x].append(max(F1_Max_List_pre))
        f1_name[x].append(max(F1_Max_List_rec))
        F1_List.append(max(F1_Max_List))
        copy_f1_list.append(max(F1_Max_List))
        print("real time f1:" + str(sum(F1_List) / len(F1_List)))

    F1_List.sort()
    F1_List = F1_List[::-1]
    print(F1_List)
    print(f1_name)
    print(sorted(f1_name.items(), key=lambda x: x[1]))
    print("top10 f1:" + str(sum(F1_List[0:10]) / 10))
    print("top30 f1:" + str(sum(F1_List[0:30]) / 30))
    print("top50 f1:" + str(sum(F1_List[0:50]) / 50))
    print("top70 f1:" + str(sum(F1_List[0:70]) / 70))
    print("top100 f1:" + str(sum(F1_List[0:100]) / 100))
    print("all f1:" + str(sum(F1_List) /7022))
    dataframe = pd.DataFrame({"author":file_list,"macro_f1": copy_f1_list})
    dataframe.to_csv(r'Evaluation_metrics_DATASET_AND_results.csv', index=False)

