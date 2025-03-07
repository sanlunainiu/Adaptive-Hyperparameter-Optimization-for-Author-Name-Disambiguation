import torch
import random
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances

from torch_geometric.nn import GAE
from loadmodel.att_gnn import ATTGNN
from dataset.load_data import load_dataset, load_graph
from dataset.save_results import save_results_for_names
from os.path import join,dirname
from .generate_pair import generate_pair

from params import set_params


args = set_params()

seed = args.seed
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device(("cuda:"+str(args.gpu)) if torch.cuda.is_available() and args.cuda else "cpu")


class BONDTrainer:
    def __init__(self) -> None:
        pass

    def onehot_encoder(self, label_list):
        """
        Transform label list to one-hot matrix.
        Arg:
            label_list: e.g. [0, 0, 1]
        Return:
            onehot_mat: e.g. [[1, 0], [1, 0], [0, 1]]
        """
        if isinstance(label_list, np.ndarray):
            labels_arr = label_list
        else:
            try:
                labels_arr = np.array(label_list.cpu().detach().numpy())
            except:
                labels_arr = np.array(label_list)
        
        num_classes = max(labels_arr) + 1
        onehot_mat = np.zeros((len(labels_arr), num_classes+1))

        for i in range(len(labels_arr)):
            onehot_mat[i, labels_arr[i]] = 1

        return onehot_mat
    
    def matx2list(self, adj):
        """
        Transform matrix to list.
        """
        adj_preds = []
        for i in adj:
            if isinstance(i, np.ndarray):
                temp = i
            else:
                temp = i.cpu().detach().numpy()
            for idx, j in enumerate(temp):
                if j == 1: 
                    adj_preds.append(idx)
                    break
                if idx == len(temp)-1:
                    adj_preds.append(-1)

        return adj_preds

    def post_match(self, pred, pubs, name, mode):
        """
        Post-match outliers.
        Args:
            pred(list): prediction e.g. [0, 0, -1, 1]
            pubs(list): paper-ids
            name(str): author name
            mode(str): train/valid/test
        Return:
            pred(list): after post-match e.g. [0, 0, 0, 1] 
        """
        #1 outlier from dbscan labels
        outlier = set()
        for i in range(len(pred)):
            if pred[i] == -1:
                outlier.add(i)

        #2 outlier from building graphs (relational)
        datapath = join(args.save_path, 'graph', mode, name)
        with open(join(datapath, 'rel_cp.txt'), 'r') as f:
            rel_outlier = [int(x) for x in f.read().split('\n')[:-1]] 

        for i in rel_outlier:
            outlier.add(i)
        
        # print(f"post matching {len(outlier)} outliers")
        paper_pair = generate_pair(pubs, name, outlier, mode)
        paper_pair1 = paper_pair.copy()
        
        K = len(set(pred))

        for i in range(len(pred)):
            if i not in outlier:
                continue
            j = np.argmax(paper_pair[i])
            while j in outlier:
                paper_pair[i][j] = -1
                last_j = j
                j = np.argmax(paper_pair[i])
                if j == last_j:
                    break

            if paper_pair[i][j] >= 1.5:
                pred[i] = pred[j]
            else:
                pred[i] = K
                K = K + 1

        for ii, i in enumerate(outlier):
            for jj, j in enumerate(outlier):
                if jj <= ii:
                    continue
                else:
                    if paper_pair1[i][j] >= 1.5:
                        pred[j] = pred[i]
        return pred

    def fit(self, datatype, name=None,lr=None):
        names, pubs = load_dataset(datatype)
        results = {}
    
        f1_list = []  # Assuming f1_list is used elsewhere, though not shown in this snippet
    
        # If a specific name is provided, limit the names list to just this name
        if name:
            names = [name] if name in names else []
    
        for name in names:
            # print("training:", name)
            results[name] = []
    
            # ==== Load data ====
            label, ft_list, data = load_graph(name)
            num_cluster = int(ft_list.shape[0] * args.compress_ratio)
            layer_shape = [ft_list.shape[1]]  # input layer shape
            layer_shape.extend(args.hidden_dim)  # hidden layer shapes
            layer_shape.append(num_cluster)  # output layer shape, adjusted to the number of nodes
    
            # get the list of pid (paper-id)
            name_pubs = []
            if datatype == 'train':
                for aid in pubs[name]:
                    name_pubs.extend(pubs[name][aid])
            else:
                for pid in pubs[name]:
                    name_pubs.append(pid)
    
            # ==== Init model ====
            model = GAE(ATTGNN(layer_shape))
            ft_list = ft_list.float().to(device)
            data = data.to(device)
            model.to(device)
    
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.l2_coef)
    
            for epoch in range(args.epochs):
                model.train()
                optimizer.zero_grad()
                logits, embd = model.encode(ft_list, data.edge_index, data.edge_attr)
                dis = pairwise_distances(embd.cpu().detach().numpy(), metric='cosine')
                db_label = DBSCAN(eps=args.db_eps, min_samples=args.db_min, metric='precomputed').fit_predict(dis)
                db_label = torch.from_numpy(db_label).to(device)
    
                # change to one-hot form
                class_matrix = torch.from_numpy(self.onehot_encoder(db_label)).to(device)
                local_label = torch.mm(class_matrix, class_matrix.t()).float().to(device)
                global_label = torch.matmul(logits, logits.t())
    
                loss_cluster = F.binary_cross_entropy_with_logits(global_label, local_label)
                loss_recon = model.recon_loss(embd, data.edge_index)
                loss_train = args.cluster_w * loss_cluster + (1 - args.cluster_w) * loss_recon
    
                # if epoch % 5 == 0:
                #     print(f'epoch: {epoch:3d}, cluster loss: {loss_cluster.item():.4f}, recon loss: {loss_recon.item():.4f}, ALL loss: {loss_train.item():.4f}')
    
                loss_train.backward()
                optimizer.step()
    
            # ==== Evaluate ====
            with torch.no_grad():
                model.eval()
                logits, embd = model.encode(ft_list, data.edge_index, data.edge_attr)
                lc_dis = pairwise_distances(embd.cpu().detach().numpy(), metric='cosine')
                local_label = DBSCAN(eps=args.db_eps, min_samples=args.db_min, metric='precomputed').fit_predict(lc_dis)
                gl_dis = pairwise_distances(logits.cpu().detach().numpy(), metric='cosine')
                gl_label = DBSCAN(eps=args.db_eps, min_samples=args.db_min, metric='precomputed').fit_predict(gl_dis)
    
                # change to one-hot form and get N * N matrix
                class_matrix = torch.from_numpy(self.onehot_encoder(local_label)).to(device)
                local_label = torch.mm(class_matrix, class_matrix.t())
                pred = self.matx2list(local_label)
    
                if args.post_match:
                    pred = self.post_match(pred, name_pubs, name, datatype)
    
                # Save results
                results[name] = pred
    
        result_path = save_results_for_names(names, pubs, results)
        # print("Done! Results saved:", result_path)
        return result_path
