import torch
from tqdm import tqdm
import math
from torch.utils.data import DataLoader
from m_utils import *
from collections import Counter
from math import ceil
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA, PCA


def compute_n_samples_per_cluster_after_filtering(cluster_labels, n_raw, ratio):
    n_total = ceil(n_raw * ratio)
    counter = Counter(cluster_labels)
    n_cluster = max(cluster_labels) + 1
    n_samples_each_cluster = [0 for _ in range(n_cluster)]
    for cluster_id in range(n_cluster):
        n_this_cluster = int(counter[cluster_id] * ratio)
        n_this_cluster = max(1, n_this_cluster)
        n_samples_each_cluster[cluster_id] = n_this_cluster
    cluster_id = 0
    while sum(n_samples_each_cluster) < n_total:
        if cluster_id >= n_cluster:
            cluster_id = 0
        n_samples_each_cluster[cluster_id] += 1
        cluster_id += 1
    return n_samples_each_cluster


class Client(object):
    def __init__(self, idx, args, train_loader, train_loader_for_filtering=None):
        self.idx = idx
        self.args = args
        self.model = None
        self.filtering_model = None
        self.train_loader_for_filtering = train_loader_for_filtering
        
        self.train_loader = train_loader
        if self.args.filtering:
            self.train_iterator = None
        else:
            self.train_iterator = iter(self.train_loader)
        
        self.original_train_loader = train_loader
        
        self.last_select_indices = None  # only used by random
        
        self.device = torch.device(f'cuda:{args.device}')
        
        self.cluster_and_center = []


    def calculated_cluster_center(self):
        if self.args.filtering_model == 'same':
            self.model.to(self.device)
            flatten_hidden_state_list = get_flatten_features(self.model, self.original_train_loader, args=self.args)
        else:
            self.filtering_model.to(self.device)
            flatten_hidden_state_list = get_flatten_features(self.filtering_model, self.train_loader_for_filtering, args=self.args)

        if self.args.feature_layer == 'tsne':
            tsne = TSNE(n_components=self.args.compound_dim, perplexity=min(30, len(flatten_hidden_state_list) - 1))
            reduced_feature_list = tsne.fit_transform(np.array(flatten_hidden_state_list))
            cluster_labels, centroids, _ = clustering(reduced_feature_list, args=self.args)
        elif self.args.feature_layer == 'pca':
            pca = PCA(n_components=self.args.compound_dim)
            reduced_feature_list = pca.fit_transform(flatten_hidden_state_list)
            cluster_labels, centroids, _ = clustering(reduced_feature_list, args=self.args)
        elif self.args.feature_layer == 'kpca':
            pca = KernelPCA(n_components=self.args.compound_dim)
            reduced_feature_list = pca.fit_transform(flatten_hidden_state_list)
            cluster_labels, centroids, _ = clustering(reduced_feature_list, args=self.args)

        # stage the features
        self.reduced_feature_list = reduced_feature_list
        self.cluster_labels = cluster_labels
        self.centroids = centroids
        self.selected_clusters = []
        return zip(cluster_labels, centroids)

    
    def build_training_set_with_precalculated_clusters(self):
        select_sample_id_list = []
        for cluster_id in self.selected_clusters:
            sample_id_in_cluster = np.argwhere(self.cluster_labels == cluster_id).flatten()
            features_in_cluster = self.reduced_feature_list[sample_id_in_cluster]
            distances = np.linalg.norm(features_in_cluster - self.centroids[cluster_id], axis=1)
            select_sample_id_list.append(sample_id_in_cluster[np.argmin(distances)])
        if len(select_sample_id_list) > 0:
            np.random.shuffle(select_sample_id_list)
            subset_train = [self.original_train_loader.dataset[idx] for idx in select_sample_id_list]
            # replace train_loader with the filtered one
            self.train_loader = DataLoader(
                    subset_train, shuffle=True, batch_size=self.args.batch_size, collate_fn=self.original_train_loader.collate_fn
                )
            self.train_iterator = iter(self.train_loader)  
        else:
            self.train_iterator = None
    
        
    def local_train(self, cur_round, memory_record_dic=None):
        self.model.to(self.device)

        if memory_record_dic is not None:
            torch.cuda.empty_cache()
        
        lr = self.args.lr * math.pow(self.args.lr_decay, cur_round - 1)
        if self.args.batch_or_epoch == 'epoch':
            iter_steps = self.args.local_step * len(self.train_loader)
        else:
            iter_steps = self.args.local_step

        self.model.train()
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            raise NotImplementedError()
        # init batch progress bar
        if self.args.batch_or_epoch == 'batch':
            loss_total_train = 0.0
            num_trained = 0
            progress_bar = tqdm(range(iter_steps))

        for cur_step in range(iter_steps):
            # init epoch progress bar
            if self.args.batch_or_epoch == 'epoch':
                if cur_step % len(self.train_loader) == 0:
                    loss_total_train = 0.0
                    num_trained = 0
                    progress_bar = tqdm(range(len(self.train_loader)))

            try:
                batch = next(self.train_iterator)
            except StopIteration:
                self.train_iterator = iter(self.train_loader)
                batch = next(self.train_iterator)
                
            batch = {
                'input_ids': batch['input_ids'].to(self.device),
                'labels': batch['labels'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device) 
            }

            outputs = self.model(**batch)
            loss = outputs.loss
            if not torch.isnan(loss):
                loss.backward()
                optimizer.step()
                loss_total_train += loss
                num_trained += len(batch['input_ids'])
            optimizer.zero_grad()
            # finish this step
            progress_bar.update(1)

            if self.args.batch_or_epoch == 'epoch':
                progress_bar.set_description(f'client {self.idx} train at epoch {int(cur_step / len(self.train_loader)) + 1}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}')
            else:
                progress_bar.set_description(f'client {self.idx} train at step {cur_step}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}')
        del optimizer
        
        if memory_record_dic is not None:
            memory_record_dic[self.device.index] = {}
            memory_record_dic[self.device.index]['max_memory_allocated'] = torch.cuda.max_memory_allocated(self.device)
            memory_record_dic[self.device.index]['max_memory_reserved'] = torch.cuda.max_memory_reserved(self.device)
        self.model = self.model.cpu()


    def clear_model(self):
        # clear model to same memory
        self.model = None

        
    def migrate(self, device):
        """
        migrate a client to a new device
        """
        self.device = device
        
        
    def pull(self, forked_global_model):
        """
        pull model from the server
        """
        self.model = forked_global_model


    def pull_filtering_model(self, filtering_model):
        self.filtering_model = filtering_model
        
        
    def clear_filtering_model(self):
        # clear model to same memory
        self.filtering_model = None