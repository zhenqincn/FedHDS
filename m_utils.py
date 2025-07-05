import numpy as np
import torch
from tqdm import tqdm

def get_flatten_features(model, data_loader, args):
    # reduction with PCA, need to flat (n_layer, n_token) to (n_layer * n_token, )
    progress_bar_eval = tqdm(range(len(data_loader)))
    # hidden_state_history = []   # (layer, idx_data, feature)
    
    flatten_hidden_state_history = [] # (idx_data, feature)
    with torch.inference_mode():
        for batch in data_loader:
            batch = {
                'input_ids': batch['input_ids'].cuda(),
                'labels': batch['labels'].cuda(),
                'attention_mask': batch['attention_mask'].cuda() 
            }
            outputs = model(**batch)
            tmp_hidden_state = []
            for idx_layer in range(len(outputs.hidden_states)):
                # remove unnecessary batch dim
                hidden_state_cur_layer = torch.squeeze(outputs.hidden_states[idx_layer])
                if args.feature_token == 'avg':
                    tmp_hidden_state.append(torch.mean(hidden_state_cur_layer, dim=0).cpu().numpy())
                else:
                    tmp_hidden_state.append(hidden_state_cur_layer[-1].cpu().numpy())
            flatten_hidden_state_history.append(np.array(tmp_hidden_state, dtype=np.float64).reshape((len(tmp_hidden_state) * len(tmp_hidden_state[0]))))
            progress_bar_eval.update(1)
            progress_bar_eval.set_description(f'extracting feature: ')
    return flatten_hidden_state_history


def clustering(features, args):
    features_target_layer = features
    cluster_labels, centroids = _cluster(features_target_layer, args)
    return cluster_labels, centroids, features_target_layer


def _cluster(features, args):
    if args.clustering.lower() == 'kmeans':
        from sklearn.cluster import KMeans
        clusterer = KMeans(n_clusters=args.n_cluster, max_iter=2000, n_init=10, init='k-means++')
        clusterer.fit(features)
        cluster_labels = clusterer.labels_
        centroids = clusterer.cluster_centers_
        return cluster_labels, centroids
    elif args.clustering.lower() == 'hdbscan':
        from hdbscan import HDBSCAN
        clusterer = HDBSCAN(min_cluster_size=args.min_cluster, allow_single_cluster=False)
        clusterer.fit(features)
        cluster_labels = clusterer.labels_
        
        centroids = []
        if np.max(cluster_labels) > -1:
            for cluster_id in range(np.max(cluster_labels) + 1):
                centroids.append(clusterer.weighted_cluster_centroid(cluster_id))
        return cluster_labels, centroids