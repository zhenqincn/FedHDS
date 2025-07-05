import argparse
import os
import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from hdbscan import HDBSCAN


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # federation
    parser.add_argument('--num_clients', type=int, default=738, help='natural instructions datasets contains 738 training tasks in its default split')
    parser.add_argument('-k', type=float, default=0.05)
    parser.add_argument('--rounds', type=int, default=40)
    parser.add_argument('--batch_or_epoch', type=str, default='epoch', choices=['epoch', 'batch'])
    parser.add_argument('--local_step', type=int, default=1)
    parser.add_argument('--equal_weight', default=False, action='store_true', help='whether the weights of clients are the same during aggregation')

    # data
    parser.add_argument('--dataset', type=str, default='alpaca')
    parser.add_argument('--data_sample', type=float, default=1.0)
    parser.add_argument('--iid', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=1024, help='the max number of tokens of a data sample')
    parser.add_argument('--zeroshot', default=True, action='store_true')
    parser.add_argument('--zerotask', default='7', type=str)
    parser.add_argument('--split', type=str, default='[0.98, 0.01, 0.01]', help='train, val, test split ratio, only works when `zeroshot` is `False`')
    parser.add_argument('--train_eval_ratio', default='[0.99, 0.01]', type=str, help='only works when `zeroshot` is `False`')
    parser.add_argument('--use_prompts', default=False, action='store_true')

    # Filtering
    parser.add_argument('--filtering', action='store_true', default=False)
    parser.add_argument('--feature_layer', default='-1', type=str, help='optional in -1, auto, elbow, pca, kpca, tsne, 0, 1, 2, etc.')
    parser.add_argument('--compound_dim', default=2, type=int)
    parser.add_argument('--feature_token', default='avg', type=str, choices=['avg', 'last'])
    parser.add_argument('--clustering_score', default='ch', type=str, choices=['ch', 'sc', 'db'])
    parser.add_argument('--clustering', type=str, default='kmeans', choices=['kmeans', 'hdbscan'])
    parser.add_argument('--n_cluster', type=int, default=7, help='will not work when dnscan is applied')
    parser.add_argument('--kernel_ratio', type=float, default=1.0, help='the ratio of filtered data samples to the raw set')
    parser.add_argument('--filtering_model', type=str, default='same')
    parser.add_argument('--dp_noise', type=float, default=0.0)
    parser.add_argument('--min_cluster', type=int, default=5)
    
    # model
    parser.add_argument('--model', type=str, default='datajuicer/LLaMA-1B-dj-refine-150B')
    parser.add_argument('--peft', action='store_true', default=False)
    parser.add_argument('--peft_method', default='lora', type=str, choices=['lora', 'prefix', 'p-tuning', 'prompt'])
    
    # training
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=1.0)
    parser.add_argument('--grad_clip', type=float, default=-100.0)
    
    
    # enviroment
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log', default=False, action='store_true')
    parser.add_argument('--log_root', default='')
    parser.add_argument('--seed', default=42, type=int, help='global seed')

    # Evaluation
    parser.add_argument('--eval_metrics', default='none', type=str)
    parser.add_argument('--generate_eval', default='rouge', type=str, choices=['rouge', 'bleu'])
    parser.add_argument('--eval_subsampling', default=False, action='store_true')
    parser.add_argument('--full_evaluation', default=False, action='store_true')
    parser.add_argument('--start_eval_epoch', default=30, type=int)
    parser.add_argument('--eval_interval', default=20, type=int)
    parser.add_argument('--loss', default=False, action='store_true')
    
    
    # ckpt
    parser.add_argument('--save', default=False, action='store_true')
    # parser.add_argument('--eval_step', type=int, default=0, help='0 means eval per epoch')

    time_stamp = str(time.time())
    args = parser.parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)
    import random
    import numpy as np
    import torch
    from server import Server
    from client import Client
    from utils_data.load_data import get_loaders, get_loaders_for_filtering
    import numpy as np
    
    import yaml
    from copy import deepcopy
    import json

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    
    train_avg_acc = []
    eval_avg_acc = []
    metric_history = {}
    memory_record_dic = {}
    
    train_time_history = []
    extract_time_history = []
    gen_time_history = []
    
    setup_seed(args.seed)
    import time
    start_time = time.time()
    list_train_loader, eval_loader, _ = get_loaders(args)
    end_time = time.time()
    print('loaded with', end_time - start_time, 'seconds')
    print('totally', len(eval_loader.dataset), 'samples for eval')

    list_train_loader_for_filtering = None
    if args.filtering_model != 'same':
        setup_seed(args.seed)
        list_train_loader_for_filtering, _, _ = get_loaders_for_filtering(args)
    
    model_path = args.model
    if 'home' in model_path or 'mnt' in model_path:
        model_path = model_path.split('/')[-1]
    
    if args.dataset == 'instruct':
        args.iid = 'meta'
        
    if args.lr_decay == 1.0:
        log_dir = os.path.join('exp', args.optimizer, f'{args.dataset}-iid{args.iid}', f'zeroshot{"" if args.zerotask == "7" else str(args.zerotask)}' if args.zeroshot else 'fewshot', model_path, str(args.lr), f'step{args.local_step}', f'seed={args.seed}', time_stamp)
    else:
        log_dir = os.path.join('exp_lrdecay', args.optimizer, f'{args.dataset}-iid{args.iid}', f'zeroshot{"" if args.zerotask == "7" else str(args.zerotask)}' if args.zeroshot else 'fewshot', model_path, str(args.lr), f'step{args.local_step}', f'seed={args.seed}', time_stamp)

    if args.log_root != '':
        log_dir = os.path.join(args.log_root, log_dir)
    if args.log:
        os.makedirs(log_dir)
    config = yaml.dump(args, None)
    config = '\n'.join(config.split('\n')[1:])
    print('Configs: ')
    print(config)
    print('=====================')
    if args.log:
        with open(os.path.join(log_dir, 'config.yaml'), 'w') as writer:
            writer.write(config)
            
    args.device = 0
    client_indices_rounds = []
    for _ in range(args.rounds):
        client_indices_rounds.append(np.random.choice(np.arange(args.num_clients), size=int(args.num_clients * args.k), replace=False))

    client_list = []

    if args.full_evaluation:
        args.eval_subsampling = False
    
    previous_metric = args.eval_metrics
    if args.dataset in ['dolly', 'alpaca', 'gsm8k', 'code', 'instruct', 'flan']:
        args.eval_metrics = args.generate_eval
    else:
        args.eval_metrics = 'acc'
    # reset seed to get the same eval loaders
    setup_seed(args.seed)
    _, eval_loader_full, main_tokenizer = get_loaders(args, only_eval=True)
    args.eval_metrics = previous_metric

    server = Server(args, eval_loader=eval_loader, log_dir=log_dir)

    if list_train_loader_for_filtering is None:
        list_train_loader_for_filtering = [None for _ in range(args.num_clients)] 
    for idx in range(args.num_clients):
        client_list.append(Client(idx, args, list_train_loader[idx], list_train_loader_for_filtering[idx]))
    
    if args.loss:
        eval_result = server.eval(cur_round=0, eval_avg_acc=eval_avg_acc)
        eval_avg_acc.append(eval_result)
    
    if args.log:
        with open(os.path.join(log_dir, 'memory.json'), 'w') as writer:
            json.dump(memory_record_dic, writer)
        with open(os.path.join(log_dir, 'results.json'), 'w') as writer:
            json.dump({
                # 'train_avg_acc': train_avg_acc,
                'eval_avg_acc': eval_avg_acc
            }, writer)
    
    # Sequentially train models
    for r in range(1, args.rounds + 1):
        print(f'start epoch {r}...')
        selected_client = [client_list[i] for i in client_indices_rounds[r-1]]
        server.prepare_aggregate()

        staged_clusterid_clientid_centroid = []
        
        if args.filtering_model != 'same':
            server.filtering_model.to(server.device)
        
        for client in selected_client:
            if args.filtering_model == 'same':
                client.pull(deepcopy(server.model))
            else:
                client.pull_filtering_model(server.filtering_model)
            start_time = time.time()
            for cluster_id, cluster_centroid in client.calculated_cluster_center():
                if args.dp_noise > 0.0:
                    cluster_centroid = np.tanh(cluster_centroid)
                    cluster_centroid += np.random.randn(*cluster_centroid.shape) * args.dp_noise
                staged_clusterid_clientid_centroid.append((cluster_id, client.idx, cluster_centroid))
            end_time = time.time()
            extract_time_history.append([end_time - start_time, len(client.original_train_loader)])
            if args.filtering_model == 'same':
                client.clear_model()
        if args.filtering_model != 'same':
            server.filtering_model.to(torch.device('cpu'))
        center_list = np.array([item[2] for item in staged_clusterid_clientid_centroid])
        clusterer = HDBSCAN(min_cluster_size=2, allow_single_cluster=False)
        clusterer.fit(center_list)
        cluster_labels = clusterer.labels_
        
        selected_cluster_id = []
        if np.max(cluster_labels) > -1:
            for cluster_id in range(np.max(cluster_labels) + 1):
                tmp_center = clusterer.weighted_cluster_centroid(cluster_id)
                sample_id_in_cluster = np.argwhere(cluster_labels == cluster_id).flatten()
                features_in_cluster = center_list[sample_id_in_cluster]
                distances = np.linalg.norm(features_in_cluster - tmp_center, axis=1)
                selected_cluster_id.append(sample_id_in_cluster[np.argmin(distances)])
        print(f'totally {len(center_list)} clusters, selected {len(selected_cluster_id)} of them')
        for idx in selected_cluster_id:
            cluster_id, client_id = staged_clusterid_clientid_centroid[idx][0], staged_clusterid_clientid_centroid[idx][1]
            client_list[client_id].selected_clusters.append(cluster_id)
        for client in selected_client:
            client.build_training_set_with_precalculated_clusters()
                
        selected_client = [client for client in selected_client if client.train_iterator is not None]
        for client in selected_client:
            client.pull(deepcopy(server.model))
            start_time = time.time()
            client.local_train(cur_round=r, memory_record_dic=memory_record_dic)
            end_time = time.time()
            train_time_history.append([end_time - start_time, len(client.train_loader) if args.batch_or_epoch == 'epoch' else args.local_step])

            server.online_aggregate(client, selected_client)
            client.clear_model()
        with open(os.path.join(log_dir, 'train_time.json'), 'w') as writer:
            json.dump(train_time_history, writer)
            
        with open(os.path.join(log_dir, 'extract_time.json'), 'w') as writer:
            json.dump(extract_time_history, writer)
            
        server.finish_aggregate()
        if args.loss:
            eval_result = server.eval(cur_round=r, eval_avg_acc=eval_avg_acc)
            eval_avg_acc.append(eval_result)
        if args.log:
            with open(os.path.join(log_dir, 'memory.json'), 'w') as writer:
                json.dump(memory_record_dic, writer)
            with open(os.path.join(log_dir, 'results.json'), 'w') as writer:
                json.dump({
                    # 'train_avg_acc': train_avg_acc,
                    'eval_avg_acc': eval_avg_acc
                }, writer)
            if r % args.eval_interval == 0 and r >= args.start_eval_epoch:
                sampled_eval_loader = server.eval_loader
                server.eval_loader = eval_loader_full
                previous_metric = args.eval_metrics
                if args.dataset in ['dolly', 'alpaca', 'gsm8k', 'code', 'instruct', 'flan']:
                    args.eval_metrics = args.generate_eval
                else:
                    args.eval_metrics = 'acc'
                eval_result = server.eval(cur_round=r, eval_avg_acc=eval_avg_acc)
                metric_history[r] = eval_result
                with open(os.path.join(log_dir, 'ckpt_eval.json'), 'w') as writer:
                    json.dump({
                        # 'train_avg_acc': train_avg_acc,
                        f'eval_metrics_{args.eval_metrics}': metric_history
                    }, writer)
                server.eval_loader = sampled_eval_loader
                args.eval_metrics = previous_metric

                server.eval_loader = sampled_eval_loader
                args.eval_metrics = previous_metric
                # print(f'final round {args.eval_metrics}: {eval_result}')

    if args.dataset in ['dolly', 'alpaca', 'gsm8k', 'code', 'instruct', 'flan']:
        args.eval_metrics = args.generate_eval
    else:
        args.eval_metrics = 'acc'
    # reset seed to have a eval_loader with the same data samples
    setup_seed(args.seed)
    _, eval_loader_final, _ = get_loaders(args, only_eval=True)
    server.eval_loader = eval_loader_final
    
    start_time = time.time()
    eval_result = server.eval(cur_round=args.rounds, eval_avg_acc=eval_avg_acc)
    end_time = time.time()
    
    with open(os.path.join(log_dir, 'gen_time.json'), 'w') as writer:
        json.dump([end_time - start_time, len(eval_loader_final)], writer)

    if args.log:
        with open(os.path.join(log_dir, 'final_eval.json'), 'w') as writer:
            json.dump({
                f'final_eval_{args.eval_metrics}': eval_result
            }, writer)
    print(f'final round {args.eval_metrics}: {eval_result}')
