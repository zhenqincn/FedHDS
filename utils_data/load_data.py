import torch
import torch.utils
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
from utils_data.default_tokens import DefaultToken
from utils_data.partition_data import *
from collections import Counter


def load_tokenizer(model_name, args):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.model_max_length = args.max_length
    special_tokens = dict()
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = DefaultToken.PAD_TOKEN.value
    if tokenizer.eos_token is None:
        special_tokens["eos_token"] = DefaultToken.EOS_TOKEN.value
    if tokenizer.bos_token is None:
        special_tokens["bos_token"] = DefaultToken.BOS_TOKEN.value
    if tokenizer.unk_token is None:
        special_tokens["unk_token"] = DefaultToken.UNK_TOKEN.value

    tokenizer.add_special_tokens(special_tokens)
    return tokenizer

def get_loaders_for_filtering(args, only_eval=False):
    tmp_model = args.model
    args.model = args.filtering_model
    var1, var2, var3 = get_loaders(args, only_eval=only_eval)
    args.model = tmp_model
    return var1, var2, var3


def get_loaders(args, only_eval=False):
    """
    Return: list of train_loaders, eval_loader
    """
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.model_max_length = args.max_length
    special_tokens = dict()
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = DefaultToken.PAD_TOKEN.value
    if tokenizer.eos_token is None:
        special_tokens["eos_token"] = DefaultToken.EOS_TOKEN.value
    if tokenizer.bos_token is None:
        special_tokens["bos_token"] = DefaultToken.BOS_TOKEN.value
    if tokenizer.unk_token is None:
        special_tokens["unk_token"] = DefaultToken.UNK_TOKEN.value

    tokenizer.add_special_tokens(special_tokens)

    # tokenizer.pad_token = tokenizer.eos_token

    # Generation task
    if args.dataset in ['dolly']:
        from utils_data.llm_dataset import LLMDataset, LLMDataCollator
        import json
        if args.eval_metrics == 'none':
            raw_datasets = LLMDataset(args.dataset, tokenizer=tokenizer, generation=False)
        else:
            raw_datasets = LLMDataset(args.dataset, tokenizer=tokenizer, generation=True)

        data_collator = LLMDataCollator(tokenizer=tokenizer)

        train_ratio, val_ratio, _ = json.loads(args.split)
        # only use a subset of raw dataset
        raw_datasets, _ = torch.utils.data.dataset.random_split(raw_datasets, [int(len(raw_datasets) * args.data_sample), len(raw_datasets) - int(len(raw_datasets) * args.data_sample)])
        if args.zeroshot:
            y_all = np.array([item['categories'] for item in raw_datasets])
            if '[' in args.zerotask:
                zerotasks = json.loads(args.zerotask)
                index_eval = []
                for t in zerotasks:
                    index_eval.extend(np.where(y_all == int(t))[0])
                index_eval = np.array(index_eval)
            else:
                index_eval = np.where(y_all == int(args.zerotask))[0]
            # delete the indices of eval samples from the all set
            index_train = np.delete(np.arange(len(y_all)), index_eval)
            raw_datasets = np.array(raw_datasets)
            train_set = raw_datasets[index_train]
            eval_set = raw_datasets[index_eval]
            y_train = np.array([item['categories'] for item in train_set])
        else:
            train_set, val_set, eval_set = torch.utils.data.dataset.random_split(raw_datasets, [int(len(raw_datasets) * train_ratio), int(len(raw_datasets) * val_ratio), len(raw_datasets) - int(len(raw_datasets) * train_ratio) - int(len(raw_datasets) * val_ratio)])
            y_train = np.array([item['categories'] for item in train_set])
        counter = Counter(y_train)
        noniid = args.iid
        if 'dir' in noniid:
            split_dic = partition_idx_labeldir(y_train, n_parties=args.num_clients, alpha=float(noniid[3:]), num_classes=len(counter))
            split_trainsets = []
            for client_id, sample_indices in split_dic.items():
                split_trainsets.append(Subset(train_set, indices=sample_indices))
        else:
            noniid = int(noniid)
            if noniid == 0:
                n_parts = [int(len(train_set) / args.num_clients) for _ in range(args.num_clients - 1)]
                n_parts.append(len(train_set) - sum(n_parts))
                split_trainsets = torch.utils.data.dataset.random_split(train_set, n_parts)
            else:
                split_dic = partition_idx_labelnoniid(y_train, n_parties=args.num_clients, label_num=int(args.iid), num_classes=len(counter))
                split_trainsets = []
                for client_id, sample_indices in split_dic.items():
                    split_trainsets.append(Subset(train_set, indices=sample_indices))

        list_train_loader = [
            DataLoader(
                subset, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator
            ) for subset in split_trainsets
        ]
        eval_loader = DataLoader(
            eval_set, batch_size=args.batch_size, collate_fn=data_collator
        )
    elif args.dataset in ['instruct']:
        from utils_data.natural_instruction_loader import get_instruction_dataset
        # num_task_train
        list_train_loader, eval_loader = get_instruction_dataset(args, tokenizer, only_eval=only_eval)
    else:
        raise AttributeError(f'dataset {args.dataset} not implemented')
    return list_train_loader, eval_loader, tokenizer