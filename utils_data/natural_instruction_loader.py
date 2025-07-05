import os
import json
from torch.utils.data import Dataset, ConcatDataset, DataLoader, random_split
import copy
import pickle
import numpy as np
from dataclasses import dataclass
import transformers
import torch


IGNORE_INDEX = -100


class LLMDataset(Dataset):
    def __init__(self,
                 data,
                 tokenizer,
                 use_prompts,
                 generation=False):
        super(LLMDataset, self).__init__()
        
        if use_prompts:
            sources = [f'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{example[0]}\n\n### Input:\n{example[1]}\n\n### Response:' for example in data]
        else:
            sources = [f'{example[0]}\n\nInput: {example[1]}\n\nOutput:' for example in data]
        targets = [f'{example[2]}{tokenizer.eos_token}' for example in data]

        data_dict = self.preprocess(sources, targets, tokenizer, generation)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]


    def _tokenize_fn(self, strings, tokenizer):
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ) for text in strings
        ]
        input_ids = labels = [
            tokenized.input_ids[0] for tokenized in tokenized_list
        ]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
            for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def preprocess(self, sources, targets, tokenizer, generation):
        if generation:
            sources_tokenized, labels_tokenized = [
                self._tokenize_fn(strings, tokenizer)
                for strings in (sources, targets)
            ]
            input_ids = self._tokenize_fn(sources, tokenizer)["input_ids"]
            labels = self._tokenize_fn(targets, tokenizer)["input_ids"]
        else:
            examples = [s + t for s, t in zip(sources, targets)]
            examples_tokenized, sources_tokenized = [
                self._tokenize_fn(strings, tokenizer)
                for strings in (examples, sources)
            ]
            input_ids = examples_tokenized["input_ids"]
            labels = copy.deepcopy(input_ids)
            for label, source_len in zip(labels,
                                        sources_tokenized["input_ids_lens"]):
                label[:source_len] = IGNORE_INDEX
        return dict(input_ids=input_ids, labels=labels)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i],
                    labels=self.labels[i])


@dataclass
class LLMDataCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        

def _get_task_splits(use_original_testset=False):
    with open(os.path.join(os.path.expanduser('~'), '.datasets', 'natural-instructions-2.8', 'splits', 'default', 'train_tasks.txt'), 'r') as reader:
        train_set_names = [f'{content.strip()}.json' for content in reader.readlines()]
    with open(os.path.join(os.path.expanduser('~'), '.datasets', 'natural-instructions-2.8', 'splits', 'default', 'test_tasks.txt'), 'r') as reader:
        eval_set_names = [f'{content.strip()}.json' for content in reader.readlines()]
    if not use_original_testset:
        train_set_names.extend(eval_set_names)
        return train_set_names, []
    else:
        return train_set_names, eval_set_names


def _filter_out_over_length(items, max_length):
    return [item for item in items if len(item['input']) < max_length]


def get_instruction_dataset(args, tokenizer, from_cache=False, only_eval=False):
    """
    only_eval: only effective with zeroshot set to `True`
    """
    if from_cache:
        list_train_loader = []
        for client_idx in range(args.num_clients):
            with open(os.path.join(os.path.expanduser('~'), '.datasets', 'natural-instructions-2.8', 'processed', f'train_loader_{client_idx}.pkl'), 'rb') as reader:
                list_train_loader.append(pickle.load(reader))
        with open(os.path.join(os.path.expanduser('~'), '.datasets', 'natural-instructions-2.8', 'processed', f'eval_loader.pkl'), 'rb') as reader:
            eval_loader = pickle.load(reader)
    else:
        if not args.zeroshot:
            train_set_names, _ = _get_task_splits(use_original_testset=args.zeroshot)
            list_train_loader = []
            data_collator = LLMDataCollator(tokenizer=tokenizer)
            
            list_eval_set = []
            if not os.path.exists(os.path.join(os.path.expanduser('~'), '.datasets', 'natural-instructions-2.8', 'processed')):
                os.makedirs(os.path.join(os.path.expanduser('~'), '.datasets', 'natural-instructions-2.8', 'processed'))
            for file_name in train_set_names:
                with open(os.path.join(os.path.expanduser('~'), '.datasets', 'natural-instructions-2.8', 'tasks', file_name)) as reader:
                    raw_data = json.load(reader)
                    instances = _filter_out_over_length(raw_data['Instances'], max_length=args.max_length)
                    if len(instances) < 30:
                        continue
                    # sample 20% dataset
                    instances = np.random.choice(instances, int(len(instances) * 0.2) + 20, replace=False)
                    print(file_name, len(instances), max([len(item['input']) for item in instances]))
                    instruct = raw_data['Definition'][0]

                    if args.eval_metrics == 'none':
                        data = []
                        for item in instances:
                            # only take the first output into consideration
                            data.append((instruct, item['input'], item['output'][0]))
                        dataset = LLMDataset(data, tokenizer, use_prompts=args.use_prompts)
                        # at least there are 2 samples
                        num_eval = max(len(dataset) - int(len(dataset) * 0.98), 2)
                        dataset, eval_subset = random_split(dataset, [len(dataset) - num_eval, num_eval])
                        list_eval_set.append(eval_subset)
                        list_train_loader.append(DataLoader(dataset, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator))
                    else:
                        num_eval = max(len(instances) - int(len(instances) * 0.98), 20)
                        index_eval = np.random.choice(np.arange(len(instances)), size=num_eval, replace=False)
                        # delete the indices of eval samples from the all set
                        index_train = np.delete(np.arange(len(instances)), index_eval)
                        data = []
                        for item in instances:
                            data.append((instruct, item['input'], item['output'][0]))
                        data = np.array(data)
                        list_eval_set.append(LLMDataset(data[index_eval], tokenizer, use_prompts=args.use_prompts, generation=True))
                        list_train_loader.append(DataLoader(LLMDataset(data[index_train], tokenizer, use_prompts=args.use_prompts, generation=False), shuffle=True, batch_size=args.batch_size, collate_fn=data_collator))
            universal_eval_set = ConcatDataset(list_eval_set)
            eval_loader = DataLoader(universal_eval_set, shuffle=False, batch_size=args.batch_size, collate_fn=data_collator)
        else:
            train_set_names, eval_set_names = _get_task_splits(use_original_testset=args.zeroshot)
            list_train_loader = []
            data_collator = LLMDataCollator(tokenizer=tokenizer)
            if not os.path.exists(os.path.join(os.path.expanduser('~'), '.datasets', 'natural-instructions-2.8', 'processed')):
                os.makedirs(os.path.join(os.path.expanduser('~'), '.datasets', 'natural-instructions-2.8', 'processed'))
            
            # if only_eval, the following lines won't be executed to save time.
            if not only_eval:
                print('load train sets')
                for file_name in train_set_names:
                    if len(list_train_loader) >= args.num_clients:
                        break
                    with open(os.path.join(os.path.expanduser('~'), '.datasets', 'natural-instructions-2.8', 'tasks', file_name)) as reader:
                        raw_data = json.load(reader)
                        instances = _filter_out_over_length(raw_data['Instances'], max_length=args.max_length)
                        # sample 20% dataset
                        if len(instances) < 20:
                            continue
                        instances = np.random.choice(instances, int(len(instances) * 0.2), replace=False)
                        print(file_name, len(instances), max([len(item['input']) for item in instances]))
                        instruct = raw_data['Definition'][0]
                        data = []
                        for item in instances:
                            # only take the first output into consideration
                            data.append((instruct, item['input'], item['output'][0]))
                        dataset = LLMDataset(data, tokenizer, use_prompts=args.use_prompts)
                        list_train_loader.append(DataLoader(dataset, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator))
                args.num_clients = len(list_train_loader)
            print('load eval')
            list_eval_set = []
            for file_name in eval_set_names:
                with open(os.path.join(os.path.expanduser('~'), '.datasets', 'natural-instructions-2.8', 'tasks', file_name)) as reader:
                    raw_data = json.load(reader)
                    instruct = raw_data['Definition'][0]
                    instances = _filter_out_over_length(raw_data['Instances'], max_length=args.max_length)
                    if args.eval_subsampling:
                        if len(instances) > 20:
                            instances = np.random.choice(instances, max(20, int(0.02 * len(instances))), replace=False)
                    data = []
                    for item in instances:
                        # only take the first output into consideration
                        data.append((instruct, item['input'], item['output'][0]))
                    if args.eval_metrics == 'none':
                        list_eval_set.append(LLMDataset(data, tokenizer, use_prompts=args.use_prompts, generation=False))
                    else:
                        list_eval_set.append(LLMDataset(data, tokenizer, use_prompts=args.use_prompts, generation=True))
            universal_eval_set = ConcatDataset(list_eval_set)
            eval_loader = DataLoader(universal_eval_set, shuffle=False, batch_size=args.batch_size, collate_fn=data_collator)
    return list_train_loader, eval_loader
