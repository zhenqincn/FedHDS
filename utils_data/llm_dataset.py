import copy
import logging
import pandas as pd

from enum import Enum
from torch.utils.data import Dataset
import json
import os
import gzip
from dataclasses import dataclass
import torch
import transformers


logger = logging.getLogger(__name__)


def load_jsonl(file_path,
               instruction='instruction',
               input='input',
               output='output',
               category='category',
               is_gzip=False):
    list_data_dict = []
    open_func = open if not is_gzip else gzip.open
    with open_func(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            new_item = dict(
                instruction=item[instruction] if instruction in item else None,
                input=item[input] if input in item else None,
                output=item[output] if output in item else None,
                category=item[category] if category in item else None)
            item = new_item
            list_data_dict.append(item)
    return list_data_dict


def load_json(file_path,
              instruction='instruction',
              input='input',
              output='output',
              category='category'):
    with open(file_path, 'r', encoding="utf-8") as f:
        list_data_dict = json.load(f)

    # Replace key
    new_list_data_dict = []
    for item in list_data_dict:
        new_item = dict(
            instruction=item[instruction] if instruction in item else None,
            input=item[input] if input in item else None,
            output=item[output] if output in item else None,
            category=item[category] if category in item else None)
        new_list_data_dict.append(new_item)
    return new_list_data_dict

class DefaultToken(Enum):
    PAD_TOKEN = "[PAD]"
    EOS_TOKEN = "</s>"
    BOS_TOKEN = "<s>"
    UNK_TOKEN = "<unk>"
    IGNORE_INDEX = -100


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, "
        "paired with an input that provides further context. "
        "Write a response for the task request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:"
        "\n{input}\n\n### Response:"),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response for the task request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"),
}


class LLMDataset(Dataset):
    def __init__(self,
                 dataset,
                 tokenizer,
                 prompt_input=PROMPT_DICT["prompt_input"],
                 prompt_no_input=PROMPT_DICT["prompt_no_input"], generation=False):
        super(LLMDataset, self).__init__()
        if dataset == 'alpaca':
            json_name = 'alpaca_data.json'
            list_data_dict = load_json(os.path.join('data', json_name))
        elif dataset == 'dolly':
            json_name = 'databricks-dolly-15k.jsonl'
            list_data_dict =  load_jsonl(os.path.join('data', json_name), 
                                        instruction='instruction',
                                        input='context',
                                        output='response',
                                        category='category')
        elif dataset == 'gsm8k':
            json_name = 'gsm8k_train.jsonl'
            list_data_dict =  load_jsonl(os.path.join('data', json_name), 
                                        instruction='question',
                                        output='answer')
        elif dataset == 'code':
            json_name = 'rosetta_alpaca.json'
            list_data_dict =  load_json(os.path.join('data', json_name), 
                                            instruction='instruction',
                                            input='input',
                                            output='output',
                                            category='input')
            
        elif dataset == 'flan':
            import pwd
            user_name = pwd.getpwuid(os.getuid())[0]
            base_dir_train = f'/home/{user_name}/.cache/huggingface/hub/datasets--Muennighoff--flan/snapshots/049702ca5ffc5b3c62bf226aac67ba1493e8d548/train'
            base_dir_test = f'/home/{user_name}/.cache/huggingface/hub/datasets--Muennighoff--flan/snapshots/049702ca5ffc5b3c62bf226aac67ba1493e8d548/test'

            list_data_dict = []
            cnt = 0
            task_cnt = 0
            for data_name in os.listdir(base_dir_test):
                data_dicts_cur_task = load_jsonl(os.path.join(base_dir_test, data_name),
                                                instruction='instruction',
                                                input='inputs',
                                                output='targets',
                                                category='task',
                                                is_gzip=False
                                                )
                cnt += len(data_dicts_cur_task)
                task_cnt += 1
                list_data_dict.extend(data_dicts_cur_task)
            # print(task_cnt, cnt)
            # print(len(list_data_dict))
            print('loaded all data, preparing tokenizing')
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != ""
            else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [
            f"{example['output']}{tokenizer.eos_token}"
            for example in list_data_dict
        ]

        data_dict = self.preprocess(sources, targets, tokenizer, generation=generation)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

        categories = [
            example['category'] if 'category' in example else None
            for example in list_data_dict
        ]
        df = pd.DataFrame(categories, columns=["category"])
        self.categories = list(pd.Categorical(df["category"]).codes)

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
                label[:source_len] = DefaultToken.IGNORE_INDEX.value
        return dict(input_ids=input_ids, labels=labels)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i],
                    labels=self.labels[i],
                    categories=self.categories[i])


@dataclass
class LLMDataCollator(object):
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
            padding_value=DefaultToken.IGNORE_INDEX.value)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )