# FedHDS
Implementation of "Federated Data-Efficient Instruction Tuning for Large Language Models", which is accepted to ACL 2025 (Findings).


## Project Structure
```Markdown
.
├── utils_data
│   ├── default_tokens.py              // definitions of some special tokens
│   ├── llm_dataset.py                 // utilities to load Dolly-15K
│   ├── load_data.py                   // entrance to get dataloaders
│   ├── natural_instruction_loader.py  // utilities to load Natural Instructions
│   └── partition_data.py              // utilities to partition datasets in Dirichlet distribution
├── client.py
├── evaluations.py
├── m_utils.py
├── main.py
└── server.py
```

## Requirements
Please see `requirements.txt`.

## Data Preparation
1. Natural Instructions
To run experiments on [Natural Instructions](https://github.com/allenai/natural-instructions), you need to unzip the downloaded dataset in directory `./data`.

2. Dolly-15K
To run experiments on [Dolly-15K](https://github.com/databrickslabs/dolly), you need to download the corresponding dataset in directory `./data`, with its name as `databricks-dolly-15k.jsonl`.


## Running Examples
We provide some example scripts to run our approaches. 
The arguments can be adjusted according to the `help` information in their definitions.
To reproduce the results in our manuscript, please check the setups described in Section 5.1 and Appendix B.

1. FedHDS on Natural Instructions
```Shell
python main.py --seed 42 --rounds 40 --start_eval_epoch 20 --eval_subsampling --model openlm-research/open_llama_3b --use_prompts --dataset instruct --zeroshot --optimizer adam --peft --peft_method lora --lr 0.0003 --lr_decay 1.00 --local_step 1 -k 0.05 --filtering --min_cluster 5 --filtering_model same --kernel_ratio 0.05 --clustering hdbscan --feature_layer tsne --feature_token last --log --device 0
```


2. FedHDS on Dolly-15K
```Shell
python main.py --seed 42 --rounds 60 --start_eval_epoch 60 --eval_subsampling --model openlm-research/open_llama_3b --use_prompts --dataset dolly --num_clients 200 --iid dir0.5 --zeroshot --optimizer adam --peft --peft_method lora --lr 0.00003 --lr_decay 1.00 --local_step 1 -k 0.05 --filtering --min_cluster 2 --filtering_model same --kernel_ratio 0.05 --clustering hdbscan --feature_layer tsne --feature_token last --log --device 0
```


3. FedHDS-Turbo on Natural Instructions
```Shell
python main.py --seed 42 --rounds 40 --start_eval_epoch 20 --eval_subsampling --model openlm-research/open_llama_3b --use_prompts --dataset instruct --zeroshot --optimizer adam --peft --peft_method lora --lr 0.0003 --lr_decay 1.00 --local_step 1 -k 0.05 --filtering --min_cluster 5 --filtering_model gpt2 --kernel_ratio 0.05 --clustering hdbscan --feature_layer tsne --feature_token last --log --device 0
```


4. FedHDS-Turbo on Dolly-15K
```Shell
python main.py --seed 42 --rounds 60 --start_eval_epoch 60 --eval_subsampling --model openlm-research/open_llama_3b --use_prompts --dataset dolly --num_clients 200 --iid dir0.5 --zeroshot --optimizer adam --peft --peft_method lora --lr 0.00003 --lr_decay 1.00 --local_step 1 -k 0.05 --filtering --min_cluster 2 --filtering_model gpt2 --kernel_ratio 0.05 --clustering hdbscan --feature_layer tsne --feature_token last --log --device 0
```

# License
This project adopts the Apache-2.0 License. If the implementations and/or our paper were useful to you, please consider citing this
```bibtex
@article{qin2024federated,
  title={Federated Data-Efficient Instruction Tuning for Large Language Models},
  author={Qin, Zhen and Wu, Zhaomin and He, Bingsheng and Deng, Shuiguang},
  journal={arXiv preprint arXiv:2410.10926},
  year={2024}
}
```