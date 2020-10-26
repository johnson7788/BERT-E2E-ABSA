# BERT-E2E-ABSA
利用BERT端到端aspect-based的情感分析
<p align="center">
    <img src="architecture.jpg" height="400"/>
</p>

## Requirements
* python 3.7.3
* pytorch 1.2.0
* transformers 2.0.0
* numpy 1.16.4
* tensorboardX 1.9
* tqdm 4.32.1
* some codes are borrowed from **allennlp** ([https://github.com/allenai/allennlp](https://github.com/allenai/allennlp), an awesome open-source NLP toolkit) and **transformers** ([https://github.com/huggingface/transformers](https://github.com/huggingface/transformers), formerly known as **pytorch-pretrained-bert** or **pytorch-transformers**)

## Architecture
* Pre-trained embedding layer: BERT-Base-Uncased (12-layer, 768-hidden, 12-heads, 110M parameters)
* Task-specific layer: 
  - Linear
  - Recurrent Neural Networks (GRU)
  - Self-Attention Networks (SAN, TFM)
  - Conditional Random Fields (CRF)

## Dataset
* ~~Restaurant: retaurant reviews from SemEval 2014 (task 4), SemEval 2015 (task 12) and SemEval 2016 (task 5) (rest_total)~~
* (**Important**) Restaurant：SemEval 2014（rest14）的restaurant点评，SemEval 2015（rest15）的restaurant点评，SemEval 2016（rest16）的restaurant点评。请参考中的新更新文件```./data```
* (**Important**) 不要使用 ```rest_total``` ，请使用我们自己建立的数据集，更多详细信息请参见最下面的 [更新结果](#更新结果（重要）).
* Laptop: laptop reviews from SemEval 2014 (laptop14)


## Quick Start
* 此项目中有效的标签策略/方案（即表示文本或实体范围的方式）为**BIEOS**（也称为**BIOES**或**BMES**），**BIO**（也称为**IOB2**）和**OT**（也称为**IO**）。如果您不熟悉这些术语，强烈提议您在运行程序之前阅读以下材料：

  a. wiki [Inside–outside–beginning (tagging)](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)). 
  
  b. 论文[Representing Text Chunks](https://www.aclweb.org/anthology/E99-1023.pdf). 
  
  c. 这个项目的论文 [paper](https://www.aclweb.org/anthology/D19-5505.pdf) . 

* 在Restaurant and Laptop 数据集上重现结果：
  ```
  # 用5种不同的随机种子数训练模型
  python fast_run.py 
  ```
* 在其他ABSA数据集上训练模型：
  
  1. 将数据文件放 `./data/[YOUR_DATASET_NAME]` 在目录中， (请注意，您需要按照的输入格式`./data/laptop14/train.txt`重新组织数据文件，以便可以将其直接应用于该项目。).
  
  2. 在 `train.sh` 中设置 `TASK_NAME` 为 `[YOUR_DATASET_NAME]`.
  
  3. 训练模型:  `sh train.sh`

* (** **New feature** **) 使用训练过的ABSA模型对test/未见过的数据进行纯推理/直接转移

  1. 将数据文件放在目录中  `./data/[YOUR_EVAL_DATASET_NAME]`.
  
  2. `TASK_NAME` 为 `[YOUR_EVAL_DATASET_NAME]`
  
  3. 在 `work.sh` 中设置  `ABSA_HOME`  为 `[HOME_DIRECTORY_OF_YOUR_ABSA_MODEL]`
  
  4. 运行: `sh work.sh`

## Environment
* OS: REHL Server 6.4 (Santiago)
* GPU: NVIDIA GTX 1080 ti
* CUDA: 10.0
* cuDNN: v7.6.1

## main函数示例
```buildoutcfg
python main.py --model_type bert --absa_type linear --tfm_mode finetune --fix_tfm 0 --model_name_or_path bert-base-uncased --data_dir ./data/rest15 --task_name rest15 --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 8 --learning_rate 2e-5 --max_steps 1500 --warmup_steps 0 --do_train --do_eval --do_lower_case --seed 42 --tagging_schema BIEOS --overfit 0 --overwrite_output_dir --eval_all_checkpoints --MASTER_ADDR localhost --MASTER_PORT 28512
```

## colab 上运行
```buildoutcfg
! git clone https://github.com/johnson7788/BERT-E2E-ABSA
! pip install transformers==2.0.0 tqdm tensorboardX awscli awsebcli urllib3==1.25.10 botocore==1.18.18 --upgrade
! python main.py --model_type bert --absa_type linear --tfm_mode finetune --fix_tfm 0 --model_name_or_path bert-base-uncased --data_dir ./data/rest15 --task_name rest15 --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 8 --learning_rate 2e-5 --max_steps 1500 --warmup_steps 0 --do_train --do_eval --do_lower_case --seed 42 --tagging_schema BIEOS --overfit 0 --overwrite_output_dir --eval_all_checkpoints --MASTER_ADDR localhost --MASTER_PORT 28512
```
## 更新结果（重要）
* ```rest_total``` 数据集的数据文件是通过将```rest14```, ```rest15``` and ```rest16``` 拼接得到的， 
我们的动机是构建一个更大的训练/测试数据集，以稳定训练/忠实地反映ABSA模型的能力。
但是，我们最近发现SemEval组织者将```rest15.train``` and ```rest15.test```的联合集直接视为rest16的训练集（即 ```rest16.train``` ），
因此在```rest_total.train``` and the ```rest_total.test```之间存在重叠，
这使得该数据集无效。当您按照我们的E2E-ABSA任务进行工作时，我们希望您不要再使用此 ```rest_total```数据集，
而应改为正式发布的```rest14```, ```rest15``` and ```rest16```.
* 为了便于将来进行比较，我们按照上述设置重新运行模型，并在```rest14```, ```rest15``` and ```rest16```上报告结果：

    | Model | rest14 | rest15 | rest16 |
    | --- | --- | --- | --- |
    | E2E-ABSA (OURS) | 67.10 | 57.27 | 64.31 |
    | [He et al. (2019)](https://arxiv.org/pdf/1906.06906.pdf) | 69.54 | 59.18 | - |
    | [Liu et al. (2020)](https://arxiv.org/pdf/2004.06427.pdf) | 68.91 | 58.37 | - |
    | BERT-Linear (OURS) | 72.61 | 60.29 | 69.67 |
    | BERT-GRU (OURS) | 73.17 | 59.60 | 70.21 |
    | BERT-SAN (OURS) | 73.68 | 59.90 | 70.51 |
    | BERT-TFM (OURS) | 73.98 | 60.24 | 70.25 |
    | BERT-CRF (OURS) | 73.17 | 60.70 | 70.37 |
    | [Chen and Qian (2019)](https://www.aclweb.org/anthology/2020.acl-main.340.pdf)| 75.42 | 66.05 | - |
    | [Liang et al. (2020)](https://arxiv.org/pdf/2004.01951.pdf)| 72.60 | 62.37 | - |

## Citation
If the code is used in your research, please star our repo and cite our paper as follows:
```
@inproceedings{li-etal-2019-exploiting,
    title = "Exploiting {BERT} for End-to-End Aspect-based Sentiment Analysis",
    author = "Li, Xin  and
      Bing, Lidong  and
      Zhang, Wenxuan  and
      Lam, Wai",
    booktitle = "Proceedings of the 5th Workshop on Noisy User-generated Text (W-NUT 2019)",
    year = "2019",
    url = "https://www.aclweb.org/anthology/D19-5505",
    pages = "34--41"
}
```
     
