import argparse
import os
import torch
import numpy as np

from glue_utils import convert_examples_to_seq_features, compute_metrics_absa,processors
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer, XLNetConfig, XLNetTokenizer, WEIGHTS_NAME
from absa_layer import BertABSATagger
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from seq_utils import ot2bieos_ts, bio2ot_ts, tag2ts

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertABSATagger, BertTokenizer),
}


def load_and_cache_examples(args, task, tokenizer):
    """
    类似main.py函数
    :param args:
    :param task: 要加载的task
    :param tokenizer:  实例化好的tokenizer
    :return:  dataset, all_evaluate_label_ids, total_words句子的列表
    """
    processor = processors[task]()
    # 设定cached_features_file的名字， 例如'./data/rest15/cached_test_bert-base-uncased_128_rest15'
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'test',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    # 如果存在cached_features_file，直接加载
    if os.path.exists(cached_features_file):
        print("cached_features_file:", cached_features_file)
        features = torch.load(cached_features_file)
        examples = processor.get_test_examples(args.data_dir, args.tagging_schema)
    else:
        # 创建cached_features_file
        # label_list是所有的labels
        label_list = processor.get_labels(args.tagging_schema)
        # test.txt文件的样本
        examples = processor.get_test_examples(args.data_dir, args.tagging_schema)
        # 文本样本通过tokenizer转换成标签
        features = convert_examples_to_seq_features(examples=examples, label_list=label_list, tokenizer=tokenizer,
                                                    cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                    cls_token=tokenizer.cls_token,
                                                    sep_token=tokenizer.sep_token,
                                                    cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                    pad_on_left=bool(args.model_type in ['xlnet']),
                                                    pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)
        # 把features cache到本地一份
        torch.save(features, cached_features_file)
    # 保存所有的字到total_words
    total_words = []
    for input_example in examples:
        text = input_example.text_a
        total_words.append(text.split(' '))

    #转换成tensor
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    # label_id也转换成tensor
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    #所有要评估的label的id的列表
    all_evaluate_label_ids = [f.evaluate_label_ids for f in features]
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset, all_evaluate_label_ids, total_words


def init_args():
    """
    参数
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--absa_home", type=str, required=True, help="经过训练的ABSA模型的主目录")
    parser.add_argument("--ckpt", type=str, required=True, help="用于评估的模型checkpoint目录")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="传入数据目录。应该包含测试/未见过的数据文件")
    parser.add_argument("--task_name", type=str, required=True, help="task name")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="在列表中选择的模型类型: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="列表中选择的预训练模型或shortcut名称的路径：" + ", ".join(ALL_MODELS))
    parser.add_argument("--cache_dir", default="", type=str,
                        help="您想在哪里存储从s3下载的预训练模型")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="分词后的最大总输入序列长度。序列比这更长将被截断，较短的序列将填充。")
    parser.add_argument('--tagging_schema', type=str, default='BIEOS', help="Tagging schema, 需要和保存的模型ckpt或者bin中的一致")

    args = parser.parse_args()

    return args


def main():
    # perform evaluation on single GPU
    args = init_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    if torch.cuda.is_available():
        args.n_gpu = torch.cuda.device_count()

    args.model_type = args.model_type.lower()
    _, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    #加载训练好的模型 (including the fine-tuned GPT/BERT/XLNET)
    print("开始加载checkpoint %s/%s..." % (args.ckpt, WEIGHTS_NAME))
    model = model_class.from_pretrained(args.ckpt)
    # 遵循加载的模型中的tokenizer的属性, e.g., do_lower_case=True
    tokenizer = tokenizer_class.from_pretrained(args.absa_home)
    if tokenizer is None:
        msg = "请确认tokenizer 文件存在，应该包括 added_tokens.json special_tokens_map.json tokenizer_config.json   vocab.txt"
        raise Exception(msg)
    model.to(args.device)
    #设置模型开始评估
    model.eval()
    predict(args, model, tokenizer)


def predict(args, model, tokenizer):
    """
    预测
    :param args:
    :param model: 已加载好的模型
    :param tokenizer:  已加载好的tokenizer
    :return:
    """
    dataset, evaluate_label_ids, total_words = load_and_cache_examples(args, args.task_name, tokenizer)
    sampler = SequentialSampler(dataset)
    # process the incoming data one by one, batch_size设为1，所以一句话一句话的预测
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1)
    print("***** Running prediction *****")

    total_preds, gold_labels = None, None
    idx = 0
    # absa_label_vocab 是标签对应数字的映射
    if args.tagging_schema == 'BIEOS':
        absa_label_vocab = {'O': 0, 'EQ': 1, 'B-POS': 2, 'I-POS': 3, 'E-POS': 4, 'S-POS': 5,
                        'B-NEG': 6, 'I-NEG': 7, 'E-NEG': 8, 'S-NEG': 9,
                        'B-NEU': 10, 'I-NEU': 11, 'E-NEU': 12, 'S-NEU': 13}
    elif args.tagging_schema == 'BIO':
        absa_label_vocab = {'O': 0, 'EQ': 1, 'B-POS': 2, 'I-POS': 3,
        'B-NEG': 4, 'I-NEG': 5, 'B-NEU': 6, 'I-NEU': 7}
    elif args.tagging_schema == 'OT':
        absa_label_vocab = {'O': 0, 'EQ': 1, 'T-POS': 2, 'T-NEG': 3, 'T-NEU': 4}
    else:
        raise Exception("Invalid tagging schema %s..." % args.tagging_schema)
    # absa_id2tag是id到标签的映射
    absa_id2tag = {}
    for k in absa_label_vocab:
        v = absa_label_vocab[k]
        absa_id2tag[v] = k
    #显示进度
    for batch in tqdm(dataloader, desc="评估"):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            # 其实不用传递labels，预测阶段，这里计算了损失，也没有用到
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                      # XLM don't use segment_ids
                      'labels': batch[3]}
            # 模型的输出[loss, logits,...], 如果不传labels，那么第一个就是logits, 即logits = outputs[0]
            outputs = model(**inputs)
            # logits的形状: (1, seq_len, num_class)
            logits = outputs[1]
            # 找出概率最大的作为预测值: 形状是，(1, seq_len)
            if model.tagger_config.absa_type != 'crf':
                preds = np.argmax(logits.detach().cpu().numpy(), axis=-1)
            else:
                #如果使用的是crf，那么使用viterbi算法推理预测
                mask = batch[1]
                preds = model.tagger.viterbi_tags(logits=logits, mask=mask)
            # 要进行评估的id，例如idx=0，表示对比第一句话的预测结果
            label_indices = evaluate_label_ids[idx]
            # 第一句话的句子的单词 ,例如 ['Love', 'Al', 'Di', 'La']
            words = total_words[idx]
            # 比对前label_indices几个需要对比的字，不用对比padding的结果
            pred_labels = preds[0][label_indices]
            # 判读预测长度和words长度相等
            assert len(words) == len(pred_labels)
            # id 转换成标签, 例如 ['O', 'O', 'O', 'O']
            pred_tags = [absa_id2tag[label] for label in pred_labels]

            if args.tagging_schema == 'OT':
                pred_tags = ot2bieos_ts(pred_tags)
            elif args.tagging_schema == 'BIO':
                pred_tags = ot2bieos_ts(bio2ot_ts(pred_tags))
            else:
                # current tagging schema is BIEOS, do nothing
                pass
            # 把预测到的结果也提取对应的单词和情感
            p_ts_sequence = tag2ts(ts_tag_sequence=pred_tags)
            #输出结果汇总
            output_ts = []
            for t in p_ts_sequence:
                #beg，end是aspect的起始位置，sentiment是情感
                beg, end, sentiment = t
                #这个aspect词语是
                aspect = words[beg:end+1]
                output_ts.append('%s: %s' % (aspect, sentiment))
            print("\n输入的句子是: %s, 预测的结果是: %s" % (' '.join(words), '\t'.join(output_ts)))
            # 下面是用于评估模型结果了，上面的是预测的结果
            if total_preds is None:
                total_preds = preds
            else:
                total_preds = np.append(total_preds, preds, axis=0)
            #提取test.txt中的labels作为gold_labels
            if inputs['labels'] is not None:
                # for the unseen data, there is no ``labels''
                if gold_labels is None:
                    gold_labels = inputs['labels'].detach().cpu().numpy()
                else:
                    gold_labels = np.append(gold_labels, inputs['labels'].detach().cpu().numpy(), axis=0)
        # 下一句话
        idx += 1
    if gold_labels is not None:
        result = compute_metrics_absa(preds=total_preds, labels=gold_labels, all_evaluate_label_ids=evaluate_label_ids,
                                      tagging_schema=args.tagging_schema)
        for (k, v) in result.items():
            print("%s: %s" % (k, v))


if __name__ == "__main__":
    main()

