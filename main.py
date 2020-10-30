import argparse
import os, time
import torch
import logging
import random
import numpy as np

from glue_utils import convert_examples_to_seq_features, output_modes, processors, compute_metrics_absa
from tqdm import tqdm, trange
from transformers import BertConfig, BertTokenizer, XLNetConfig, XLNetTokenizer, WEIGHTS_NAME
from transformers import AdamW, WarmupLinearSchedule
from absa_layer import BertABSATagger, XLNetABSATagger

from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from tensorboardX import SummaryWriter

import glob
import json

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertABSATagger, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetABSATagger, XLNetTokenizer)
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="数据目录. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="模型的类型: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--absa_type", default=None, type=str, required=True,
                        help="在列表中选择的下游Absa图层类型: [linear, gru, san, tfm, crf]")
    parser.add_argument("--tfm_mode", default=None, type=str, required=True,
                        help="预训练transformer的模式,使用bert或xlnet作为预训练模型，进行微调训练: [finetune]")
    parser.add_argument("--fix_tfm", default=None, type=int, required=True,
                        help="是否固定transformer的模型的参数，例如固定bert的参数")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="列表中选择的预训练模型路径或shortcut名称： " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="列表中选择的要训练任务的名称：" + ", ".join(processors.keys()))

    ## 其他参数
    parser.add_argument("--config_name", default="", type=str,
                        help="预训练的配置名称或路径（如果与model_name不同）")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="预训练的toknizer名称或路径（如果与model_name不同")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="您想在哪里存储从s3下载的预训练模型")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="分词后的最大总输入序列长度。序列比这更长将被截断，较短的序列将填充。")
    parser.add_argument("--do_train", action='store_true',
                        help="是否进行训练。")
    parser.add_argument("--do_eval", action='store_true',
                        help="是否在开发集上运行评估。")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="训练期间在每个测井步骤运行评估。")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="如果使用的是无大小写的模型，请设置此flag。")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="训练时每个GPU / CPU的批次大小。")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="每个GPU / CPU的批次大小以进行评估。")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="在执行向后/更新过程之前要累积的更新步骤数。")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="Adam的初始学习率。")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="要执行的训练epoch总数。")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="如果> 0：设置要执行的训练步骤总数。覆盖num_train_epochs")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=100,
                        help="每X个更新步骤保存一个checkpoint")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="评估所有检查点，这些检查点以与model_name相同的前缀开头，并以步骤号结尾")
    parser.add_argument("--no_cuda", action='store_true',
                        help="避免在可用时使用CUDA")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="覆盖输出目录的内容")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="覆盖缓存的训练和评估集")
    parser.add_argument('--seed', type=int, default=42,
                        help="随机种子进行初始化")
    parser.add_argument('--tagging_schema', type=str, default='BIEOS',help="序列标签结构")

    parser.add_argument("--overfit", type=int, default=0, help="是否评估过拟合")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--MASTER_ADDR', type=str)
    parser.add_argument('--MASTER_PORT', type=str)
    args = parser.parse_args()
    output_dir = '%s-%s-%s-%s' % (args.model_type, args.absa_type, args.task_name, args.tfm_mode)

    if args.fix_tfm:
        output_dir = '%s-fix' % output_dir
    if args.overfit:
        output_dir = '%s-overfit' % output_dir
        args.max_steps = 3000
    args.output_dir = output_dir
    return args


def train(args, train_dataset, model, tokenizer):
    """ 训练模型 """
    #保存SummaryWriter
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    # 根据gpu数量计算batch_size
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # 从shuffled后的数据集中提取训练样本
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    #sampler, 定义从数据集中抽取样本的策略
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    # 根据提供的max_steps或epochs计算训练的总的steps数量
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # 准备 optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    #修改需要进行weight_decay的参数
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    #开始训练
    logger.info("***** 开始 training *****")
    logger.info("  总的样本数 = %d", len(train_dataset))
    logger.info("  Epochs数 = %d", args.num_train_epochs)
    logger.info("  每个GPU的Batch size = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    #tr_loss是训练的总损失
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    # trange 进度条
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    #保存好的模型的文件夹列表
    model_dirs = []
    # 为了复现，设置随机数种子
    set_seed(args)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            # 如果是gpu，把tensor放到gpu
            batch = tuple(t.to(args.device) for t in batch)
            # 取出一条数据的features
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                      'labels':         batch[3]}
            ouputs = model(**inputs)
            # 第一个返回值是训练的损失
            loss = ouputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            #反向传播loss
            loss.backward()
            #梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            #损失累加, loss.item()从tensor变成数字
            tr_loss += loss.item()
            # 记录损失到日志文件, 没50步记录一次
            if tr_loss != 0 and global_step !=0 and global_step % 50 == 0:
                logger.info(f"第{global_step}个step的损失是: {tr_loss/global_step}")
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                #更新学习率
                scheduler.step()  # Update learning rate schedule
                #清空过往梯度
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # 只保留最近的3个模型
                    if len(model_dirs) > 2:
                        # 删除最旧的模型
                        import shutil
                        shutil.rmtree(model_dirs[0])
                        model_dirs.pop(0)
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    #把最新的模型加到列表
                    model_dirs.append(output_dir)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("保存checkpoint到 %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, mode, prefix=""):
    """
    评估模型
    :param args:
    :param model: 加载好的模型
    :param tokenizer: 加载好的tokenizer
    :param mode: dev还是test
    :return: 返回类似格式
    eval_loss = 0.14909637707974263
    macro-f1 = 0.580691820122606
    micro-f1 = 0.6091904960800573
    precision = 0.6401535578632168
    recall = 0.5811752698617894
    """
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)
    # 存储评估结果
    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset, eval_evaluate_label_ids = load_and_cache_examples(args, eval_task, tokenizer, mode=mode)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)
        #根据gpu，计算batch_size
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # 请注意，使用DistributedSampler随机采样
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        #logger.info("***** Running evaluation on %s.txt *****" % mode)
        eval_loss = 0.0
        #记录评估的步数
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        crf_logits, crf_mask = [], []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            #去取出一个batch的数据，放入inputs
            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                          'labels':         batch[3]}
                outputs = model(**inputs)
                # logits的形状 (batch_size, seq_len, label_size)
                # 这里的损失是masked的损失
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
                # 收集logits
                crf_logits.append(logits)
                crf_mask.append(batch[1])
            #评估完一个step，步数加1
            nb_eval_steps += 1
            print(f"\n评估完第{nb_eval_steps}个step")
            #第一次时preds为None，否则，把logits都都收集起来
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
        #计算平均损失
        eval_loss = eval_loss / nb_eval_steps
        # argmax在最后一个维度上的操作
        if model.tagger_config.absa_type != 'crf':
            # greedy decoding
            preds = np.argmax(preds, axis=-1)
        else:
            # viterbi 算法 for CRF-based model
            crf_logits = torch.cat(crf_logits, dim=0)
            crf_mask = torch.cat(crf_mask, dim=0)
            preds = model.tagger.viterbi_tags(logits=crf_logits, mask=crf_mask)
        result = compute_metrics_absa(preds, out_label_ids, eval_evaluate_label_ids, args.tagging_schema)
        result['eval_loss'] = eval_loss
        results.update(result)
        #写入到文件，保存评估结果
        output_eval_file = os.path.join(eval_output_dir, "%s_results.txt" % mode)
        with open(output_eval_file, "w") as writer:
            logger.info("***** %s results *****" % mode)
            for key in sorted(result.keys()):
                if 'eval_loss' in key:
                    logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def load_and_cache_examples(args, task, tokenizer, mode='train'):
    """
    加载文本到features，并缓存文本
    :param args:
    :param task: 任务的名字
    :param tokenizer: 使用的tokenizer
    :param mode: 是train还是dev还是test
    :return: 返回TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids) 和all_evaluate_label_ids
    """
    processor = processors[task]()
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        mode,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        print("读取已缓存的features file:", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("开始创建features file，并缓存%s", args.data_dir)
        #读取这个标签类型的所有labels，例如，['O', 'EQ', 'B-POS', 'I-POS', 'E-POS', 'S-POS', 'B-NEG', 'I-NEG', 'E-NEG', 'S-NEG', 'B-NEU', 'I-NEU', 'E-NEU', 'S-NEU']
        label_list = processor.get_labels(args.tagging_schema)
        if mode == 'train':
            examples = processor.get_train_examples(args.data_dir, args.tagging_schema)
        elif mode == 'dev':
            examples = processor.get_dev_examples(args.data_dir, args.tagging_schema)
        elif mode == 'test':
            examples = processor.get_test_examples(args.data_dir, args.tagging_schema)
        else:
            raise Exception("Invalid data mode %s..." % mode)
        # 开始使用examples制作features
        features = convert_examples_to_seq_features(examples=examples, label_list=label_list, tokenizer=tokenizer,
                                                    cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                    cls_token=tokenizer.cls_token,
                                                    sep_token=tokenizer.sep_token,
                                                    cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                    pad_on_left=bool(args.model_type in ['xlnet']),
                                                    pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)
        if args.local_rank in [-1, 0]:
            #logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # 转换成tensor，建立dataset,包括input_id input_mask, segment_id, label_id,evaluate_label_id,
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    # all_evaluate_label_ids是要评估的那些位置的id
    all_evaluate_label_ids = [f.evaluate_label_ids for f in features]
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset, all_evaluate_label_ids


def main():
    """
    主函数
    :return:
    """
    args = init_args()
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # 设置CUDA，GPU和分布式训练, local_rank == -1 表示不使用分布式训练
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        os.environ['MASTER_ADDR'] = args.MASTER_ADDR
        os.environ['MASTER_PORT'] = args.MASTER_PORT
        torch.distributed.init_process_group(backend='nccl', rank=args.local_rank, world_size=1)
        args.n_gpu = 1

    args.device = device

    # 日志格式
    logdir = "log"
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    logfile = os.path.join(logdir, time.strftime("%Y%m%d%H%M",time.localtime()))
    logging.basicConfig(filename=logfile,format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    # not using 16-bits training
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: False",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))

    # Set seed
    set_seed(args)

    #准备数据处理部分
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("在我们自定义的任务列表中没有发现任务: %s" % args.task_name)
    processor = processors[args.task_name]()
    # output_mode 是 classification，表示分类
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels(args.tagging_schema)
    num_labels = len(label_list)
    #分布式训练设置
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    #初始化预训练模型,args.model_type是bert或xlnet等, 注意cache_dir设置
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case, cache_dir='./model_cache')
    #absa_type 是最后用linear还是crf，还是rnn或self-attention
    config.absa_type = args.absa_type
    config.tfm_mode = args.tfm_mode
    # 是否固定bert参数
    config.fix_tfm = args.fix_tfm
    #加载，使用自定义的预训练模型，并缓存到model_cache， 例如BertABSATagger
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),config=config, cache_dir='./model_cache')
    #更改模型的device
    model.to(args.device)
    #分布式并行训练，如果启用, 如果是多GPU，使用并行方式
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    #加载和缓存数据集,训练
    if args.do_train:
        train_dataset, train_evaluate_label_ids = load_and_cache_examples(args, args.task_name, tokenizer, mode='train')
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(f"训练完成steps {global_step}, 损失为 {tr_loss}")

    if args.do_train and (args.local_rank == -1 or dist.get_rank() == 0):
        #创建输出文件夹，保存模型
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.mkdir(args.output_dir)
        #保存模型和tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: 将训练参数也同时保存
        # 保存训练参数
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # 加载经过微调的训练过的模型和tokenizer
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)
        logger.info("训练完成")

    if args.do_eval:
        #验证阶段
        results = {}
        best_f1 = -999999.0
        best_checkpoint = None
        # checkpoints eg: ['bert-gru-cosmetics-finetune/checkpoint-1300', 'bert-gru-cosmetics-finetune/checkpoint-1500']
        checkpoints = [args.output_dir]
        #是否要评估所有保存的checkpoints
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("在以下checkpoints上执行验证: %s", checkpoints)
        test_results = {}
        for checkpoint in checkpoints:
            # 提取文件夹后缀的step checkpoint-1500, global_step=1500
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            if global_step == 'finetune' or global_step == 'train' or global_step == 'fix' or global_step == 'overfit':
                continue
            #验证集评估
            logger.info(f"开始在开发集上进行评估{checkpoint}")
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            dev_result = evaluate(args, model, tokenizer, mode='dev', prefix=global_step)

            # 使用micro-f1 作为模型选择的标准
            if int(global_step) > 10 and dev_result['micro-f1'] > best_f1:
                best_f1 = dev_result['micro-f1']
                best_checkpoint = checkpoint
            dev_result = dict((k + '_{}'.format(global_step), v) for k, v in dev_result.items())
            results.update(dev_result)
            #测试集上测试
            logger.info("开始在测试集上进行评估")
            test_result = evaluate(args, model, tokenizer, mode='test', prefix=global_step)
            test_result = dict((k + '_{}'.format(global_step), v) for k, v in test_result.items())
            test_results.update(test_result)
        # 例如： bert-linear-rest15-finetune/checkpoint-1500
        best_ckpt_string = "\nThe best checkpoint is %s" % best_checkpoint
        logger.info(best_ckpt_string)
        dev_f1_values, dev_loss_values = [], []
        for k in results:
            v = results[k]
            if 'micro-f1' in k:
                dev_f1_values.append((k, v))
            if 'eval_loss' in k:
                dev_loss_values.append((k, v))
        test_f1_values, test_loss_values = [], []
        for k in test_results:
            v = test_results[k]
            if 'micro-f1' in k:
                test_f1_values.append((k, v))
            if 'eval_loss' in k:
                test_loss_values.append((k, v))
        # 把每步的dev和test集的验证结果写到日志中
        log_file_path = '%s/log.txt' % args.output_dir
        log_file = open(log_file_path, 'a')
        log_file.write("\tValidation:\n")
        for (test_f1_k, test_f1_v), (test_loss_k, test_loss_v), (dev_f1_k, dev_f1_v), (dev_loss_k, dev_loss_v) in zip(
                test_f1_values, test_loss_values, dev_f1_values, dev_loss_values):
            global_step = int(test_f1_k.split('_')[-1])
            if not args.overfit and global_step <= 1000:
                continue
            print('test-%s: %.5lf, test-%s: %.5lf, dev-%s: %.5lf, dev-%s: %.5lf' % (test_f1_k,
                                                                                    test_f1_v, test_loss_k, test_loss_v,
                                                                                    dev_f1_k, dev_f1_v, dev_loss_k,
                                                                                    dev_loss_v))
            validation_string = '\t\tdev-%s: %.5lf, dev-%s: %.5lf' % (dev_f1_k, dev_f1_v, dev_loss_k, dev_loss_v)
            log_file.write(validation_string+'\n')

        n_times = args.max_steps // args.save_steps + 1
        for i in range(1, n_times):
            step = i * 100
            log_file.write('\tStep %s:\n' % step)
            precision = test_results['precision_%s' % step]
            recall = test_results['recall_%s' % step]
            micro_f1 = test_results['micro-f1_%s' % step]
            macro_f1 = test_results['macro-f1_%s' % step]
            log_file.write('\t\tprecision: %.4lf, recall: %.4lf, micro-f1: %.4lf, macro-f1: %.4lf\n'
                           % (precision, recall, micro_f1, macro_f1))
        log_file.write("\tBest checkpoint: %s\n" % best_checkpoint)
        log_file.write('******************************************\n')
        log_file.close()


if __name__ == '__main__':
    main()




