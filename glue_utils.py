# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open

from seq_utils import *

logger = logging.getLogger(__name__)

SMALL_POSITIVE_CONST = 1e-4


class InputExample(object):
    """A single training/test example for simple sequence classification.
    用于简单序列分类的单个训练/测试样本。
    """

    def __init__(self, guid, text_a, text_b=None, label=None, location=None):
        """构建 a InputExample.

        Args:
            guid: 样本的唯一uid, eg: 'train-4'
            text_a: string. 第一个序列，没有tokenizer的text，对于单序列任务，只需要text_a
            text_b: (Optional) string. 第二个序列，没有tokenizer的text，只有在进行序列对任务时才需要指定
            label: (Optional) string. 样本的标签。这应该是指定用于训练和开发样本，但不用于测试样本。 测试样本不需要标签
            location: (Optional) string. 单词的位置信息，是一个列表，[(start,end),(start,end),...]
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.location = location


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class SeqInputFeatures(object):
    """ABSA任务的一组数据特征"""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, locations, evaluate_label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.locations = locations
        # mapping between word index and head token index
        self.evaluate_label_ids = evaluate_label_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(cell for cell in line)
                lines.append(line)
            return lines


class ABSAProcessor(DataProcessor):
    """Processor for the ABSA datasets"""

    def get_train_examples(self, data_dir, tagging_schema):
        return self._create_examples(data_dir=data_dir, set_type='train', tagging_schema=tagging_schema)

    def get_dev_examples(self, data_dir, tagging_schema):
        return self._create_examples(data_dir=data_dir, set_type='dev', tagging_schema=tagging_schema)

    def get_test_examples(self, data_dir, tagging_schema):
        return self._create_examples(data_dir=data_dir, set_type='test', tagging_schema=tagging_schema)

    def get_labels(self, tagging_schema):
        """
        根据不同的tagging方式，返回不同的形式的所有labels集合
        :param tagging_schema:
        :return:
        """
        if tagging_schema == 'OT':
            return []
        elif tagging_schema == 'BIO':
            return ['O', 'EQ', 'B-POS', 'I-POS', 'B-NEG', 'I-NEG', 'B-NEU', 'I-NEU']
        elif tagging_schema == 'BIEOS':
            return ['O', 'EQ', 'B-POS', 'I-POS', 'E-POS', 'S-POS',
                    'B-NEG', 'I-NEG', 'E-NEG', 'S-NEG',
                    'B-NEU', 'I-NEU', 'E-NEU', 'S-NEU']
        else:
            raise Exception("Invalid tagging schema %s..." % tagging_schema)

    def _create_examples(self, data_dir, set_type, tagging_schema):
        """
        :param data_dir:  例如 './data/rest15'
        :param set_type: 例如是train，或test或dev，组成train.txt
        :param tagging_schema: 例如'BIEOS'
        :return: 这个文件的所有行组成的examples，[InputExample(guid,label,text_a,text_b),...]
        """
        examples = []
        # 原文件路径
        file = os.path.join(data_dir, "%s.txt" % set_type)
        # class_count [0. 0. 0.], 存储统计所有的类别数量，【POS，NEG，NEU】-->【积极的标签总数，消极的标签总数，中性的标签总数】
        class_count = np.zeros(3)
        with open(file, 'r', encoding='UTF-8') as fp:
            # sample_id样本计数，共计多少行
            sample_id = 0
            for line in fp:
                # 用####分割出标签和原始文本
                sent_string, tag_string = line.strip().split('####')
                # 存储每个单词
                words = []
                # 存储每个单词对应的标签
                tags = []
                for tag_item in tag_string.split(' '):
                    eles = tag_item.split('=')
                    if len(eles) == 1:
                        raise Exception("无效样本 %s..." % tag_string)
                    elif len(eles) == 2:
                        word, tag = eles
                    else:
                        # 如果存在多个=号的情况，取最后一个=号后面的最为标签，其它作为单词
                        word = ''.join((len(eles) - 2) * ['='])
                        tag = eles[-1]
                    words.append(word)
                    tags.append(tag)
                # tagging方式从ot转换成BIEOS
                if tagging_schema == 'BIEOS':
                    tags = ot2bieos_ts(tags)
                elif tagging_schema == 'BIO':
                    tags = ot2bio_ts(tags)
                else:
                    # 原始标签遵循OT标签架构，不执行任何操作
                    pass
                # eg: 'train-0'
                guid = "%s-%s" % (set_type, sample_id)
                text_a = ' '.join(words)
                # label = [absa_label_vocab[tag] for tag in tags]
                # gold_ts例如 ['O', 'O', 'S-NEG', 'O'] --> [(2, 2, 'NEG')]，整个词语的起始位置和情感，用于统计class_count
                gold_ts = tag2ts(ts_tag_sequence=tags)
                for (b, e, s) in gold_ts:
                    if s == 'POS':
                        class_count[0] += 1
                    if s == 'NEG':
                        class_count[1] += 1
                    if s == 'NEU':
                        class_count[2] += 1
                # guid= 'train-0', text_a='Avoid this place !', label ['O', 'O', 'S-NEG', 'O']
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=tags))
                sample_id += 1
        print("%s 类别统计数量: %s" % (set_type, class_count))
        return examples


class CosmeticsProcessor(DataProcessor):
    """
    处理自定定义的数据
    """

    def get_train_examples(self, data_dir, tagging_schema):
        return self._create_examples(data_dir=data_dir, set_type='train', tagging_schema=tagging_schema)

    def get_dev_examples(self, data_dir, tagging_schema):
        return self._create_examples(data_dir=data_dir, set_type='dev', tagging_schema=tagging_schema)

    def get_test_examples(self, data_dir, tagging_schema):
        return self._create_examples(data_dir=data_dir, set_type='test', tagging_schema=tagging_schema)

    def get_labels(self, tagging_schema):
        """
        根据不同的tagging方式，返回不同的形式的所有labels集合
        :param tagging_schema:
        :return:
        """
        if tagging_schema == 'OT':
            return []
        elif tagging_schema == 'BIO':
            return ['O', 'EQ', 'B-POS', 'I-POS', 'B-NEG', 'I-NEG', 'B-NEU', 'I-NEU']
        elif tagging_schema == 'BIEOS':
            return ['O', 'EQ', 'B-POS', 'I-POS', 'E-POS', 'S-POS',
                    'B-NEG', 'I-NEG', 'E-NEG', 'S-NEG',
                    'B-NEU', 'I-NEU', 'E-NEU', 'S-NEU']
        elif tagging_schema == 'SENTIMENT':
            return ['NEG', 'NEU', 'POS']
        else:
            raise Exception("Invalid tagging schema %s..." % tagging_schema)

    def _create_examples(self, data_dir, set_type, tagging_schema):
        """
        :param data_dir:  例如 './data/rest15'
        :param set_type: 例如是train，或test或dev，组成train.txt
        :param tagging_schema:  这里使用自定义的schema，'SENTIMENT'
        :return: 这个文件的所有行组成的examples，[InputExample(guid,label,text_a,text_b),...]
        """
        examples = []
        # 原文件路径
        file = os.path.join(data_dir, "%s.txt" % set_type)
        # class_count [0. 0. 0.], 存储统计所有的类别数量，【POS，NEG，NEU】-->【积极的标签总数，消极的标签总数，中性的标签总数】
        class_count = np.zeros(3)
        with open(file, 'r', encoding='UTF-8') as fp:
            # sample_id样本计数，共计多少行
            sample_id = 0
            for line in fp:
                # 用####分割出标签和原始文本
                sent_string, tag_string = line.strip().split('####')
                # 存储每个单词
                words = []
                # 存储每个单词对应的标签
                tags = []
                # 存储每个单词的位置信息，在句子中的起始和结束位置， (start,end)
                location = []
                for tag_item in tag_string.split(' '):
                    eles = tag_item.split('=')
                    if len(eles) == 1:
                        # 对于是空格的地方，过滤掉
                        continue
                    elif len(eles) == 2:
                        word, tag = eles
                    else:
                        # 如果存在多个=号的情况，取最后一个=号后面的最为标签，其它作为单词
                        word = ''.join((len(eles) - 2) * ['='])
                        tag = eles[-1]
                    words.append(word)
                    tag = tag.replace("(", "").replace(")", "")
                    tag_split = tag.split(',')
                    if len(tag_split) != 3:
                        raise Exception(f"注意，这条数据提取标签时格式有问题，请检查{line}")
                    tags.append(tag_split[2])
                    start, end = int(tag_split[0]), int(tag_split[1])
                    location.append((start, end))
                # eg: 'train-0'
                guid = "%s-%s" % (set_type, sample_id)
                text_a = sent_string
                for s in tags:
                    if s == 'POS':
                        class_count[0] += 1
                    if s == 'NEG':
                        class_count[1] += 1
                    if s == 'NEU':
                        class_count[2] += 1
                # guid= 'train-0', text_a='Avoid this place !', label ['O', 'O', 'S-NEG', 'O']
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=tags, location=location))
                sample_id += 1
        print("%s 类别统计数量[POS, NEG, NEU]: %s" % (set_type, class_count))
        return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_seq_features(examples, label_list, tokenizer,
                                     cls_token_at_end=False, pad_on_left=False, cls_token='[CLS]',
                                     sep_token='[SEP]', pad_token=0, sequence_a_segment_id=0,
                                     sequence_b_segment_id=1, cls_token_segment_id=1, pad_token_segment_id=0,
                                     mask_padding_with_zero=True):
    """
    :param examples:
    :param label_list:
    :param tokenizer:
    :param cls_token_at_end:是否把CLS Token放到最后，默认是放在开头，这种形式：CLS，x，x，x，SEP
    :param pad_on_left: 在序列的左边进行padding还是右边
    :param cls_token:  例如CLS
    :param sep_token:  例如SEP
    :param pad_token:
    :param sequence_a_segment_id: 设置为0，第一个序列的segment_id,默认是用作预测下一句用的，NSP，这里没有用到
    :param sequence_b_segment_id: 设置为1
    :param cls_token_segment_id: CLS token的代表数字
    :param pad_token_segment_id: PAD token的代表数字
    :param mask_padding_with_zero: bool， 使用0作为mask，否则是1
    :return:
    """
    # feature extraction for sequence labeling
    # label_map label到数字映射,例如{'O': 0, 'EQ': 1, 'B-POS': 2, 'I-POS': 3, 'E-POS': 4, 'S-POS': 5, 'B-NEG': 6, 'I-NEG': 7, 'E-NEG': 8, 'S-NEG': 9, 'B-NEU': 10, 'I-NEU': 11, 'E-NEU': 12, 'S-NEU': 13}
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    # max_seq_length统计下最长的序列长度
    max_seq_length = -1
    examples_tokenized = []
    for (ex_index, example) in enumerate(examples):
        # 使用空格拆分句子到单词, 计算要进行评估的单词的位置信息evaluate_label_ids
        words = example.text_a.split(' ')
        tokens_a = words
        labels_a = example.label
        evaluate_label_ids = np.array(example.location, dtype=np.int32)
        locations_a = np.array(example.location, dtype=np.int32)
        examples_tokenized.append((tokens_a, labels_a, evaluate_label_ids, locations_a))
        if len(tokens_a) > max_seq_length:
            max_seq_length = len(tokens_a)
    # 最长的序列+2，因为count on the [CLS] and [SEP]
    max_seq_length += 2
    for ex_index, (tokens_a, labels_a, evaluate_label_ids, locations_a) in enumerate(examples_tokenized):
        # 末尾添加SEP的token
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        if cls_token_at_end:
            # 评估标签ID不变，以为在末尾加CLS_TOKEN
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids
            # 所有数+1，评估时id，向右移动一位
            evaluate_label_ids += 1
            locations_a += 1
        # 把字转换成id，例如[101, 2057, 1010, 2045, 2020, 2176, 1997, 2149, 1010, 3369, 2012, 11501, 1010, 1996, 2173, 2001, 4064, 1010, 1998, 1996, 3095, 6051, 2066, 2057, 2020, 16625, 2006, 2068, 1998, 2027, 2020, 2200, 12726, 1012, 102]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # 输入的mask，例如[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        # 修正mask，只关注要分类的词的字的位置，这个位置为1， 其它位置为0
        if mask_padding_with_zero:
            input_mask = [0] * len(input_ids)
            input_mask[locations_a[0][0]:locations_a[0][1]] = [1] * (locations_a[0][1] - locations_a[0][0])
        # padding到最大序列长度Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        # print("Current labels:", labels), labels标签字符转换成id， 例如[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        single_label = [label_map[label] for label in labels_a]
        # 随便取一个临时数值0作为初始，最后计算时也不会用到这个，只是占位
        label_ids = [0] * len(input_ids)
        # 修改固定位置为这个label
        label_ids[locations_a[0][0]:locations_a[0][1]] = single_label * (locations_a[0][1] - locations_a[0][0])
        # 填充输入序列和mask序列, 从左边开始padding还是右边
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([0] * padding_length) + label_ids
            # right shift padding_length for evaluate_label_ids
            evaluate_label_ids += padding_length
        else:
            # 在序列的右边开始，例如最大序列长度是83： [101, 2057, 1010, 2045, 2020, 2176, 1997, 2149, 1010, 3369, 2012, 11501, 1010, 1996, 2173, 2001, 4064, 1010, 1998, 1996, 3095, 6051, 2066, 2057, 2020, 16625, 2006, 2068, 1998, 2027, 2020, 2200, 12726, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            # evaluate_label_ids要评估的id是不变的，我们不评估padding的那些id， padding evaluate ids not change
            input_ids = input_ids + ([pad_token] * padding_length)
            # 增加mask的长度，padding的位置的mask数字要和input_mask的不一样，最终结果，例如[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            label_ids = label_ids + ([0] * padding_length)
            # padding部分的segment id扩充
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
        # 验证这些长度都达到了序列最大长度
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        # 打印前5个样本
        if ex_index < 5:
            logger.info("*** 打印前5个样本示例 ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("labels: %s " % ' '.join([str(x) for x in label_ids]))
            logger.info("evaluate label ids: %s" % evaluate_label_ids)
        # 把转换好的feature加入到features列表
        features.append(
            SeqInputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             label_ids=label_ids,
                             locations=locations_a,
                             evaluate_label_ids=evaluate_label_ids))
    print("最大序列长度是: ", max_seq_length)
    return features


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def match_ts(gold_ts_sequence, pred_ts_sequence):
    """
    计算正确预测的目标情感数量
    :param gold_ts_sequence: gold standard targeted sentiment sequence, eg [(1, 3, 'POS')]
    :param pred_ts_sequence: predicted targeted sentiment sequence, eg: []
    :return:
    """
    # positive, negative and neutral
    tag2tagid = {'POS': 0, 'NEG': 1, 'NEU': 2}
    hit_count, gold_count, pred_count = np.zeros(3), np.zeros(3), np.zeros(3)
    for t in gold_ts_sequence:
        # print(t)
        ts_tag = t[2]
        tid = tag2tagid[ts_tag]
        gold_count[tid] += 1
    for t in pred_ts_sequence:
        ts_tag = t[2]
        tid = tag2tagid[ts_tag]
        # 如果（begin，end，sentiment）完全匹配，算是hit一个
        if t in gold_ts_sequence:
            hit_count[tid] += 1
        pred_count[tid] += 1
    return hit_count, gold_count, pred_count


def compute_metrics_absa(preds, labels, all_evaluate_label_ids, tagging_schema):
    """
    计算abas模型的metric
    :param preds:  预测值 [all_senetences, sequence_length]， 例如[685,77] 代表685个句子，每个句子77长度
    :param labels:  真实值 [all_senetences, sequence_length]
    :param all_evaluate_label_ids: 列表，长度等于all_senetences，每个item是每个句子中真实的单词长度的列表，代表需要评估的单词
    :param tagging_schema: 序列标注的schema，默认是BIEOS
    :return:
    """
    if tagging_schema == 'BIEOS':
        absa_label_vocab = {'O': 0, 'EQ': 1, 'B-POS': 2, 'I-POS': 3, 'E-POS': 4, 'S-POS': 5,
                            'B-NEG': 6, 'I-NEG': 7, 'E-NEG': 8, 'S-NEG': 9,
                            'B-NEU': 10, 'I-NEU': 11, 'E-NEU': 12, 'S-NEU': 13}
    elif tagging_schema == 'BIO':
        absa_label_vocab = {'O': 0, 'EQ': 1, 'B-POS': 2, 'I-POS': 3,
                            'B-NEG': 4, 'I-NEG': 5, 'B-NEU': 6, 'I-NEU': 7}
    elif tagging_schema == 'OT':
        absa_label_vocab = {'O': 0, 'EQ': 1, 'T-POS': 2, 'T-NEG': 3, 'T-NEU': 4}
    elif tagging_schema == 'SENTIMENT':
        absa_label_vocab = {"NEG": 0, "NEU": 1, "POS": 2, }
    else:
        raise Exception("评估时提供的tagging schema无效 %s..." % tagging_schema)
    absa_id2tag = {}
    for k in absa_label_vocab:
        v = absa_label_vocab[k]
        absa_id2tag[v] = k
    # 初始化
    correct_labels = 0
    pred_tags = []
    gold_tags = []
    n_samples = len(all_evaluate_label_ids)
    # 对每个样本进行循环
    for i in range(n_samples):
        # 第i个样本的真实的单词数  eg: [1 2 3 4]
        evaluate_label_ids = all_evaluate_label_ids[i]
        # 找出单词的预测的标签,  eg: [0 0 0 0]
        pred_labels = preds[i]
        # 真实的单词的标签, 例如 [0 2 3 4]
        gold_labels = labels[i][evaluate_label_ids[0][0]]
        # 把id转换成tag
        pred_tag = absa_id2tag[pred_labels]
        gold_tag = absa_id2tag[gold_labels]
        pred_tags.append(pred_tag)
        gold_tags.append(gold_tag)
        if pred_labels == gold_labels:
            correct_labels += 1
    accuracy = float(correct_labels)/float(n_samples)
    scores = {'accuracy': accuracy, 'predict_tag': pred_tags, 'ground_truth':gold_tags}
    return scores


processors = {
    "laptop14": ABSAProcessor,
    "rest_total": ABSAProcessor,
    "rest_total_revised": ABSAProcessor,
    "rest14": ABSAProcessor,
    "rest15": ABSAProcessor,
    "rest16": ABSAProcessor,
    "cosmetics": CosmeticsProcessor,
}

output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    "laptop14": "classification",
    "rest_total": "classification",
    "rest14": "classification",
    "rest15": "classification",
    "rest16": "classification",
    "rest_total_revised": "classification",
    "cosmetics": "classification",
}
