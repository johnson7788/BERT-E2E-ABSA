import torch
import torch.nn as nn
from transformers import BertModel, XLNetModel
from seq_utils import *
from bert import BertPreTrainedModel, XLNetPreTrainedModel
from torch.nn import CrossEntropyLoss


class TaggerConfig:
    """
    模型的配置
    """
    def __init__(self):
        self.hidden_dropout_prob = 0.1
        self.hidden_size = 768
        self.n_rnn_layers = 1  # not used if tagger is non-RNN model
        self.bidirectional = True  # not used if tagger is non-RNN model


class SAN(nn.Module):
    """
    原始的self-attention结构
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super(SAN, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """

        :param src:
        :param src_mask:
        :param src_key_padding_mask:
        :return:
        """
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout(src2)
        # apply layer normalization
        src = self.norm(src)
        return src


class GRU(nn.Module):
    # customized GRU with layer normalization
    def __init__(self, input_size, hidden_size, bidirectional=True):
        """
        使用GRU模型
        :param input_size: 输入层尺寸
        :param hidden_size: 隐藏层尺寸
        :param bidirectional: 是否是双向gru
        """
        super(GRU, self).__init__()
        self.input_size = input_size
        if bidirectional:
            self.hidden_size = hidden_size // 2
        else:
            self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.Wxrz = nn.Linear(in_features=self.input_size, out_features=2*self.hidden_size, bias=True)
        self.Whrz = nn.Linear(in_features=self.hidden_size, out_features=2*self.hidden_size, bias=True)
        self.Wxn = nn.Linear(in_features=self.input_size, out_features=self.hidden_size, bias=True)
        self.Whn = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=True)
        self.LNx1 = nn.LayerNorm(2*self.hidden_size)
        self.LNh1 = nn.LayerNorm(2*self.hidden_size)
        self.LNx2 = nn.LayerNorm(self.hidden_size)
        self.LNh2 = nn.LayerNorm(self.hidden_size)

    def forward(self, x):
        """
        GRU 模型
        :param x: input tensor, shape: (batch_size, seq_len, input_size) ,例如 torch.Size([16, 83, 768])
        :return: gru模型的输出
        """
        def recurrence(xt, htm1):
            """
            :param xt: current input, [batch_size, demision_size]， 每个时间步的输入
            :param htm1: 上一时刻的隐藏状态
            :return: 当前时刻的输出的隐藏状态
            """
            # gates_rz [batch_size,demision_size]
            gates_rz = torch.sigmoid(self.LNx1(self.Wxrz(xt)) + self.LNh1(self.Whrz(htm1)))
            #因为做的一次计算，现拆出来rt重置门和zt更新门, rt [batch_size, demision_size/2], zt [batch_size, demision_size/2]
            rt, zt = gates_rz.chunk(2, 1)  #维度1上拆出2份
            # nt 是ht_hat [batch_size, demision_size/2]
            nt = torch.tanh(self.LNx2(self.Wxn(xt))+rt*self.LNh2(self.Whn(htm1)))
            # 最终的ht [bath_size, demision_size/2]
            ht = (1.0-zt) * nt + zt * htm1
            return ht
        # 时间步，x.size(1)是序列长度
        steps = range(x.size(1))
        bs = x.size(0)
        #第一个时间步，用0初始化，初始化h0为0, [batch_size,hidden_size]
        hidden = self.init_hidden(bs)
        # 调换0和1的维度， 变成(seq_len, batch_size, input_size)
        input = x.transpose(0, 1)
        output = []
        #对每个时间步进行循环,t=0,1,2,3,....., input[t]是第t个时间步的输入，hidden是上一个的输出，作为下一个gru的输入
        for t in steps:
            # input[t]的维度是 [batch_size, input_size]，作为每个step的输入
            hidden = recurrence(input[t], hidden)
            output.append(hidden)
        # 把每个时间步输出的列表output，拼接回(seq_len, batch_size,demision_size)，然后转换成(batch_size, seq_len, demision_size)
        output = torch.stack(output, 0).transpose(0, 1)
        #如果是双向rnn, 反向进行一遍GRU模型
        if self.bidirectional:
            output_b = []
            #初始化h0
            hidden_b = self.init_hidden(bs)
            #反向每个时间步
            for t in steps[::-1]:
                hidden_b = recurrence(input[t], hidden_b)
                output_b.append(hidden_b)
            #输出结果还是要正回来，按照正常序列
            output_b = output_b[::-1]
            # output_b (batch_size, seq_len, demision_size)
            output_b = torch.stack(output_b, 0).transpose(0, 1)
            # 把正向和反向的拼接在一起 (batch_size, seq_len, demision_size) --> (batch_size, seq_len, demision_size*2), demision_size*2就是input_size
            output = torch.cat([output, output_b], dim=-1)
        return output, None

    def init_hidden(self, bs):
        """
        初始化h0，这里如果是双向RNN，那么self.hidden_size是input的一半
        :param bs:
        :return:
        """
        if torch.cuda.is_available():
            h_0 = torch.zeros(bs, self.hidden_size).cuda()
        else:
            h_0 = torch.zeros(bs, self.hidden_size)
        return h_0


class CRF(nn.Module):
    # borrow the code from 
    # https://github.com/allenai/allennlp/blob/master/allennlp/modules/conditional_random_field.py
    def __init__(self, num_tags, constraints=None, include_start_end_transitions=None):
        """

        :param num_tags:
        :param constraints:
        :param include_start_end_transitions:
        """
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.include_start_end_transitions = include_start_end_transitions
        self.transitions = nn.Parameter(torch.Tensor(self.num_tags, self.num_tags))
        constraint_mask = torch.Tensor(self.num_tags+2, self.num_tags+2).fill_(1.)
        if include_start_end_transitions:
            self.start_transitions = nn.Parameter(torch.Tensor(num_tags))
            self.end_transitions = nn.Parameter(torch.Tensor(num_tags))
        # register the constraint_mask
        self.constraint_mask = nn.Parameter(constraint_mask, requires_grad=False)
        self.reset_parameters()

    def forward(self, inputs, tags, mask=None):
        """

        :param inputs: (bsz, seq_len, num_tags), logits calculated from a linear layer
        :param tags: (bsz, seq_len)
        :param mask: (bsz, seq_len), mask for the padding token
        :return:
        """
        if mask is None:
            mask = torch.ones(*tags.size(), dtype=torch.long)
        log_denominator = self._input_likelihood(inputs, mask)
        log_numerator = self._joint_likelihood(inputs, tags, mask)
        return torch.sum(log_numerator - log_denominator)

    def reset_parameters(self):
        """
        initialize the parameters in CRF
        :return:
        """
        nn.init.xavier_normal_(self.transitions)
        if self.include_start_end_transitions:
            nn.init.normal_(self.start_transitions)
            nn.init.normal_(self.end_transitions)

    def _input_likelihood(self, logits, mask):
        """

        :param logits: emission score calculated by a linear layer, shape: (batch_size, seq_len, num_tags)
        :param mask:
        :return:
        """
        bsz, seq_len, num_tags = logits.size()
        # Transpose batch size and sequence dimensions
        mask = mask.float().transpose(0, 1).contiguous()
        logits = logits.transpose(0, 1).contiguous()

        # Initial alpha is the (batch_size, num_tags) tensor of likelihoods combining the
        # transitions to the initial states and the logits for the first timestep.
        if self.include_start_end_transitions:
            alpha = self.start_transitions.view(1, num_tags) + logits[0]
        else:
            alpha = logits[0]

        for t in range(1, seq_len):
            # iteration starts from 1
            emit_scores = logits[t].view(bsz, 1, num_tags)
            transition_scores = self.transitions.view(1, num_tags, num_tags)
            broadcast_alpha = alpha.view(bsz, num_tags, 1)

            # calculate the likelihood
            inner = broadcast_alpha + emit_scores + transition_scores

            # mask the padded token when met the padded token, retain the previous alpha
            alpha = (logsumexp(inner, 1) * mask[t].view(bsz, 1) + alpha * (1 - mask[t]).view(bsz, 1))
        # Every sequence needs to end with a transition to the stop_tag.
        if self.include_start_end_transitions:
            stops = alpha + self.end_transitions.view(1, num_tags)
        else:
            stops = alpha

        # Finally we log_sum_exp along the num_tags dim, result is (batch_size,)
        return logsumexp(stops)

    def _joint_likelihood(self, logits, tags, mask):
        """
        calculate the likelihood for the input tag sequence
        :param logits:
        :param tags: shape: (bsz, seq_len)
        :param mask: shape: (bsz, seq_len)
        :return:
        """
        bsz, seq_len, _ = logits.size()

        # Transpose batch size and sequence dimensions:
        logits = logits.transpose(0, 1).contiguous()
        mask = mask.float().transpose(0, 1).contiguous()
        tags = tags.transpose(0, 1).contiguous()

        # Start with the transition scores from start_tag to the first tag in each input
        if self.include_start_end_transitions:
            score = self.start_transitions.index_select(0, tags[0])
        else:
            score = 0.0

        for t in range(seq_len-1):
            current_tag, next_tag = tags[t], tags[t+1]
            # The scores for transitioning from current_tag to next_tag
            transition_score = self.transitions[current_tag.view(-1), next_tag.view(-1)]

            # The score for using current_tag
            emit_score = logits[t].gather(1, current_tag.view(bsz, 1)).squeeze(1)

            score = score + transition_score * mask[t+1] + emit_score * mask[t]

        last_tag_index = mask.sum(0).long() - 1
        last_tags = tags.gather(0, last_tag_index.view(1, bsz)).squeeze(0)

        # Compute score of transitioning to `stop_tag` from each "last tag".
        if self.include_start_end_transitions:
            last_transition_score = self.end_transitions.index_select(0, last_tags)
        else:
            last_transition_score = 0.0

        last_inputs = logits[-1]  # (batch_size, num_tags)
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1))  # (batch_size, 1)
        last_input_score = last_input_score.squeeze()  # (batch_size,)

        score = score + last_transition_score + last_input_score * mask[-1]

        return score

    def viterbi_tags(self, logits, mask):
        """

        :param logits: (bsz, seq_len, num_tags), emission scores
        :param mask:
        :return:
        """
        _, max_seq_len, num_tags = logits.size()

        # Get the tensors out of the variables
        logits, mask = logits.data, mask.data

        # Augment transitions matrix with start and end transitions
        start_tag = num_tags
        end_tag = num_tags + 1
        transitions = torch.Tensor(num_tags + 2, num_tags + 2).fill_(-10000.)

        # Apply transition constraints
        constrained_transitions = (
                self.transitions * self.constraint_mask[:num_tags, :num_tags] +
                -10000.0 * (1 - self.constraint_mask[:num_tags, :num_tags])
        )

        transitions[:num_tags, :num_tags] = constrained_transitions.data

        if self.include_start_end_transitions:
            transitions[start_tag, :num_tags] = (
                    self.start_transitions.detach() * self.constraint_mask[start_tag, :num_tags].data +
                    -10000.0 * (1 - self.constraint_mask[start_tag, :num_tags].detach())
            )
            transitions[:num_tags, end_tag] = (
                    self.end_transitions.detach() * self.constraint_mask[:num_tags, end_tag].data +
                    -10000.0 * (1 - self.constraint_mask[:num_tags, end_tag].detach())
            )
        else:
            transitions[start_tag, :num_tags] = (-10000.0 *
                                                 (1 - self.constraint_mask[start_tag, :num_tags].detach()))
            transitions[:num_tags, end_tag] = -10000.0 * (1 - self.constraint_mask[:num_tags, end_tag].detach())

        best_paths = []
        # Pad the max sequence length by 2 to account for start_tag + end_tag.
        tag_sequence = torch.Tensor(max_seq_len + 2, num_tags + 2)

        for prediction, prediction_mask in zip(logits, mask):
            # perform viterbi decoding sample by sample
            seq_len = torch.sum(prediction_mask)
            # Start with everything totally unlikely
            tag_sequence.fill_(-10000.)
            # At timestep 0 we must have the START_TAG
            tag_sequence[0, start_tag] = 0.
            # At steps 1, ..., sequence_length we just use the incoming prediction
            tag_sequence[1:(seq_len + 1), :num_tags] = prediction[:seq_len]
            # And at the last timestep we must have the END_TAG
            tag_sequence[seq_len + 1, end_tag] = 0.
            viterbi_path = viterbi_decode(tag_sequence[:(seq_len + 2)], transitions)
            viterbi_path = viterbi_path[1:-1]
            best_paths.append(viterbi_path)
        return best_paths


class LSTM(nn.Module):
    # customized LSTM with layer normalization
    def __init__(self, input_size, hidden_size, bidirectional=True):
        """

        :param input_size:
        :param hidden_size:
        :param bidirectional:
        """
        super(LSTM, self).__init__()
        self.input_size = input_size
        if bidirectional:
            self.hidden_size = hidden_size // 2
        else:
            self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.LNx = nn.LayerNorm(4*self.hidden_size)
        self.LNh = nn.LayerNorm(4*self.hidden_size)
        self.LNc = nn.LayerNorm(self.hidden_size)
        self.Wx = nn.Linear(in_features=self.input_size, out_features=4*self.hidden_size, bias=True)
        self.Wh = nn.Linear(in_features=self.hidden_size, out_features=4*self.hidden_size, bias=True)

    def forward(self, x):
        """

        :param x: input, shape: (batch_size, seq_len, input_size)
        :return:
        """
        def recurrence(xt, hidden):
            """
            recurrence function enhanced with layer norm
            :param input: input to the current cell
            :param hidden:
            :return:
            """
            htm1, ctm1 = hidden
            gates = self.LNx(self.Wx(xt)) + self.LNh(self.Wh(htm1))
            it, ft, gt, ot = gates.chunk(4, 1)
            it = torch.sigmoid(it)
            ft = torch.sigmoid(ft)
            gt = torch.tanh(gt)
            ot = torch.sigmoid(ot)
            ct = (ft * ctm1) + (it * gt)
            ht = ot * torch.tanh(self.LNc(ct))  # n_b x hidden_dim

            return ht, ct
        output = []
        # sequence_length
        steps = range(x.size(1))
        hidden = self.init_hidden(x.size(0))
        # change to: (seq_len, bs, hidden_size)
        input = x.transpose(0, 1)
        for t in steps:
            hidden = recurrence(input[t], hidden)
            output.append(hidden[0])
        # (bs, seq_len, hidden_size)
        output = torch.stack(output, 0).transpose(0, 1)

        if self.bidirectional:
            hidden_b = self.init_hidden(x.size(0))
            output_b = []
            for t in steps[::-1]:
                hidden_b = recurrence(input[t], hidden_b)
                output_b.append(hidden_b[0])
            output_b = output_b[::-1]
            output_b = torch.stack(output_b, 0).transpose(0, 1)
            output = torch.cat([output, output_b], dim=-1)
        return output, None

    def init_hidden(self, bs):
        h_0 = torch.zeros(bs, self.hidden_size).cuda()
        c_0 = torch.zeros(bs, self.hidden_size).cuda()
        return h_0, c_0


class BertABSATagger(BertPreTrainedModel):
    def __init__(self, bert_config):
        """
        自定义预训练模型，继承自BertPreTrainedModel
        :param bert_config: configuration for bert model
        """
        super(BertABSATagger, self).__init__(bert_config)
        # num_labels 样本的总数量
        self.num_labels = bert_config.num_labels
        #设置模型的配置
        self.tagger_config = TaggerConfig()
        # absa_type是linear或者[ gru, san, tfm, crf]
        self.tagger_config.absa_type = bert_config.absa_type.lower()
        #
        if bert_config.tfm_mode == 'finetune':
            # 使用预训练的BERT初始化并执行微调
            # print("Fine-tuning the pre-trained BERT...")
            self.bert = BertModel(bert_config)
        else:
            raise Exception("无效的transformer mode %s!!!" % bert_config.tfm_mode)
        self.bert_dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        # 固定BERT中的参数并将其视为特征提取器
        if bert_config.fix_tfm:
            # 在微调期间固定（预训练或随机初始化的）transformer的参数
            for p in self.bert.parameters():
                p.requires_grad = False
        # 表示进行处理的tagger层，可以是lstm，gru等
        self.tagger = None
        if self.tagger_config.absa_type == 'linear':
            # 倒数第二层的hidden size， 如果是linear
            penultimate_hidden_size = bert_config.hidden_size
        else:
            #设置dropout
            self.tagger_dropout = nn.Dropout(self.tagger_config.hidden_dropout_prob)
            if self.tagger_config.absa_type == 'lstm':
                self.tagger = LSTM(input_size=bert_config.hidden_size,
                                   hidden_size=self.tagger_config.hidden_size,
                                   bidirectional=self.tagger_config.bidirectional)
            elif self.tagger_config.absa_type == 'gru':
                self.tagger = GRU(input_size=bert_config.hidden_size,
                                  hidden_size=self.tagger_config.hidden_size,
                                  bidirectional=self.tagger_config.bidirectional)
            elif self.tagger_config.absa_type == 'tfm':
                # 使用transformer结构
                self.tagger = nn.TransformerEncoderLayer(d_model=bert_config.hidden_size,
                                                         nhead=12,
                                                         dim_feedforward=4*bert_config.hidden_size,
                                                         dropout=0.1)
            elif self.tagger_config.absa_type == 'san':
                # vanilla self attention networks
                self.tagger = SAN(d_model=bert_config.hidden_size, nhead=12, dropout=0.1)
            elif self.tagger_config.absa_type == 'crf':
                self.tagger = CRF(num_tags=self.num_labels)
            else:
                raise Exception('Unimplemented downstream tagger %s...' % self.tagger_config.absa_type)
            #penultimate_hidden_size 是tagger分类器处理完的维度
            penultimate_hidden_size = self.tagger_config.hidden_size
        #最后一个线性层，分类结果
        self.classifier = nn.Linear(penultimate_hidden_size, bert_config.num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        """
        前向传播
        :param input_ids:
        :param token_type_ids:
        :param attention_mask:
        :param labels:
        :param position_ids:
        :param head_mask:
        :return:
        """
        # bert返回结果  sequence_output, pooled_output, (hidden_states), (attentions)
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # 最后一个Bert层的隐藏状态的shape: (batch_size, seq_len, hidden_size), bert的输出结果作为我们的输入
        tagger_input = outputs[0]
        #做一次Dropout, 形状不变 (batch_size, seq_len, hidden_size)
        tagger_input = self.bert_dropout(tagger_input)
        #print("tagger_input.shape:", tagger_input.shape)
        if self.tagger is None or self.tagger_config.absa_type == 'crf':
            # 如果是选择的crf或者linear，直接使用线性分类器作为最后一层
            logits = self.classifier(tagger_input)
        else:
            if self.tagger_config.absa_type == 'lstm':
                # customized LSTM
                classifier_input, _ = self.tagger(tagger_input)
            elif self.tagger_config.absa_type == 'gru':
                # 如果是gru，把bert的输出，输入到gru模型中，作为classifier_input的输入
                classifier_input, _ = self.tagger(tagger_input)
            elif self.tagger_config.absa_type == 'san' or self.tagger_config.absa_type == 'tfm':
                # vanilla self-attention networks or transformer
                # adapt the input format for the transformer or self attention networks
                tagger_input = tagger_input.transpose(0, 1)
                classifier_input = self.tagger(tagger_input)
                classifier_input = classifier_input.transpose(0, 1)
            else:
                raise Exception("Unimplemented downstream tagger %s..." % self.tagger_config.absa_type)
            classifier_input = self.tagger_dropout(classifier_input)
            logits = self.classifier(classifier_input)
        #输出元素outputs元祖，把原来的outputs的从第二位开始的也追加到现在的outputs里面, logits维度[batch_size,seq_lenth, num_labels]
        outputs = (logits,) + outputs[2:]
        #labels的维度[batch_size, sequence_length], 如果labels存在，计算损失，否则，只返回输出logits
        if labels is not None:
            if self.tagger_config.absa_type != 'crf':
                #使用交叉熵计算损失
                loss_fct = CrossEntropyLoss()
                if attention_mask is not None:
                    #如果attention_mask存在时，损失的计算需要计算真实损失, attention_mask维度[batch_sieze,seq_len] flatten 到1维
                    active_loss = attention_mask.view(-1) == 1
                    # active_loss是bool值，padding的部分为false，没有padding的部分为true,
                    # logits的维度从[batch_size,seq_lenth, num_labels]变成 [batch_size * seq_length, num_labels],
                    # 然后只取active_loss为True部分的维度，计算真实损失 [True_length, num_labels]
                    active_logits = logits.view(-1, self.num_labels)[active_loss]
                    #真实的labels[batch_size,seq_lenth] --> 拉平 [batch_size * seq_lenth], 取active_loss为True的部分,[True_length]
                    active_labels = labels.view(-1)[active_loss]
                    # 使用交叉熵计算损失, active_logits [True_length, num_labels]  active_labels [True_length]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                outputs = (loss,) + outputs
            else:
                #通过crf计算损失，如果absa的type是crf
                log_likelihood = self.tagger(inputs=logits, tags=labels, mask=attention_mask)
                loss = -log_likelihood
                outputs = (loss,) + outputs
        return outputs


class XLNetABSATagger(XLNetPreTrainedModel):
    # TODO
    def __init__(self, xlnet_config):
        super(XLNetABSATagger, self).__init__(xlnet_config)
        self.num_labels = xlnet_config.num_labels
        self.xlnet = XLNetModel(xlnet_config)
        self.tagger_config = xlnet_config.absa_tagger_config
        self.tagger = None
        if self.tagger_config.tagger == '':
            # hidden size at the penultimate layer
            penultimate_hidden_size = xlnet_config.d_model
        else:
            self.tagger_dropout = nn.Dropout(self.tagger_config.hidden_dropout_prob)
            if self.tagger_config.tagger in ['RNN', 'LSTM', 'GRU']:
                # 2-layer bi-directional rnn decoder
                self.tagger = getattr(nn, self.tagger_config.tagger)(
                    input_size=xlnet_config.d_model, hidden_size=self.tagger_config.hidden_size//2,
                    num_layers=self.tagger_config.n_rnn_layers, batch_first=True, bidirectional=True)
            elif self.tagger_config.tagger in ['CRF']:
                # crf tagger
                raise Exception("Unimplemented now!!")
            else:
                raise Exception('Unimplemented tagger %s...' % self.tagger_config.tagger)
            penultimate_hidden_size = self.tagger_config.hidden_size
        self.tagger_dropout = nn.Dropout(self.tagger_config.hidden_dropout_prob)
        self.classifier = nn.Linear(penultimate_hidden_size, xlnet_config.num_labels)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, input_mask=None, attention_mask=None, mems=None,
                perm_mask=None, target_mapping=None, labels=None, head_mask=None):
        """

        :param input_ids: Indices of input sequence tokens in the vocabulary
        :param token_type_ids: A parallel sequence of tokens (can be used to indicate various portions of the inputs).
        The embeddings from these tokens will be summed with the respective token embeddings
        :param input_mask: Mask to avoid performing attention on padding token indices.
        :param attention_mask: Mask to avoid performing attention on padding token indices.
        :param mems: list of torch.FloatTensor (one for each layer):
        that contains pre-computed hidden-states (key and values in the attention blocks)
        :param perm_mask:
        :param target_mapping:
        :param labels:
        :param head_mask:
        :return:
        """
        transformer_outputs = self.xlnet(input_ids, token_type_ids=token_type_ids,
                                               input_mask=input_mask, attention_mask=attention_mask,
                                               mems=mems, perm_mask=perm_mask, target_mapping=target_mapping,
                                               head_mask=head_mask)
        # hidden states from the last transformer layer, xlnet has done the dropout,
        # no need to do the additional dropout
        tagger_input = transformer_outputs[0]

        if self.tagger is None:
            # regard classifier as the tagger
            logits = self.classifier(tagger_input)
        else:
            if self.tagger_config.tagger in ['RNN', 'LSTM', 'GRU']:
                classifier_input, _= self.tagger(tagger_input)
            else:
                raise Exception("Unimplemented tagger %s..." % self.tagger_config.tagger)
            classifier_input = self.tagger_dropout(classifier_input)
            logits = self.classifier(classifier_input)
        # transformer outputs: (last_hidden_state, mems, hidden_states, attentions)
        outputs = (logits,) + transformer_outputs[1:]

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs
