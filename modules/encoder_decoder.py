import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .att_model import pack_wrapper, AttModel
import matplotlib.pyplot as plt

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed

    def forward(self, src, tgt, src_mask, tgt_mask, global_features=None):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask, global_features)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, hidden_states, src_mask, tgt, tgt_mask, global_features=None):
        return self.decoder(self.tgt_embed(tgt), hidden_states, global_features, src_mask, tgt_mask)

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)


    def forward(self, x, hidden_states, global_features, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, hidden_states, global_features, src_mask, tgt_mask)
        return self.norm(x)

class DualBranchDecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, global_attn, feed_forward, dropout):
        super(DualBranchDecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.global_attn = global_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ConditionalSublayerConnection(d_model, dropout), 4)  # 增加一个子层
        self.MLP = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1))
        self.w1 = nn.Linear(512, 512)
        self.w2 = nn.Linear(512, 512)
        self.w3 = nn.Linear(512, 512)
        self.w4 = nn.Linear(512, 512)

        self.w5 = nn.Linear(512, 512)
        self.w6 = nn.Linear(512, 512)
        self.w7 = nn.Linear(512, 512)
        self.w8 = nn.Linear(512, 512)

        self.score1 = nn.Linear(512, 512)
        self.score2 = nn.Linear(512, 512)
        self.score3 = nn.Linear(512, 512)
        self.score4 = nn.Linear(512, 512)
        self.relu = nn.ReLU()
        self.MLP4 = nn.Sequential(
            nn.Linear(1024, 768),
            nn.ReLU(),
            nn.Dropout(0.2),      # Dropout 防止过拟合
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.MLP = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU())

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.xavier_uniform_(self.w3.weight)
        nn.init.xavier_uniform_(self.w4.weight)

        nn.init.xavier_uniform_(self.w5.weight)
        nn.init.xavier_uniform_(self.w6.weight)
        nn.init.xavier_uniform_(self.w7.weight)
        nn.init.xavier_uniform_(self.w8.weight)

        nn.init.constant_(self.w1.bias, 0)
        nn.init.constant_(self.w2.bias, 0)
        nn.init.constant_(self.w3.bias, 0)
        nn.init.constant_(self.w4.bias, 0)

        nn.init.constant_(self.w5.bias, 0)
        nn.init.constant_(self.w6.bias, 0)
        nn.init.constant_(self.w7.bias, 0)
        nn.init.constant_(self.w8.bias, 0)
    def forward(self, x, hidden_states, global_features, src_mask, tgt_mask):
        m = hidden_states
        v_avg = torch.mean(m, dim=1).unsqueeze(1).expand_as(x)
        x_v = torch.cat((x, v_avg), 2)
        x = self.MLP4(x_v)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        hidden = x
        v_mean = torch.mean(m, dim=1)
        x_yuan = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        if global_features is not None:
            x_global = self.sublayer[2](x, lambda x: self.global_attn(x, global_features, global_features, src_mask))
            x_yuan = self.score1(self.relu(self.score2(x_yuan)))
            x_global = self.score3(self.relu(self.score4(x_global)))
            x_local = x_yuan - x_global
            score = torch.sigmoid(x_yuan)
            ones = torch.ones(score.shape).cuda()
            x = score * x_yuan + (ones - score) * x_local
        return self.sublayer[3](x, self.feed_forward)

class ConditionalSublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(ConditionalSublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)




#yuan relu-gelu
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=3, stride=1, dilation=2, padding=2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=3, stride=1, dilation=2, padding=2, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class DualBranchDecoder(nn.Module):
    def __init__(self, layer, N):
        super(DualBranchDecoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, hidden_states, global_features, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, hidden_states, global_features, src_mask, tgt_mask)
        return self.norm(x)

class EncoderDecoder(AttModel):

    def make_model(self, tgt_vocab):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        src_attn = MultiHeadedAttention(self.num_heads, self.d_model)
        global_attn = MultiHeadedAttention(self.num_heads, self.d_model)  # 全局注意力
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        model = Transformer(
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),
            DualBranchDecoder(
                DualBranchDecoderLayer(self.d_model, c(attn), c(src_attn), c(global_attn), c(ff), self.dropout),
                self.num_layers),
            lambda x: x,
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)), )
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, args, tokenizer):
        super(EncoderDecoder, self).__init__(args, tokenizer)
        self.args = args
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout

        tgt_vocab = self.vocab_size + 1

        self.model = self.make_model(tgt_vocab)
        self.logit = nn.Linear(args.d_model, tgt_vocab)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        # 保持返回4个值，以兼容 att_model.py
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq=None)
        memory = self.model.encode(att_feats, att_masks)
        # 不返回 global_features
        return fc_feats, att_feats, memory, att_masks

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
        memory = self.model.encode(att_feats, att_masks)
        # 生成全局特征，例如通过全局平均池化
        global_features = torch.mean(memory, dim=1, keepdim=True).repeat(1, memory.size(1), 1)
        out = self.model.decode(memory, att_masks, seq, subsequent_mask(seq.size(1)).to(memory.device), global_features)
        outputs = F.log_softmax(self.logit(out), dim=-1)
        return outputs

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        # 生成全局特征
        global_features = torch.mean(memory, dim=1, keepdim=True).repeat(1, memory.size(1), 1)
        out = self.model.decode(memory, mask, ys, subsequent_mask(ys.size(1)).to(memory.device), global_features)
        return out[:, -1], [ys.unsqueeze(0)]
