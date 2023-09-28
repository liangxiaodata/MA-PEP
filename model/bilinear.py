import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from dgllife.model.gnn import GCN
from ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm
from pep_seq_encoder import BERT
import numpy as np


def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss


def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent


class MAPEP(nn.Module):
    def __init__(self, config):
        super(MAPEP, self).__init__()
        drug_hidden_feats = [128, 128, 128]
        num_filters = [128, 128, 128]
        mlp_in_dim = 256
        mlp_hidden_dim = 512
        mlp_out_dim = 128
        out_binary = 2
        ban_heads = 2
        self.peptide_extractor = BERT(config)
        self.feature_extractor = MyNet(input_size=512, output_size=128)
        self.feature_extractor2 = MyNet(input_size=1200, output_size=128)
        self.cross_attention = CrossAttention(128, 32)
        self.bcn = weight_norm(
            BANLayer(v_dim=drug_hidden_feats[-1], q_dim=num_filters[-1], h_dim=mlp_in_dim, h_out=ban_heads),
            name='h_mat', dim=None)
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim)
        self.classifier = nn.Linear(mlp_out_dim, 2)

    def forward(self, input_ids, other_feature):
        input_o1 = other_feature[:, :512] # morgan
        input_o2 = other_feature[:, 512:] # aaindex
        v_f1 = self.feature_extractor(input_o1)
        v_f2 = self.feature_extractor2(input_o2)
        v_f = self.cross_attention(v_f2, v_f1)
        _, v_p = self.peptide_extractor(input_ids)
        v_p = v_p.view(-1, 8, 128)
        f, att = self.bcn(v_f, v_p)
        score = self.mlp_classifier(f)
        logits_clsf = self.classifier(score)
        return logits_clsf, f


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        return x


class CrossAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.linear = nn.Linear(output_dim, 16 * 128)

    def forward(self, input1, input2):
        q = self.query(input1)
        k = self.key(input2)
        v = self.value(input2)
        attention_scores = torch.matmul(q, k.transpose(-2, -1))  # 计算注意力分数
        attention_weights = F.softmax(attention_scores, dim=-1)  # 归一化得到注意力权重
        fused_feature = torch.matmul(attention_weights, v)  # 加权融合特征
        output = self.linear(fused_feature)
        output = output.view(-1, 16, 128)
        return output

class MultiheadCrossAttention(nn.Module):
    def __init__(self, input_dim, output_dim, head_num):
        super(MultiheadCrossAttention, self).__init__()
        self.head_num = head_num
        self.out_dim = output_dim
        self.query = nn.Linear(input_dim, output_dim * head_num)
        self.key = nn.Linear(input_dim, output_dim * head_num)
        self.value = nn.Linear(input_dim, output_dim * head_num)
        self.linear1 = nn.Linear(output_dim * head_num, output_dim)
        self.linear2 = nn.Linear(output_dim, 16 * 128)

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        q = self.query(input1).view(batch_size, -1, self.head_num, self.out_dim).transpose(1, 2)
        k = self.key(input2).view(batch_size, -1, self.head_num, self.out_dim).transpose(1, 2)
        v = self.value(input2).view(batch_size, -1, self.head_num, self.out_dim).transpose(1, 2)
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.out_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)  # 归一化得到注意力权重
        fused_feature = torch.matmul(attention_weights, v)  # 加权融合特征
        fused_feature = fused_feature.transpose(1, 2).contiguous().view(batch_size, -1, self.out_dim * self.head_num)
        output = self.linear1(fused_feature)
        output = self.linear2(output)
        output = output.view(-1, 16, 128)
        return output


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 128)
        self.relu = nn.ReLU()

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(self.relu(self.fc1(x)))


class MyNet(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(MyNet, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, 512)
        self.linear2 = torch.nn.Linear(512, 256)
        self.linear3 = torch.nn.Linear(256, 256)
        self.linear5 = torch.nn.Linear(256, output_size)
        self.norm = torch.nn.LayerNorm(256)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.norm(self.linear3(x) + x)
        x = self.norm(self.linear3(x) + x)
        x = self.norm(self.linear3(x) + x)
        x = F.relu(self.linear5(x))
        return x
