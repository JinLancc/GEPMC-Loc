import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super(MLPBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class BioFeatureAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super(BioFeatureAdapter, self).__init__()
        self.compressor = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.compressor(x)


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16, dropout=0.1):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class LightweightGating(nn.Module):
    def __init__(self, input_dim, num_branches=3, reduction=12, dropout=0.3):
        super(LightweightGating, self).__init__()
        bottleneck_dim = max(input_dim // reduction, 8)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, num_branches),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.gate(x)


def sequence_to_one_hot_dynamic(sequences: List[str], max_len_limit: int) -> torch.Tensor:
    map_dict = {'A': 0, 'a': 0, 'C': 1, 'c': 1, 'G': 2, 'g': 2,
                'T': 3, 't': 3, 'U': 3, 'u': 3, 'N': 4, 'n': 4}
    batch_size = len(sequences)
    truncated_seqs = [seq[:max_len_limit] for seq in sequences]
    batch_max_len = max(len(seq) for seq in truncated_seqs)
    batch_max_len = max(batch_max_len, 10)
    indices = torch.full((batch_size, batch_max_len), 5, dtype=torch.long)
    for i, seq in enumerate(truncated_seqs):
        l = len(seq)
        idx_list = [map_dict.get(base, 4) for base in seq]
        indices[i, :l] = torch.tensor(idx_list, dtype=torch.long)
    one_hot = F.one_hot(indices, num_classes=6).permute(0, 2, 1).float()
    result = one_hot[:, :4, :].clone()
    return result


class MultiScaleCNN_Optimized(nn.Module):
    def __init__(self, in_channels=4, base_filters=32, kernel_sizes=[3, 5, 7],
                 cnn_dropout=0.3, att_dropout=0.1):
        super(MultiScaleCNN_Optimized, self).__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, base_filters, k, padding=k // 2),
                nn.BatchNorm1d(base_filters),
                nn.GELU(),
                nn.Dropout(cnn_dropout)
            ) for k in kernel_sizes
        ])
        total_filters = base_filters * len(kernel_sizes)
        self.se_block = SEBlock(total_filters, dropout=att_dropout)
        self.residual_proj = nn.Sequential(
            nn.Conv1d(in_channels, total_filters, kernel_size=1),
            nn.BatchNorm1d(total_filters)
        )
        self.output_dim = total_filters * 2

    def forward(self, x):
        conv_outputs = [conv(x) for conv in self.convs]
        main_feat = torch.cat(conv_outputs, dim=1)
        main_feat = self.se_block(main_feat)
        res_feat = self.residual_proj(x)
        fused_feat = main_feat + res_feat
        max_out = F.adaptive_max_pool1d(fused_feat, 1).squeeze(-1)
        avg_out = F.adaptive_avg_pool1d(fused_feat, 1).squeeze(-1)
        out = torch.cat([max_out, avg_out], dim=1)
        return out


class GEPMC_Loc(nn.Module):
    def __init__(self,
                 protrna_dim=1280,
                 ernierna_dim=768,
                 seq_max_len_limit=9216,
                 num_classes=2,
                 compressed_dim=128,
                 hidden_dim=64,
                 mlp_dropout=0.3,
                 cnn_dropout=0.3,
                 att_dropout=0.1):
        super(GEPMC_Loc, self).__init__()
        self.seq_max_len_limit = seq_max_len_limit
        self.adapt_prot = BioFeatureAdapter(protrna_dim, compressed_dim, dropout=mlp_dropout)
        self.mlp_prot = MLPBlock(compressed_dim, hidden_dim, dropout=mlp_dropout)
        self.head_prot = nn.Linear(hidden_dim, num_classes)
        self.adapt_ernie = BioFeatureAdapter(ernierna_dim, compressed_dim, dropout=mlp_dropout)
        self.mlp_ernie = MLPBlock(compressed_dim, hidden_dim, dropout=mlp_dropout)
        self.head_ernie = nn.Linear(hidden_dim, num_classes)
        self.cnn_extractor = MultiScaleCNN_Optimized(
            in_channels=4,
            base_filters=32,
            kernel_sizes=[3, 5, 7],
            cnn_dropout=cnn_dropout,
            att_dropout=att_dropout
        )
        cnn_out_dim = self.cnn_extractor.output_dim
        self.adapt_seq = BioFeatureAdapter(cnn_out_dim, compressed_dim, dropout=mlp_dropout)
        self.mlp_seq = MLPBlock(compressed_dim, hidden_dim, dropout=mlp_dropout)
        self.head_seq = nn.Linear(hidden_dim, num_classes)
        self.weight_gate = LightweightGating(
            input_dim=hidden_dim * 3,
            num_branches=3,
            reduction=12,
            dropout=mlp_dropout
        )

    def forward(self, protrna_emb, ernierna_emb, sequences: List[str]):
        device = protrna_emb.device
        prot_feat_raw = self.adapt_prot(protrna_emb)
        prot_feat = self.mlp_prot(prot_feat_raw)
        ernie_feat_raw = self.adapt_ernie(ernierna_emb)
        ernie_feat = self.mlp_ernie(ernie_feat_raw)
        one_hot = sequence_to_one_hot_dynamic(sequences, self.seq_max_len_limit).to(device)
        seq_feat_cnn = self.cnn_extractor(one_hot)
        seq_feat = self.mlp_seq(self.adapt_seq(seq_feat_cnn))
        logits_prot = self.head_prot(prot_feat)
        logits_ernie = self.head_ernie(ernie_feat)
        logits_seq = self.head_seq(seq_feat)
        concat_feat = torch.cat([prot_feat, ernie_feat, seq_feat], dim=1)
        weights = self.weight_gate(concat_feat)
        ensemble_logits = (
                weights[:, 0].unsqueeze(1) * logits_prot +
                weights[:, 1].unsqueeze(1) * logits_ernie +
                weights[:, 2].unsqueeze(1) * logits_seq
        )
        return ensemble_logits, (logits_prot, logits_ernie, logits_seq), (prot_feat, ernie_feat, seq_feat)
