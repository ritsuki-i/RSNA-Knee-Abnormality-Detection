# -*- coding: utf-8 -*-
"""GeoQueryNet: 全断面のスライス特徴を患者座標つきトークンにし、所見ごとのクエリが 3D 上で注意を向ける。

  スライス (3ch = 前/中/後) ─ ConvNeXt ─ (C, g, g) ─ Linear ─┐
  トークン座標 (g, g, 3) ─ Fourier ─ MLP ──────────────────────┼─ 加算 → メモリ
  シリーズ種類 (断面×液体強調) ─ Embedding ────────────────────┘
  所見クエリ (12 × q) ─ TransformerDecoder(メモリへ交差注意) ─ 所見ごとの線形 → ロジット
"""
from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import N_TYPES


class FourierPE(nn.Module):
    def __init__(self, d, n_freq=6):
        super().__init__()
        self.register_buffer("freq", (2.0 ** torch.arange(n_freq)) * math.pi, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(3 + 6 * n_freq, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, xyz):
        a = xyz[..., None] * self.freq                       # (..., 3, F)
        f = torch.cat([xyz, a.sin().flatten(-2), a.cos().flatten(-2)], -1)
        return self.mlp(f)


class GeoQueryNet(nn.Module):
    def __init__(self, backbone="convnext_tiny.in12k_ft_in1k_384", pretrained=True, n_targets=12,
                 d=256, q_per=2, n_layers=3, n_heads=8, drop_path=0.1, use_coords=True, head="dec", gap=False,
                 pool=1, xattn=0):
        super().__init__()
        self.head, self.gap, self.pool = head, gap, pool
        self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="",
                                     drop_path_rate=drop_path)
        c = self.enc.num_features
        self.proj = nn.Sequential(nn.Linear(c, d), nn.LayerNorm(d))
        self.use_coords = use_coords
        self.pe = FourierPE(d)
        self.type_emb = nn.Embedding(N_TYPES, d)
        nn.init.normal_(self.type_emb.weight, std=0.02)       # 既定の N(0,1) だと中身の特徴を覆い隠す
        nn.init.normal_(self.pe.mlp[-1].weight, std=0.002)
        nn.init.zeros_(self.pe.mlp[-1].bias)
        self.n_targets, self.q_per = n_targets, q_per
        # 断面をまたぐトークン間の自己注意（座標の埋め込みを含むので、同じ場所の矢状断・冠状断・軸位断が情報を交換できる）
        self.xattn = None
        if xattn > 0:
            layer = nn.TransformerEncoderLayer(d, n_heads, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
            self.xattn = nn.TransformerEncoder(layer, xattn, enable_nested_tensor=False)
        if head == "dec":
            self.query = nn.Parameter(torch.randn(n_targets * q_per, d) * 0.02)
            layer = nn.TransformerDecoderLayer(d, n_heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
            self.dec = nn.TransformerDecoder(layer, n_layers)
            self.norm = nn.LayerNorm(d)
            self.head_w = nn.Parameter(torch.randn(n_targets, q_per * d) * 0.02)
        else:   # "mil": 所見ごとのゲート付き注意（トークン特徴に座標を含むので、場所を見て重みを決める）
            self.att_u = nn.Linear(d, 128)
            self.att_v = nn.Linear(d, 128)
            self.att_w = nn.Linear(128, n_targets)
            self.val = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
            self.norm = nn.LayerNorm(d)
            self.head_w = nn.Parameter(torch.randn(n_targets, d) * 0.02)
        if gap:   # 全トークン平均からの近道（学習の立ち上がりを速くする）
            self.gap_norm = nn.LayerNorm(d)
            self.gap_head = nn.Linear(d, n_targets)
            nn.init.zeros_(self.gap_head.bias)
        self.head_b = nn.Parameter(torch.zeros(n_targets))
        # スライスごとの補助ヘッド: 各スライスに検査のラベルで直接損失をかけ、エンコーダに強い信号を届ける
        self.inst_norm = nn.LayerNorm(c)
        self.inst_head = nn.Linear(c, n_targets)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255, persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255, persistent=False)

    def encode(self, imgs):
        """imgs: (N, 3, S, S) uint8 または float(0-255) → トークン (N, g*g, d), スライスのロジット (N, n_targets)

        N を 16 の倍数に詰め物してから通す。cuDNN は入力の形が変わるたびに実行計画を作り直し、
        1 回 12〜30 秒かかるので、形の種類を減らす。
        """
        n = imgs.shape[0]
        pad = (-n) % 16
        if pad:
            imgs = torch.cat([imgs, imgs.new_zeros((pad,) + imgs.shape[1:])])
        x = (imgs.float() - self.mean) / self.std
        f = self.enc.forward_features(x.contiguous(memory_format=torch.channels_last))   # (N, C, g, g)
        f = f[:n]
        inst = self.inst_head(self.inst_norm(f.mean((2, 3))))
        if self.pool > 1:
            f = F.avg_pool2d(f.float(), self.pool)
        return self.proj(f.flatten(2).transpose(1, 2)), inst

    def memory(self, feats, coords, types):
        """feats: (N, g*g, d), coords: (N, g, g, 3), types: (N,) → (N, g*g, d)"""
        m = feats + self.type_emb(types)[:, None, :]
        if self.use_coords:
            if self.pool > 1:
                coords = F.avg_pool2d(coords.permute(0, 3, 1, 2).float(), self.pool).permute(0, 2, 3, 1)
            m = m + self.pe(coords.flatten(1, 2).to(feats.dtype))
        return m

    def decode(self, mem, pad_mask):
        """mem: (B, T, d), pad_mask: (B, T) True = 無効 → (B, n_targets) ロジット"""
        if self.xattn is not None:
            mem = self.xattn(mem, src_key_padding_mask=pad_mask)
        if self.head == "dec":
            q = self.query[None].expand(mem.shape[0], -1, -1)
            h = self.norm(self.dec(q, mem, memory_key_padding_mask=pad_mask))
            h = h.view(mem.shape[0], self.n_targets, self.q_per * h.shape[-1])
            out = (h * self.head_w).sum(-1) + self.head_b
        else:
            s = self.att_w(torch.tanh(self.att_u(mem)) * torch.sigmoid(self.att_v(mem))).float()   # (B, T, n)
            if pad_mask is not None:
                s = s.masked_fill(pad_mask[..., None], float("-inf"))
            a = s.softmax(1)
            z = self.norm(torch.einsum("btn,btd->bnd", a, self.val(mem).float()))
            out = (z * self.head_w).sum(-1) + self.head_b
        if self.gap:
            if pad_mask is None:
                g = mem.float().mean(1)
            else:
                keep = (~pad_mask).float()[..., None]
                g = (mem.float() * keep).sum(1) / keep.sum(1).clamp(min=1)
            out = out + self.gap_head(self.gap_norm(g))
        return out

    def forward(self, imgs, coords, types, slice_mask):
        """imgs: (B, N, 3, S, S), coords: (B, N, g, g, 3), types: (B, N), slice_mask: (B, N) True = 有効"""
        b, n = imgs.shape[:2]
        valid = slice_mask.flatten()
        feats, inst = self.encode(imgs.flatten(0, 1)[valid])
        g2 = feats.shape[1]
        inst_full = inst.new_zeros(b * n, inst.shape[-1])
        inst_full[valid] = inst
        mem = self.memory(feats, coords.flatten(0, 1)[valid], types.flatten()[valid])
        full = mem.new_zeros(b * n, g2, mem.shape[-1])
        full[valid] = mem
        full = full.view(b, n * g2, -1)
        pad = (~slice_mask)[:, :, None].expand(b, n, g2).reshape(b, n * g2)
        return self.decode(full, pad), inst_full.view(b, n, -1)
