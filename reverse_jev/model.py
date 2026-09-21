"""MdLMMoE — vendored from alrobles/ecoreasoner scripts/train_mdlm_moe_v2.py.

Masked-diffusion backbone (bidirectional transformer, MoE FFN) plus the
decision readout heads that turn it into a System-One-style model.

Checkpoint compatibility: loads ecoreasoner `checkpoint-gN/model.pt` files
({"model": state_dict}, optional "module." prefix from DDP).
"""
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _default_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, std=0.02)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class MoEMLP(nn.Module):
    """Sparse MLP FFN with top-k router over n_experts."""

    def __init__(self, dim, ff, n_experts, k):
        super().__init__()
        self.n, self.k = n_experts, k
        self.gate = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, ff), nn.GELU(), nn.Linear(ff, dim))
            for _ in range(n_experts)])
        self.register_buffer("_fcount", torch.zeros(n_experts), persistent=False)
        self._gate_probs = None
        self._probe_x = None
        self._tokens = 0

    def forward(self, x):
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        g = torch.softmax(self.gate(flat).float(), dim=-1)
        gv, gi = g.topk(self.k, dim=-1)
        if self.training:
            self._gate_probs = g
            self._fcount.zero_()
            self._fcount.scatter_add_(0, gi.reshape(-1), torch.ones(gi.numel(), device=gi.device))
            self._tokens = gi.numel()
            s = min(self.n, flat.shape[0])
            self._probe_x = flat[torch.arange(s, device=flat.device)].detach()
        out = torch.zeros_like(flat)
        for rank in range(self.k):
            ids = gi[:, rank]
            w = gv[:, rank]
            for e in range(self.n):
                sel = (ids == e)
                if sel.any():
                    out[sel] += w[sel, None] * self.experts[e](flat[sel])
        return out.reshape(B, T, D)

    def balance_loss(self, alpha=0.01, probe_alpha=0.01):
        if self._gate_probs is None or self._tokens == 0 or self._probe_x is None:
            return torch.zeros((), device=self.gate.weight.device)
        P = self._gate_probs.mean(0)
        f = self._fcount.to(P.dtype) / max(self._tokens, 1)
        router_aux = alpha * self.n * (f * P).sum()
        probe = torch.zeros((), device=P.device)
        for e in range(self.n):
            ye = self.experts[e](self._probe_x)
            probe = probe + (ye ** 2).mean()
        probe = probe / self.n
        return router_aux + probe_alpha * probe


class RoPEMultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len=2048):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.d_head, 2).float() / self.d_head))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :])
        self.register_buffer("sin", emb.sin()[None, None, :, :])

    def _apply_rotary(self, x):
        d = x.shape[-1]
        x1, x2 = x[..., :d // 2], x[..., d // 2:]
        rot = torch.cat([-x2, x1], dim=-1)
        return x * self.cos[:, :, :x.size(2), :d] + rot * self.sin[:, :, :x.size(2), :d]

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self._apply_rotary(q)
        k = self._apply_rotary(k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out(out)


class Block(nn.Module):
    def __init__(self, dim, ff, heads, n_experts, k, use_rope=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = (RoPEMultiheadAttention(dim, heads) if use_rope
                     else nn.MultiheadAttention(dim, heads, batch_first=True))
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MoEMLP(dim, ff, n_experts, k)
        self._use_rope = use_rope

    def forward(self, x):
        h = self.ln1(x)
        if self._use_rope:
            a = self.attn(h)
        else:
            a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class TiedHead(nn.Module):
    def __init__(self, tok_emb, vocab):
        super().__init__()
        self.tok_emb = tok_emb
        self.vocab = vocab

    def forward(self, h):
        return F.linear(h, self.tok_emb.weight[:self.vocab])


class MdLMMoE(nn.Module):
    """Bidirectional masked-diffusion backbone. MASK token id == vocab."""

    def __init__(self, vocab, hidden, layers, heads, ff_mult, seq_len,
                 n_experts, k, use_rope=False, weight_tying=False):
        super().__init__()
        self.vocab = vocab
        self.use_rope = use_rope
        self.tok_emb = nn.Embedding(vocab + 1, hidden)
        self.pos = None if use_rope else nn.Embedding(seq_len, hidden)
        self.blocks = nn.ModuleList([
            Block(hidden, hidden * ff_mult, heads, n_experts, k, use_rope=use_rope)
            for _ in range(layers)])
        self.ln_f = nn.LayerNorm(hidden)
        self.head = (TiedHead(self.tok_emb, vocab) if weight_tying
                     else nn.Linear(hidden, vocab))
        self.apply(_default_init)

    def forward(self, ids, return_hidden=False):
        B, T = ids.shape
        h = self.tok_emb(ids)
        if not self.use_rope:
            h = h + self.pos(torch.arange(T, device=ids.device))
        for b in self.blocks:
            h = b(h)
        h = self.ln_f(h)
        logits = self.head(h)
        if return_hidden:
            return logits, h
        return logits

    @property
    def mask_id(self):
        return self.vocab

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class DecisionHead(nn.Module):
    """Laya-style option-marker scorer: hidden[marker] -> 1 logit.

    Same scorer applied to every marker position; softmax over a question's
    markers gives the option distribution (readout R2).
    """

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, h):
        return self.net(h).squeeze(-1)


DEFAULT_CONFIG = dict(vocab=126080, hidden=768, layers=12, heads=12,
                      ff_mult=4, seq_len=768, n_experts=8, k=1,
                      use_rope=False, weight_tying=False)


def load_backbone(ckpt_path, config=None, device="cpu"):
    """Load an ecoreasoner checkpoint into MdLMMoE.

    `config` is a dict like DEFAULT_CONFIG (harness yaml "model" section works).
    Falls back to DEFAULT_CONFIG; infers vocab/hidden from the state dict.
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update({k: v for k, v in config.items() if k in cfg})
    ck = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ck, dict) and "model" in ck:
        ck = ck["model"]
    if all(k.startswith("module.") for k in ck):
        ck = {k[len("module."):]: v for k, v in ck.items()}
    # infer dims from the state dict when possible
    emb_w = ck.get("tok_emb.weight")
    if emb_w is not None:
        cfg["vocab"] = emb_w.shape[0] - 1
        cfg["hidden"] = emb_w.shape[1]
    if "head.weight" in ck:
        cfg["weight_tying"] = False
    elif emb_w is not None:
        cfg["weight_tying"] = True
    cfg["layers"] = max(int(k.split(".")[1]) for k in ck
                        if k.startswith("blocks.") and k.split(".")[1].isdigit()) + 1
    cfg["use_rope"] = any("inv_freq" in k for k in ck)
    if "pos.weight" in ck:
        cfg["seq_len"] = ck["pos.weight"].shape[0]
    model = MdLMMoE(**cfg).to(device)
    model.load_state_dict(ck, strict=True)
    model.eval()
    return model
