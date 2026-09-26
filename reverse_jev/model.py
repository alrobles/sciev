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
        self.seq_len = seq_len
        self.use_rope = use_rope
        self.model_config = dict(vocab=vocab, hidden=hidden, layers=layers,
                                 heads=heads, ff_mult=ff_mult, seq_len=seq_len,
                                 n_experts=n_experts, k=k, use_rope=use_rope,
                                 weight_tying=weight_tying)
        self.tok_emb = nn.Embedding(vocab + 1, hidden)
        self.pos = None if use_rope else nn.Embedding(seq_len, hidden)
        self.blocks = nn.ModuleList([
            Block(hidden, hidden * ff_mult, heads, n_experts, k, use_rope=use_rope)
            for _ in range(layers)])
        self.ln_f = nn.LayerNorm(hidden)
        self.head = (TiedHead(self.tok_emb, vocab) if weight_tying
                     else nn.Linear(hidden, vocab))
        self.apply(_default_init)

    def forward(self, ids, return_hidden=False, skip_head=False):
        B, T = ids.shape
        h = self.tok_emb(ids)
        if not self.use_rope:
            h = h + self.pos(torch.arange(T, device=ids.device))
        for b in self.blocks:
            h = b(h)
        h = self.ln_f(h)
        if skip_head:
            return h
        logits = self.head(h)
        if return_hidden:
            return logits, h
        return logits

    def hidden_layers(self, ids, layer_idx):
        """(L,B,T,D) stacked block outputs at the given indices (negatives ok);
        the last block's output goes through ln_f."""
        h = self.tok_emb(ids)
        if not self.use_rope:
            h = h + self.pos(torch.arange(ids.shape[1], device=ids.device))
        outs = []
        for b in self.blocks:
            h = b(h)
            outs.append(h)
        outs[-1] = self.ln_f(outs[-1])
        return torch.stack([outs[i] for i in layer_idx])

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


class AttnPoolHead(nn.Module):
    """Attention-pooling span scorer over stacked hidden layers.

    For each layer l, a learned query q_l weights the tokens of each option
    span (softmax over the span); the per-layer pooled features are
    concatenated and scored by a shared MLP. Richer than mean-pooling and
    reads knowledge from middle layers, not only the last one.

    forward(h_layers, bounds): h_layers (L,T,D) -> (K,) logits.
    """

    def __init__(self, dim, n_layers=1):
        super().__init__()
        self.n_layers = n_layers
        self.q = nn.Parameter(torch.randn(n_layers, dim) * 0.02)
        self.net = nn.Sequential(
            nn.LayerNorm(n_layers * dim),
            nn.Linear(n_layers * dim, dim), nn.GELU(),
            nn.Linear(dim, 1))

    def forward(self, h_layers, bounds):
        if h_layers.dim() == 2:
            h_layers = h_layers.unsqueeze(0)
        assert h_layers.shape[0] == self.n_layers
        feats = []
        for li in range(self.n_layers):
            q = self.q[li].float()
            pooled = []
            for s, e in bounds:
                span = h_layers[li, s:e].float()
                w = torch.softmax(span @ q, dim=0)
                pooled.append((w.unsqueeze(-1) * span).sum(0))
            feats.append(torch.stack(pooled))
        return self.net(torch.cat(feats, dim=-1)).squeeze(-1)


def forward_feats(model, head, ids, mode, pos, layers=(-1,)):
    """One forward + head call -> (K,) logits. Shared by train and eval.

    AttnPoolHead reads (L,T,D) stacked layers over span bounds (spanpool
    mode only); DecisionHead reads marker positions or mean-pooled spans.
    """
    if isinstance(head, AttnPoolHead):
        h = model.hidden_layers(ids.unsqueeze(0), layers).squeeze(1)
        return head(h, pos)
    h = model(ids.unsqueeze(0), skip_head=True).squeeze(0)
    feats = (h[pos] if mode == "marker"
             else torch.stack([h[s:e].mean(0) for s, e in pos]))
    return head(feats.float())


def marker_layout(ctx, options, mask_id):
    """Laya-style R2 layout: ctx + [M] opt_1 + [M] opt_2 + ...

    Returns (ids, marker_positions). The backbone's own MASK token is the
    marker — it is already trained as a 'fill me' slot whose hidden state
    attends bidirectionally to the option that follows it.
    """
    ids = list(ctx)
    pos = []
    for opt in options:
        pos.append(len(ids))
        ids.append(mask_id)
        ids.extend(opt)
    return ids, pos


def spanpool_layout(ctx, options):
    """R2-spanpool layout: ctx + opt_1 + opt_2 + ... (no markers).

    Returns (ids, span_bounds) where span_bounds = [(start, end), ...].
    Each option's score comes from the head over its mean hidden state —
    option tokens are observed, so their hidden states are contextualized
    content representations (unlike a MASK slot, which is optimized to
    predict a missing token).
    """
    ids = list(ctx)
    bounds = []
    for opt in options:
        start = len(ids)
        ids.extend(opt)
        bounds.append((start, len(ids)))
    return ids, bounds


DEFAULT_CONFIG = dict(vocab=126080, hidden=768, layers=12, heads=12,
                      ff_mult=4, seq_len=768, n_experts=8, k=1,
                      use_rope=False, weight_tying=False)


class HFBackbone(nn.Module):
    """Pretrained HF masked-diffusion LM (e.g. GSAI-ML/LLaDA-8B-Instruct)
    adapted to the MdLMMoE interface used by the decision-head pipeline.

    LLaDA attends bidirectionally by construction (no causal mask), so its
    hidden states are directly comparable to our backbone's for spanpool.
    """

    def __init__(self, name_or_path, device="cpu", dtype="bfloat16",
                 seq_len=2048, lora_adapter=None):
        super().__init__()
        from transformers import AutoModelForCausalLM
        self.hf_name = name_or_path
        self.hf = AutoModelForCausalLM.from_pretrained(
            name_or_path, trust_remote_code=True,
            torch_dtype=getattr(torch, dtype))
        if lora_adapter:
            from peft import PeftModel
            self.hf = PeftModel.from_pretrained(self.hf, lora_adapter)
            self.hf = self.hf.merge_and_unload()
        self.hf.to(device)
        self.tok_emb = self.hf.get_input_embeddings()
        cfg = self.hf.config
        self.vocab = getattr(cfg, "vocab_size", self.tok_emb.num_embeddings)
        self._mask_id = getattr(cfg, "mask_token_id", self.vocab - 1)
        self.seq_len = seq_len

    @property
    def mask_id(self):
        return self._mask_id

    def _hidden(self, ids):
        try:
            out = self.hf(input_ids=ids, output_hidden_states=True)
        except TypeError:
            out = self.hf(input_ids=ids)
        hs = getattr(out, "hidden_states", None)
        if hs:
            return hs[-1]
        inner = getattr(self.hf, "model", None)
        if inner is not None:
            try:
                o = inner(input_ids=ids, output_hidden_states=True)
            except TypeError:
                o = inner(input_ids=ids)
            hs = getattr(o, "hidden_states", None)
            return hs[-1] if hs else o.last_hidden_state
        raise RuntimeError("cannot extract hidden states from HF backbone")

    def forward(self, ids, return_hidden=False, skip_head=False):
        if skip_head:
            return self._hidden(ids)
        try:
            out = self.hf(input_ids=ids,
                          output_hidden_states=bool(return_hidden))
        except TypeError:
            out = self.hf(input_ids=ids)
        if return_hidden:
            hs = getattr(out, "hidden_states", None)
            return out.logits, (hs[-1] if hs else self._hidden(ids))
        return out.logits

    def hidden_layers(self, ids, layer_idx):
        """(L,B,T,D) hidden states at the given indices (negatives ok).
        HF hidden_states[0] is the embedding; [-1] is the final norm output."""
        try:
            out = self.hf(input_ids=ids, output_hidden_states=True)
        except TypeError:
            out = self.hf(input_ids=ids)
        hs = getattr(out, "hidden_states", None)
        if not hs:
            inner = getattr(self.hf, "model", None)
            o = inner(input_ids=ids, output_hidden_states=True)
            hs = o.hidden_states
        return torch.stack([hs[i] for i in layer_idx])

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def load_backbone(ckpt_path, config=None, device="cpu"):
    """Load an ecoreasoner checkpoint into MdLMMoE.

    `config` is a dict like DEFAULT_CONFIG (harness yaml "model" section works).
    Falls back to DEFAULT_CONFIG; infers vocab/hidden from the state dict.
    """
    cfg = dict(DEFAULT_CONFIG)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(ck, dict) and "model" in ck:
        saved_config = ck.get("model_config", {})
        if not isinstance(saved_config, dict):
            raise ValueError("checkpoint model_config must be an object")
        cfg.update({k: v for k, v in saved_config.items() if k in cfg})
        ck = ck["model"]
    if config:
        cfg.update({k: v for k, v in config.items() if k in cfg})
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


def load_decision(ckpt_path, config=None, device="cpu", lora_adapter=None):
    """Load a decision checkpoint: backbone + optional trained DecisionHead.

    Accepts either an ecoreasoner {"model": sd} file (returns head=None)
    or a reverse-jev {"model": sd, "head": sd} file — or an HF-adapter
    checkpoint {"hf_backbone": name, "head": sd} where only the head is
    stored and the backbone is fetched from HF/the local cache.
    """
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    head_sd = raw.get("head") if isinstance(raw, dict) else None
    if isinstance(raw, dict) and "hf_backbone" in raw:
        model = HFBackbone(raw["hf_backbone"], device=device,
                           lora_adapter=lora_adapter or raw.get("lora_adapter"))
    else:
        model = load_backbone(ckpt_path, config, device=device)
    head = None
    if head_sd is not None:
        meta = raw.get("meta", {}) if isinstance(raw, dict) else {}
        kind = raw.get("head_kind", meta.get("head_kind", "mlp")) \
            if isinstance(raw, dict) else "mlp"
        if kind == "attnpool":
            head = AttnPoolHead(model.tok_emb.embedding_dim,
                                n_layers=raw.get("n_layers", 1)).to(device)
        else:
            head = DecisionHead(model.tok_emb.embedding_dim).to(device)
        head.load_state_dict(head_sd)
        head.eval()
    return model, head
