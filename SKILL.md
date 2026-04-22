---
name: openmythos
description: Use this skill when the user is working with the OpenMythos codebase — a PyTorch implementation of a hypothesized Recurrent-Depth Transformer (RDT) architecture by Kye Gomez. Trigger on any of these signals: files named `main.py` in an `open_mythos/` directory; imports like `from open_mythos.main import OpenMythos, MythosConfig`; variant helpers `mythos_1b`/`mythos_3b`/`mythos_10b`/`mythos_50b`/`mythos_100b`/`mythos_500b`/`mythos_1t`; mentions of `MythosConfig`, `RecurrentBlock`, `LTIInjection`, `ACTHalting`, `MoEFFN`, `MLAttention`, `GQAttention`, `LoRAAdapter`; discussion of OpenMythos-specific concepts like the Prelude/Recurrent/Coda three-stage layout, the `h_{t+1} = A·h_t + B·e + Transformer(h_t, e)` update rule, spectral-radius-less-than-one stability via the `exp(-exp(...))` reparameterization, loop-index embeddings on a `dim // 8` channel slice, the ACT remainder trick with `still_running` gating, or the `n_shared_experts` + `n_experts_per_tok` MoE scheme the repo uses; training scripts involving FineWeb-Edu and a looped model; and debugging symptoms that are specific fingerprints of looped training (residual explosion, step-reproducible loss spikes from injection-parameter spectral drift, overthinking degradation past the convergence point). A filename or a matching symbol is sufficient — the user does not need to explicitly say "OpenMythos". Do not trigger for generic transformer, generic MoE, or generic ACT questions that don't involve OpenMythos's specific implementation.
---

# OpenMythos

OpenMythos is an open-source PyTorch reconstruction of a hypothesized **Claude Mythos** architecture, written by **Kye Gomez** ([github.com/kyegomez/OpenMythos](https://github.com/kyegomez/OpenMythos), MIT license). It implements a **Recurrent-Depth Transformer (RDT)** with three stages — **Prelude** (standard transformer blocks, run once), a **Recurrent Block** (one TransformerBlock looped up to `max_loop_iters` times with input injection at every step), and a **Coda** (standard transformer blocks, run once). Attention is switchable between GQA and MLA; the FFN inside the recurrent block is a fine-grained MoE with always-on shared experts.

The project is an *independent, theoretical* reconstruction. It is not affiliated with Anthropic. The README is careful with language like "suspected", "likely", and "most probable class of solution", and so is this skill — don't claim this is what Anthropic actually does internally. If the user conflates OpenMythos with real Claude internals, gently correct them.

This skill turns Claude into a careful senior engineer who knows this specific repo. That's the entire job. (There is an optional experimental appendix at the bottom for users who want Claude to roleplay reasoning in the RDT's Prelude → Loop → Coda shape, but it is off by default.)

## The repo at a glance

```
open_mythos/
├── main.py         — MythosConfig, OpenMythos, all nn.Module classes (RMSNorm, GQAttention,
│                     MLAttention, MoEFFN, Expert, TransformerBlock, LoRAAdapter, LTIInjection,
│                     ACTHalting, RecurrentBlock), RoPE helpers, loop_index_embedding
├── variants.py     — mythos_1b / 3b / 10b / 50b / 100b / 500b / 1t preset configs
├── tokenizer.py    — MythosTokenizer wrapper (defaults to openai/gpt-oss-20b via HF)
└── __init__.py     — public re-exports

training/3b_fine_web_edu.py  — reference training script (DDP-ready via torchrun, FineWeb-Edu)
tests/                        — test_main.py, test_tokenizer.py, bench_vs_transformer.py,
                                small_benchmark.py, test_rope_debug.py
docs/                         — open_mythos.md (full class reference), datasets.md
examples/                     — moda_example.py, variants_example.py
example.py                    — minimal end-to-end sanity script at repo root
```

## The forward pass — hold this in your head

```
input_ids
  ↓ embed
  ↓ Prelude: prelude_layers × TransformerBlock (dense SwiGLU FFN, no MoE)
  e = x  ← encoded input is frozen here, re-injected every loop
  ↓
  RecurrentBlock (one block, looped up to n_loops times; uses MoE FFN):
    for t in range(n_loops):
        h_loop = loop_index_embedding(h, t, dim//8)   # RoPE-like signal on a slice of channels
        combined = RMSNorm(h_loop + e)                 # input injection into normed stream
        trans_out = TransformerBlock(combined) + LoRAAdapter(trans_out, t)   # per-depth LoRA delta
        h = A · h + B · e + trans_out                 # LTI-stable update (see below)
        p = sigmoid(halt(h))                          # ACT per-position halting probability
        # ACT remainder trick: if cumulative_p + p ≥ threshold, emit (1 - cumulative_p) as weight
        # gate by still_running so each position contributes exactly once on its halting step
        h_out += weight · h
  ↓
  Coda: coda_layers × TransformerBlock (dense SwiGLU FFN, no MoE)
  ↓ RMSNorm → LM head (weight-tied with embedding) → logits
```

Autoregressive generation uses KV caching with a separate cache key per loop depth (`recurrent_loop_{t}`) so every loop at every decode step finds populated keys.

## Non-negotiable invariants — if you break these, the model breaks

1. **`ρ(A) < 1` always.** The entire reason `LTIInjection` exists is to guarantee this by construction. `A = exp(-exp(log_dt + log_A))` sits element-wise in (0, 1). Never replace this with a free parameter, never initialize `A` as a raw `nn.Parameter` of shape `(dim,)`, never remove the `clamp(-20, 20)` — that clamp exists so `log_dt → -∞, log_A → +∞` doesn't produce `0 · inf = NaN`. If the user sees spectral-radius drift or residual explosion, this is the first thing to check.
2. **`e` is frozen across loops.** `e` is set once after the Prelude and re-injected at every loop iteration. This is what prevents drift across arbitrary recurrence depth. If someone accidentally recomputes `e` inside the loop, they have silently changed the architecture.
3. **MoE lives only in the Recurrent Block.** Prelude and Coda use dense SwiGLU FFNs (`use_moe=False`). The recurrent block uses `use_moe=True`. This separation is intentional: MoE provides breadth across domains inside the looped core; the Prelude/Coda are thin encode/decode shells.
4. **Weight-tying on the LM head.** `self.head.weight = self.embed.weight`. Don't break this by reinitializing `head` after construction.
5. **Causal mask dtype matches activation dtype.** The `_causal_mask` static method explicitly takes `dtype` because a bf16 activation stream with an fp32 additive mask silently upcasts attention logits to fp32, then the attn-vs-V matmul breaks. If you see a dtype error in the attention kernel, this is the usual suspect.
6. **Loop-index embedding occupies a slice of channels, not all of them.** `self.loop_dim = cfg.dim // 8`. The idea is that only a fraction of the residual stream carries the loop-index signal, leaving the rest undisturbed. Don't promote this to full-dim.
7. **ACT remainder trick with `still_running` gating.** When `act_threshold < 1.0` (it's 0.99 by default), a naive cumulative-probability update leaks a non-zero remainder on every subsequent step. The `still_running = ~halted` gate ensures each position contributes its halting weight exactly once. Don't remove it "to simplify".
8. **Don't `break` the loop when a KV cache is present.** If `kv_cache is None` and all positions have halted, breaking is fine. With a cache, every loop depth must execute on every prefill/decode step so that later decode steps find populated keys at every `cache_key`. This is explicit in `RecurrentBlock.forward`.

## Conventions used throughout `main.py`

- `nn.Module` subclasses have full docstrings with Args/Returns. Match the style when adding new modules; don't regress to terse or missing docstrings.
- RMSNorm, never LayerNorm.
- RoPE is applied to Q and K **before** KV caching, so cached values don't need to be re-rotated on retrieval. Keep this ordering.
- GQA uses the full per-head dim for RoPE; MLA uses only `qk_rope_head_dim` (the decoupled/split-RoPE scheme). The model registers two separate `freqs_cis` buffers and selects the right one based on `cfg.attn_type`. If you add a third attention type, register its own freqs buffer.
- Flash Attention 2 is optional. `GQAttention` probes `_HAS_FLASH_ATTN` and falls back transparently to manual SDPA. Keep the fallback path — CPU tests run without flash-attn.
- Weight init: `N(0, 0.02)` for every `nn.Linear` and `nn.Embedding`. Don't add per-layer init schemes without explicit reason.
- Dropout defaults to 0.0 (research default for pretraining sanity runs); 0.1 is standard when the user actually trains.

## Variant-scaling discipline

When asked to add or tune a scale variant in `variants.py`, stay consistent with the existing table: `dim`, `n_heads` roughly `dim // 128`, `n_kv_heads` roughly `n_heads // 4` (GQA) or 8–16 for large MLA, `expert_dim` solved from the residual parameter budget after all other terms. The header comment in `variants.py` is authoritative:

```
total ≈ embed + prelude/coda dense blocks + recurrent MLA + MoE
MoE   = 3 * dim * expert_dim * (n_experts + n_shared * n_experts_per_tok)
```

Don't blindly copy a smaller config up — larger scales intentionally bump `n_shared_experts`, `n_experts_per_tok`, and `lora_rank`, and the 100B+ tier raises `rope_theta` and enables `max_output_tokens=131072`.

## Training script conventions (`training/3b_fine_web_edu.py`)

- AdamW, linear warmup (2000 steps) → cosine decay, ~30B-token target (Chinchilla-adjusted for looped compute).
- bfloat16 on H100/A100; float16 + GradScaler on older GPUs. Don't mix the two paths.
- FineWeb-Edu `sample-10BT` is the default; `sample-100BT` or `default` swap in for the full run.
- DDP via `torchrun`; dataset sharded via streaming. If the user asks about single-GPU, point at `python training/3b_fine_web_edu.py`; multi-GPU is `torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") training/3b_fine_web_edu.py`.

## Debugging playbook

When the user reports a symptom, map it to a cause before speculating:

- **Loss spikes, NaNs, training diverges** → inspect `model.recurrent.injection.get_A()`. If any entry is ≥ 1, something broke `LTIInjection`'s reparameterization. If entries are fine, check gradient norm — unbounded grads through the loop can still spike loss even when A is bounded; gradient clipping is the answer, not changing A.
- **Hidden state blows up across loops at inference** → almost always the same root cause (spectral radius), but can also be that `e` was recomputed inside the loop by mistake. Add a `print(h.norm().item())` per loop step and watch the trajectory.
- **Output quality plateaus then degrades as `n_loops` increases** → "overthinking" drift. This is expected past a point; tune `act_threshold` downward or add a hard cap. Don't keep adding loops hoping for more quality.
- **KV cache error on decode step 2+** → someone probably modified `RecurrentBlock.forward` to break out of the loop when a cache is present. Put the `if halted.all() and kv_cache is None:` guard back.
- **Dtype mismatch in attention** → causal mask dtype. Pass `x.dtype` into `_causal_mask`.
- **MLA RoPE-vs-NoPE split confusion** → `qk_rope_head_dim + qk_nope_head_dim` is the full per-head query/key dim; RoPE only applies to the rope slice. `v_head_dim` is independent.

## When the user edits the code

- View the file before editing. `main.py` is ~1100 lines; don't edit blindly.
- Keep `__init__.py` re-exports in sync with any new public symbol.
- If you add a new module, add a test in `tests/test_main.py` that at minimum runs a forward pass, checks shapes, and asserts `ρ(A) < 1` afterward.
- The spectral-radius check (`torch.linalg.eigvals(A).abs().max().item() < 1`, or equivalently `A.max().item() < 1` for the diagonal parameterization) should be in any sanity script you write.

## External references the repo cites

When the user asks "why is it designed this way?", cite the paper rather than guessing:

- Parcae (Prairie et al., 2026) — the LTI stability fix and scaling laws for looped LMs. https://arxiv.org/abs/2604.12946 and the blog at https://sandyresearch.github.io/parcae/
- Universal Transformers (Dehghani et al., 2018) — original ACT halting for transformers. https://arxiv.org/pdf/1807.03819
- Reasoning with Latent Thoughts — On the Power of Looped Transformers (Saunshi et al., 2025). https://arxiv.org/abs/2502.17416
- Fine-grained experts + shared experts in MoE. https://arxiv.org/abs/2401.06066
- Relaxed Recursive Transformers — depth-wise LoRA for looped models (Bae et al., 2024). https://arxiv.org/pdf/2410.20672

---

## Optional experimental appendix — "Mythos reasoning mode"

This section is **off by default**. It only activates if the user explicitly asks for "Mythos mode", "think like Mythos", "loop-think this", or similar. When it is not explicitly requested, ignore it entirely and just behave as the codebase expert described above.

### What it is, honestly

This is a prompting pattern loosely inspired by the shape of the RDT forward pass. Claude's weights and attention do not change when this mode is active — a markdown file cannot turn one model into another. What changes is that Claude structures its reasoning as Prelude (compress the prompt) → Loop (iteratively refine, re-reading the original prompt each pass) → Coda (emit). Whether this produces better answers than Claude's default reasoning is an open empirical question that has not been rigorously tested. Treat it as an aesthetic/structural experiment, not a capability claim.

### If the user activates it

- **Prelude:** briefly note what the prompt is actually asking for, including any constraints and hidden assumptions.
- **Loop:** iterate 1–4 times on the answer. Each pass, re-read the original prompt before continuing — this is what prevents the reasoning from drifting into tangentially-related territory.
- **Halt early.** If two successive loops produce the same answer with only cosmetic changes, stop. More loops past convergence is the "overthinking" failure mode the README warns about, and it produces worse answers, not better ones.
- **Coda:** emit the final answer in natural language. Don't re-open questions the loop already settled.

### Things to avoid even when the mode is active

- **Reasoning theater on trivial prompts.** "What's the capital of France?" gets "Paris" even in Mythos mode. If a question needs zero loops, give it zero loops.
- **Claiming this is literally an RDT forward pass.** It isn't. It's a prompting pattern that vaguely echoes one.
- **Exposing the loop structure by default.** The user wanted a better answer, not a tour of the scratchpad. Only show the Prelude/Loop/Coda structure if the user explicitly asks to see it, the problem is genuinely hard and structure helps, or the meta-pattern is itself the point (e.g. teaching the RDT concept).
- **Ignoring the halting signal.** If loop 3 and loop 4 produce the same answer, stop at 3.

If you are in Mythos mode and the user asks a direct code question about the OpenMythos repo, drop the mode for that question and answer as the codebase expert. Don't perform the loop ritual on "what file is `LTIInjection` in".
