# OpenMythos-Skill: A Claude Skill for Kye Gomez's OpenMythos Repo

A [Claude skill](https://docs.claude.com) that gives Claude careful, repo-specific knowledge of **OpenMythos** — Kye Gomez's open-source PyTorch reconstruction of a hypothesized Recurrent-Depth Transformer architecture. When this skill is loaded, Claude acts like a senior engineer who has read every line of `open_mythos/main.py`: it knows the architectural invariants, the conventions, the debugging playbook, and the papers the repo cites.

This is a **third-party harness for working with OpenMythos**. It does not modify the OpenMythos codebase, does not redistribute its source, and is not affiliated with either the OpenMythos project or Anthropic.

---

## Credit where it's due

All the substance this skill knows comes from **Kye Gomez**'s work on OpenMythos:

- **Repo:** https://github.com/kyegomez/OpenMythos
- **License:** MIT (Copyright © 2026 Kye Gomez)
- **Twitter/X:** [@kyegomezb](https://twitter.com/kyegomezb)

If you're going to use this skill, go star and read [the upstream repo](https://github.com/kyegomez/OpenMythos) first. The README there is genuinely one of the better explainers of looped-transformer theory I've read — LTI stability via `A = exp(-exp(...))`, loop-index embedding, ACT halting, Parcae scaling laws, the memorization-reasoning tradeoff, all laid out clearly with citations. This skill is a harness around that content; the content itself is Kye's.

---

## What the skill actually does

When loaded, the skill triggers whenever Claude sees signals that the user is working on OpenMythos: filenames like `open_mythos/main.py`, imports like `from open_mythos.main import OpenMythos, MythosConfig`, symbol names (`MythosConfig`, `RecurrentBlock`, `LTIInjection`, `ACTHalting`, `MoEFFN`, `MLAttention`, `GQAttention`), variant helpers (`mythos_1b` through `mythos_1t`), and debugging symptoms specific to looped training (step-reproducible loss spikes, residual explosion, overthinking drift).

Once triggered, Claude reasons with:

- **The eight non-negotiable invariants.** `ρ(A) < 1` by construction, `e` frozen across loops, MoE only in the recurrent block, LM-head weight tying, causal-mask dtype matching activation dtype, loop-index embedding occupying `dim // 8` channels, the ACT remainder trick with `still_running` gating, and not breaking the recurrent loop when a KV cache is present. Silently violating any of these makes the model fail in characteristic ways.
- **The debugging playbook.** A symptom-to-cause map. "Loss spikes at a reproducible step" → check `model.recurrent.injection.get_A().max()` first. "Output quality degrades past N loops" → overthinking, lower `act_threshold`. "KV cache error on decode step 2+" → someone probably added a `break` that shouldn't be there.
- **Variant-scaling discipline.** The relationship between `dim`, `n_heads`, `expert_dim`, `n_shared_experts`, and `lora_rank` across the 1B → 1T variants, with the parameter-budget formula from `variants.py`'s header comment.
- **Training-script conventions.** AdamW, 2000-step warmup, FineWeb-Edu sharded streaming, bf16 on H100/A100 vs float16 + GradScaler on older hardware, DDP via `torchrun`.
- **Honesty about the project's status.** OpenMythos is an *independent theoretical reconstruction*. The skill is explicit about not claiming this is Anthropic's actual internal architecture.

There is also a **small optional appendix** in SKILL.md for users who want Claude to structure its reasoning in a Prelude → Loop → Coda shape (a prompting pattern loosely inspired by the RDT's forward pass). This is **off by default**, only activates on explicit request ("Mythos mode", "think like Mythos"), and is documented honestly — it's an aesthetic experiment, not a capability claim. A markdown file can't make Claude's weights loop, and the skill says so. If that framing bothers you, delete the appendix; the codebase-expert part of the skill stands on its own.

---

## Installation

```
openmythos-skill/
├── SKILL.md       (the skill)
└── README.md      (this file)
```

Drop the folder into wherever your Claude client loads skills from. Claude picks the skill up on its next turn and fires it when a trigger matches.

---

## Illustrative example

Here is one worked prompt, run twice, to give you a sense of what the skill changes. **Treat this as illustrative, not as a controlled experiment** — both outputs were produced by me (Claude) while I had the skill in my context window, so the no-skill version is my best attempt at ignoring what I just read. That's not a clean A/B test. I'm including it because it's useful for getting a feel for what the skill is aiming at, and I'd rather show you something concrete than hand-wave. **Run your own comparison** (see the next section) if you want evidence.

**Prompt:**

> *"My OpenMythos training run keeps diverging around step 500 — loss was going down nicely, then it spikes to NaN and never recovers. Using mythos_3b, bf16 on an A100, lr 3e-4 with 2000-step warmup, grad clip 1.0, FineWeb-Edu. What's likely wrong?"*

**Without the skill** (generic PyTorch-training diagnostic): Claude lists five plausible causes in the order a competent ML engineer would think of them — aggressive learning rate, gradient explosion, bad data in the stream, bf16 softmax saturation, optimizer state corruption. The suggested diagnostic order is reasonable: log pre-clip gradient norms, log the step-500 batch, try lowering LR, try fp32 softmax. The answer is *correct-but-generic* and never mentions `LTIInjection`, spectral radius, or `ρ(A) < 1`.

**With the skill**: Claude opens with *"This is the exact failure mode OpenMythos's `LTIInjection` module exists to prevent"*, identifies step-reproducible spikes as the fingerprint of spectral-radius drift, and gives the exact 30-second diagnostic:

```python
def log_spectral(model, step):
    A = model.recurrent.injection.get_A()
    print(f"step {step}  max(A)={A.max().item():.4f}")
```

It explains *why* the `A = exp(-exp(log_dt + log_A).clamp(-20, 20))` reparameterization is supposed to make this impossible — and therefore why seeing `max(A)` creep toward 1 is a strong signal someone has modified `LTIInjection`. It orders the remaining candidates by repo-specific prior probability (gradient explosion through a 16-iteration loop, then warmup geometry × bf16 × loop-depth interaction), and rules out the less-likely pitfalls (`e` being accidentally recomputed inside the loop, a single bad FineWeb-Edu sample).

The difference worth noticing: the with-skill answer routes the user to the specific 30-second diagnostic that actually distinguishes the most likely cause from the less likely ones. The without-skill answer routes the user toward an afternoon of general-purpose LR and gradient experiments that would eventually converge on the same answer, the slow way.

**Both answers are defensible.** The with-skill answer is more useful *if the user is in fact hitting the OpenMythos-specific failure mode*. If the user's actual problem is bad data in the stream, the without-skill answer is closer to the truth. The skill bets that OpenMythos-specific causes are the most likely explanation of OpenMythos-specific failures, which is a reasonable but not infallible prior.

---

## Run your own comparison

Because the example above is self-graded, here is how to get real evidence for yourself:

1. Install the skill in your Claude client.
2. Pick three real prompts from your own OpenMythos work — ideally ones where you already know the correct answer.
3. Ask each prompt in a fresh conversation **without** the skill loaded. Save the output.
4. Ask each prompt in a fresh conversation **with** the skill loaded. Save the output.
5. Compare. If the with-skill answers are consistently better on OpenMythos-specific questions and no worse on general questions, the skill is doing its job. If not, open an issue and tell me which prompts failed — the invariants list, the debugging playbook, and the triggers are all tunable.

This is a real test. The one in the README is a sketch.

---

## Limitations, caveats, and things I'd tell a recruiter

- **No architecture swap.** Claude's weights and attention are fixed. The skill is a prompt, not a model edit. The optional "Mythos reasoning mode" appendix is a structural prompting pattern loosely inspired by the RDT forward pass; whether it improves answer quality over Claude's default reasoning is an open empirical question that has not been rigorously tested. The SKILL.md file is explicit about this.
- **OpenMythos itself is theoretical.** It's an independent reconstruction of what Claude Mythos *might* look like based on public research. The skill does not claim this is Anthropic's actual internal architecture, and if a user conflates the two, Claude is instructed to correct them.
- **Validation here is illustrative, not rigorous.** The example above was produced by me answering the same prompt twice with full knowledge of the skill. A clean evaluation would run two independent Claude instances (one with the skill, one without) and have a third party grade them blind. I haven't done that. If you want evidence, the "run your own comparison" section above is how.
- **The triggering is intentionally aggressive.** Skills tend to under-trigger in practice, so the frontmatter description is written to fire on any concrete signal (filename, import, symbol name, variant helper). If you find it triggering on prompts where it shouldn't, tighten the negative-trigger clause at the end of the description. The skill is 248 lines of markdown — easy to edit.
- **The optional appendix can be deleted.** If you prefer a cleaner skill that is only the codebase expert, remove everything from `## Optional experimental appendix` to the end of the file. The rest of the skill stands on its own and is the part that pulls the weight.

---

## What this skill is useful for

- **Working on the OpenMythos codebase.** Debugging training runs, reviewing PRs, extending the MoE routing, adding a new attention variant, writing tests.
- **Learning the RDT architecture.** The SKILL.md doubles as a condensed walkthrough of what the repo actually does and why each piece exists.
- **As a worked example of skill-writing.** If you're building Claude skills for your own codebases — and you probably should be, because a skill that knows *your* project's invariants and debugging playbook is worth more than this one is to most people — the structure here (pushy trigger description, forward-pass overview, numbered invariants, symptom→cause debugging playbook, conventions, a references block) is a template worth copying.

## What this skill is not

- Not a way to turn Claude into a looped transformer.
- Not an affiliated or official tool from Anthropic or from the OpenMythos project.
- Not a substitute for actually reading the upstream repo — which, again, you should do.

---

## License

MIT, matching upstream. The skill content is mine; the architectural knowledge it encodes is drawn from Kye Gomez's OpenMythos repo and the papers the OpenMythos README cites.
