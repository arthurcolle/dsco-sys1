# Scaling sys1: from 31M to an 80B-active decision model

This is a plan, not a result. Numbers marked *estimate* are arithmetic from stated assumptions. Numbers marked *published* come from the cited papers. The only measured sys1 results so far are in the README.

## What we are scaling

sys1 is a decision model, not a chat model. It reads agent state (goal, recent steps, tool results, stated policy) and answers typed questions with calibrated probabilities:

- **choice:** pick one option from a set (next op-code, tool, route, handler)
- **score:** rate against ordered levels
- **noul:** probability that a yes/no question is true
- **box:** locate a described region in an image

Three things set it apart from a general LLM of the same size:

1. **A constructed vocabulary.** 100,352 tokens: about 95K learned BPE plus reserved regions for 2,062 agent-mechanics op-codes, 601 family/verb/object tokens, tool names and control markers. Every op-code is one token, so a decision is one step of the model, not a generated sentence.
2. **Typed heads instead of free text.** Pointer-choice, yes/no and value heads read one decision position. They train with proper scoring rules, so the probabilities can be calibrated and gated.
3. **Outcome-labeled agent data.** The model learns from real agent trajectories, labeled by what happened next (tool success, goal reached, user acceptance), from Jev decisions over generated situations, and from multi-provider adjudication of disagreements.

## The ladder

Each rung must pass a gate before money goes to the next one.

| Rung | Params (active / total) | Shape | Tokens | Training FLOPs (6ND, active) | H100-hours at 40% MFU | Cost at $2-4/hr |
|---|---|---|---|---|---|---|
| R0 | 31.8M dense | 12 x 384 | 0.03B | 6e15 | under 1 | local |
| R1 | 0.67B / 1.0B dense | 46-47 layers x 1152-1280 | 20-100B | 1.2e20 - 6e20 | 84 - 421 | $170 - $1.7K |
| R2 | 3B dense | ~40 x 2560 | 300B | 5.4e21 | 3.8K | $8K - $15K |
| R3 | 8B dense | ~40 x 4096, GQA | 2T | 9.6e22 | 67K | $135K - $270K |
| R4 | 16B / ~200B MoE | 48 layers, 128 experts, top-8 + shared | 6T | 5.8e23 | 400K | $0.8M - $1.6M |
| R5 | **80B / ~1.2T MoE** | ~80 layers, 256 fine-grained experts, top-8 + 1 shared | **30T** | **1.44e25** | **10.1M** | **$20M - $40M** |

*Estimates.* FLOPs use 6 x active params x tokens and ignore attention FLOPs, which add roughly 10-20% at 4-8K context. H100 dense bf16 peak is about 989 TFLOP/s; 40% MFU gives about 396 TFLOP/s sustained. FP8 training and newer accelerators (B200-class) would cut R4-R5 GPU-hours by roughly 2-4x.

### Reference points (*published*)

- DeepSeek-V3: 671B total, 37B active, 14.8T tokens, about 2.79M H800 GPU-hours for the full run (arXiv:2412.19437).
- Kimi K2: about 1T total, 32B active, 15.5T tokens (arXiv:2507.20534).
- Llama 3: 405B dense on about 15.6T tokens (arXiv:2407.21783).
- Fine-grained experts with shared experts: DeepSeekMoE (arXiv:2401.06066).
- Compute-optimal token counts: Chinchilla (arXiv:2203.15556). Small models are now routinely trained far past Chinchilla, because inference cost dominates. sys1 is meant to run on every agent step, so the same logic applies.

R5 is about 2x DeepSeek-V3's active parameters and 2x its tokens, so about 4x its training compute. That puts it at frontier scale for 2026. It only makes sense if R3-R4 show the decision metrics still improving with scale.

## Architecture by rung

**R0-R2 (dense, deep and thin).** Decoder-only, RMSNorm, rotary positions, grouped-query attention, SwiGLU, tied embeddings, LayerNorm scaling by 1/sqrt(depth) (arXiv:2502.05795). Below about 3B parameters, depth beats width at fixed size (MobileLLM, arXiv:2402.14905). The R1 presets already exist: `sys1-999m-v100k` (998.6M, 46 x 1280) and `sys1-666m-v100k` (669.3M, 47 x 1152).

**R3 (dense, wider).** Width grows faster than depth past a few billion parameters. Untie the embeddings once they fall below about 3% of parameters. Context goes from 4K to 32K so the model can read long tool outputs, with masked observations recalled by key.

**R4-R5 (mixture of experts).** Fine-grained routed experts plus always-on shared experts, auxiliary-loss-free load balancing, multi-token prediction as an auxiliary objective, and FP8 mixed precision. The decision heads stay dense and small. Only the trunk is sparse.

**Kept constant across rungs:** the vocabulary and op-code ids (append-only, so labels and evaluations carry across every rung), the typed heads, and the input format (state, then options, then the decide marker).

## Data at each scale

R5 needs about 30T tokens. Plausible sources:

| Source | Role | Scale |
|---|---|---|
| Filtered public web and educational text | language and world knowledge | most of the 30T |
| Permissively licensed code and configuration | tool arguments, schemas, shell, code edits | several T |
| Synthetic agent-format text | op-code definitions, decision situations, tool transcripts, schemas | 100B+ |
| Jev-labeled decisions over generated situations | calibrated soft targets for distillation | billions of decisions |
| Real agent trajectories with outcome labels | the signal that sets sys1 apart | grows with deployment |
| Adjudicated hard cases | disagreements resolved by several strong models | millions |

*Estimate:* Jev input costs $0.042 per million tokens, so labeling 1B decisions at about 500 tokens each costs about $21K. Decision supervision at scale is cheap. Diverse, realistic situations are the constraint.

The private trajectories never leave the operator's machines. Public releases ship the generators, formats and evaluations, not the private data.

## Gates between rungs

A rung passes only if all of these hold on held-out data:

1. **Scaling-law fit.** Language-modeling loss and decision loss follow a clean power law in compute across the rung's ablations. A flattening curve stops the ladder.
2. **Decision quality.** Next-action accuracy on held-out real sessions beats the strongest simple baseline (repeat-last was 64.1% at R0) and the previous rung by a set margin.
3. **Transfer.** Leave-one-family-out accuracy on decision families the model never trained on.
4. **Calibration.** Expected calibration error at or below 0.05 after temperature scaling, and selective accuracy at least 95% at the deployment confidence threshold.
5. **Hard cases.** Accuracy on multi-concern compositions. Jev reached about 76-83% there, so sys1 should beat that at R2 or above.
6. **Cost per decision.** Latency and serving cost low enough to run on every agent step.

## Training infrastructure

- **R0-R1:** local Apple Silicon. The 64 GB machine trains R1 with activation checkpointing. A second Mac joins through DiLoCo (arXiv:2311.08105): each machine trains its own replica and they average every few hundred steps, which suits LAN links far better than per-step gradient sync.
- **R2:** one 8-GPU node with plain data parallelism. A 1-GPU de-risking run comes first.
- **R3:** 32-128 GPUs with FSDP or ZeRO-3, sequence packing, and checkpoints every 30-60 minutes.
- **R4-R5:** expert parallelism plus pipeline and data parallelism across 1K-4K accelerators. FP8, fused MoE kernels and failure-tolerant checkpointing are required.

## What to open-source

**Publish:** the architecture and presets, the constructed vocabulary and op-code definitions, the corpus generators for synthetic agent data, the Jev map-reduce and adjudication harness, the evaluation suites and gates, this scaling plan, and the R0-R2 checkpoints with model cards.

**Withhold:** private trajectories and their labels, exact data-mix weights and filtering thresholds for the large rungs, and any operator-specific prompts or infrastructure details.

## Risks

- **Scale may not buy decision quality.** Many agent decisions are narrow and rule-like. A 3B model with good data might match an 80B one. That is exactly what the R2-R3 gates test before the big spend.
- **Benchmark leakage.** Situations generated from templates are easy to memorize. Hold out by family and template, and weight real-session evaluation most.
- **Label quality.** Outcome labels from logs are noisy: most episodes end "successfully". Better labels (tests passing, user acceptance, retries) matter more than more of the same.
- **Cost.** R5 is a $20M-$40M training run at 2026 prices before data and staff costs. It needs outside capital and a clear deployment path.
