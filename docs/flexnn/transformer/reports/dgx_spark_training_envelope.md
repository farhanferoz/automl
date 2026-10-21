# DGX Spark training envelope (R5b)

Scope: can a single NVIDIA DGX Spark unit train a small Python-focused coding LM (candidate
sizes ~125M, ~350M, ~1B parameters) from scratch, and what is the realistic wall-clock cost?
Every claim below carries a source URL fetched during this research pass (2026-07-18). Items
that could not be pinned down this run are listed under **OPEN** rather than asserted.

## 1. Base hardware specs (re-verified)

From NVIDIA's own product and docs pages
([nvidia.com/dgx-spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/),
[docs.nvidia.com hardware overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)):

- GB10 Grace Blackwell Superchip: 20-core Arm CPU (10× Cortex-X925 + 10× Cortex-A725), Blackwell
  GPU with 5th-gen Tensor Cores, 6,144 CUDA cores, 48 SMs.
- 128 GB LPDDR5x **unified** CPU+GPU memory, 256-bit / 16-channel interface, 4266 MHz
  (8533 MT/s effective) → **273 GB/s** memory bandwidth (same two sources; cross-checked against
  the [LMSYS DGX Spark review](https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark/), which
  independently states 273 GB/s and flags it as the likely bottleneck vs GDDR7/HBM parts).
  This bandwidth number matters directly below: it is shared between the CPU and GPU, so
  training and any concurrent data-loading/OS activity compete for the same 273 GB/s.
- Storage: 1 TB or 4 TB NVMe M.2 (self-encrypting). Power: 240 W external supply, GB10 SoC TDP
  140 W.
- NVIDIA's own marketing figure is **"up to 1 PFLOP FP4 with sparsity"** (≈1,000 TOPS) — this is
  an **inference** figure (FP4, sparse) and is not a training throughput number. NVIDIA's public
  pages do not publish BF16/FP8/FP32 dense TFLOPS figures.

The plan's carried-over "31 TFLOPS FP32" figure: **could not be found on any NVIDIA-authored
page**. It traces to third-party aggregator sites and a community benchmark, not an NVIDIA spec
sheet — see §2 and OPEN item 1.

## 2. Precision support for TRAINING (not inference)

### Measured dense throughput (community benchmark, not an NVIDIA spec)

A user-run `mmapeak` microbenchmark, posted and discussed on the [NVIDIA Developer Forum
thread "Detailed Compute Performance Metrics for DGX Spark"](https://forums.developer.nvidia.com/t/detailed-compute-performance-metrics-for-dgx-spark/351993),
reports (forum participants explicitly note **no official NVIDIA white paper exists** with these
numbers, so treat as best-available-but-unofficial):

| Precision | Measured dense TFLOPS |
|---|---|
| FP4 | ~427 |
| FP8 | ~214 |
| FP16 | ~213 |
| BF16 | ~213 |
| TF32 | ~53 |

(The forum's TF32 figure, ~53 TFLOPS, is the closest independent cross-check available for the
plan's "31 TFLOPS FP32" carry-over; the two numbers are in the same order of magnitude but do
not match exactly — see OPEN item 1.)

### NVFP4/MXFP8 training maturity on GB10 specifically

Blackwell natively supports NVFP4 (4-bit microscaled format) and this is validated for
**pretraining at datacenter scale** — NVIDIA's own research pretrained a 12B hybrid
Mamba-Transformer to a 10-trillion-token horizon in NVFP4 with no measurable accuracy loss vs an
FP8 baseline ([NVIDIA Technical Blog, "NVFP4 Trains with Precision of 16-Bit..."](https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/);
[arXiv:2509.25149, "Pretraining Large Language Models with NVFP4"](https://arxiv.org/html/2509.25149v2)).
**However, that validation is on datacenter Blackwell (B200/GB200, compute capability `sm_100`),
not the GB10's `sm_121`.** Two independent, dated (2026) sources establish that GB10's training
software stack lags behind:

- The [Kubesimplify "Day 3: DGX Spark Unpacked" post](https://blog.kubesimplify.com/day-3-the-dgx-spark-unpacked-gb10-unified-memory-sm-121-and-the-one-reason-this-hardware-exists)
  states explicitly: *"NVFP4 wins for inference. Training is a different story (gradients need
  higher precision in the backward pass), and on Spark today BF16 can actually train faster than
  NVFP4 at nanochat scale."* It also notes `sm_121` is a distinct CUDA target from datacenter
  `sm_100`, so "software built only for datacenter Blackwell is not automatically a native match,"
  and `sm_121` is typically the last variant to get prebuilt library support.
- An [NVIDIA Developer Forum thread, "Transformer Engine and GB10 - MXFP8 and MXFP4 training not
  yet supported?"](https://forums.developer.nvidia.com/t/transformer-engine-and-gb10-mxfp8-and-mxfp4-training-not-yet-supported/351220)
  confirms an `AssertionError: MXFP8 (for all gemm layouts) is not supported on 12.0+
  architectures yet` — i.e. MXFP8 **training** kernels are not functional on `sm_121` today. A
  related [vLLM issue](https://github.com/vllm-project/vllm/issues/43906) shows FP8 CUTLASS
  kernels also fail to dispatch on `sm_121a` and fall back to legacy (inference) kernels.

**Conclusion for this dossier: the realistic training precision on a DGX Spark today is BF16.**
NVFP4/FP8 training is a hardware capability of Blackwell in general but not a working software
path on this specific chip variant as of the sources fetched this run (this is a software-stack
state, not a permanent hardware ceiling — re-check before executing, see OPEN item 2).

## 3. Empirical training throughput anchor (real run, not theoretical peak)

Rather than derive wall-clock estimates from the unofficial peak-TFLOPS table above, this
dossier anchors on a **real completed training run on a single DGX Spark**, which implicitly
captures real thermal/software conditions that a theoretical-peak calculation would miss.

[objectgraph.com, "Training Your Own ChatGPT on DGX Spark: Getting nanochat to Work with CUDA
13.0"](https://objectgraph.com/blog/nanochat-dgx-spark/) reports training Andrej Karpathy's
`nanochat` speedrun model (560M params) on a single DGX Spark (119.7 GB usable memory):

- 11.2 billion tokens processed, 3.92 × 10¹⁹ total FLOPs, **261 hours** wall-clock, Model FLOPs
  Utilization (MFU) 4.35%.
- A companion account of the same effort in the [karpathy/nanochat GitHub Discussion #28,
  "Anyone managed to run training on an NVIDIA Spark yet?"](https://github.com/karpathy/nanochat/discussions/28)
  gives a slightly different total (~234.8–261 h across different posters/runs) and model config
  (20 layers, 1,280 model dim, 560.9M params) and quotes a per-step log line ("tok/sec: 1,659")
  that is **not** consistent with the run's own headline tokens/duration (see arithmetic below)
  — treated as a partial/misleading sub-metric, not used directly (OPEN item 3 flags this).

**Deriving a sustained-throughput number from the run's own headline totals** (FLOPs, tokens,
duration — the three numbers least likely to be internally inconsistent):

```
effective FLOP/s  = 3.92e19 FLOPs / (261 h × 3600 s/h)
                  = 3.92e19 / 939,600 s
                  ≈ 4.17e13 FLOP/s   (≈ 41.7 TFLOPS sustained, BF16)

cross-check via MFU:
  implied reference peak = 4.17e13 / 0.0435 (reported MFU)
                          ≈ 9.6e14 FLOP/s ≈ 0.96 PFLOP

cross-check via tokens:
  avg tokens/sec = 11.2e9 tokens / 939,600 s ≈ 11,920 tok/s
```

The implied reference peak (~0.96 PFLOP) lands almost exactly on NVIDIA's marketed "up to 1
PFLOP FP4-sparse" figure (§1), which confirms nanochat's own MFU calculation is using that
marketing figure as its denominator (even though the actual training precision is BF16, not
FP4) — i.e. the 4.17e13 FLOP/s figure is internally self-consistent and is adopted below as
**the empirical sustained-throughput anchor: ≈ 42 TFLOP/s (BF16, single DGX Spark, single dense
transformer, no tensor/pipeline parallelism)**.

Cross-check against the standard "6N FLOPs/token" training-cost approximation (forward +
backward; standard in the scaling-law literature, e.g. used throughout Kaplan et al. and
Hoffmann et al.): 6 × 560e6 params × 11.2e9 tokens = 3.76 × 10¹⁹ FLOPs, within ~4% of the
reported 3.92 × 10¹⁹ — close enough to use "6N·D" as the FLOPs estimator for the wall-clock
projections in §5.

## 4. Candidate architectures and memory budget arithmetic

To get concrete per-layer numbers, this dossier anchors the three candidate sizes to GPT-3's
published architecture table ([arXiv:2005.14165, "Language Models are Few-Shot Learners," Table
2.1](https://ar5iv.labs.arxiv.org/html/2005.14165)), since it is a well-documented, standard
reference architecture at exactly these scales:

| Candidate | n_params | n_layers (L) | d_model (h) | n_heads (a) |
|---|---|---|---|---|
| ~125M | 125M | 12 | 768 | 12 |
| ~350M | 350M | 24 | 1024 | 16 |
| ~1B | **1.3B** (closest documented anchor; not exactly 1B, used as-is) | 24 | 2048 | 24 |

(Note: the paper's own XL row has 24 heads × 128 head-dim = 3,072 ≠ 2,048 d_model — an
inconsistency in the source table itself, reproduced as quoted; does not affect the
parameter/layer/hidden-dim numbers used below.)

### Parameters + optimizer state (mixed-precision AdamW)

Standard mixed-precision training memory accounting is **16 bytes/parameter**: 2 B (BF16
weights) + 2 B (BF16 gradients) + 4 B (FP32 master weights) + 4+4 B (FP32 Adam first/second
moment). This is the convention used in the ZeRO paper and repeated across several
practitioner references surfaced this run (e.g. the breakdown in
[Michael Brenndoerfer's mixed-precision-training writeup](https://mbrenndoerfer.com/writing/mixed-precision-training-fp16-bf16-loss-scaling)
and the ZeRO-Offload paper, [arXiv:2101.06840](https://arxiv.org/pdf/2101.06840)).

```
125M:  125e6  × 16 B =  2.0 GB
350M:  350e6  × 16 B =  5.6 GB
1.3B:  1.3e9  × 16 B = 20.8 GB
```

All three are trivial relative to 128 GB.

### Activation memory (the actual constraint on batch size)

Using the per-layer transformer activation-memory formula from
[Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models,"
arXiv:2205.05198](https://arxiv.org/abs/2205.05198) (no tensor/sequence parallelism, i.e. t=1,
which matches a single-GPU run), **without any activation recomputation**:

```
per-layer activation memory = s · b · h · (34 + 5·a·s/h)   bytes
  s = sequence length, b = micro-batch size, h = d_model, a = n_heads
```

At s = 2048 (standard context length for this model class):

```
125M (h=768,  a=12, L=12): per-layer @ b=1 = 2048×768×(34+5·12·2048/768)  ≈ 291 MiB
                            × 12 layers ≈ 3.41 GiB @ b=1  →  ≈ 54.6 GiB @ b=16

350M (h=1024, a=16, L=24): per-layer @ b=1 = 2048×1024×(34+5·16·2048/1024) ≈ 388 MiB
                            × 24 layers ≈ 9.09 GiB @ b=1  →  ≈ 72.8 GiB @ b=8

1.3B (h=2048, a=24, L=24): per-layer @ b=1 = 2048×2048×(34+5·24·2048/2048) ≈ 616 MiB
                            × 24 layers ≈ 14.44 GiB @ b=1 →  ≈ 57.8 GiB @ b=4
```

Total (params+optimizer+activations, worst case, no checkpointing):

```
125M @ b=16:  2.0 + 54.6  ≈  56.6 GB  (44% of 128 GB)
350M @ b=8:   5.6 + 72.8  ≈  78.4 GB  (61% of 128 GB)
1.3B @ b=4:  20.8 + 57.8  ≈  78.6 GB  (61% of 128 GB)
```

**Finding: at these candidate sizes, unified memory is not the constraint.** Even without
activation recomputation (which the Korthikanti paper shows cuts activation memory ~5×), all
three candidates fit comfortably with room to spare for the OS/CPU share of the unified 128 GB
pool. The binding constraint is compute throughput (§5), not memory capacity.

## 5. Scaling-law token budgets and wall-clock estimates

**Which scaling law is standard now:** Chinchilla (Hoffmann et al. 2022, DeepMind) remains the
standard first-order reference for compute-optimal pretraining — its core "~20 tokens per
parameter" rule was re-verified this run via
[lifearchitect.ai's Chinchilla summary](https://lifearchitect.ai/chinchilla/), which quotes the
70B/1.4T-token example (1,400B ÷ 70B = 20). However, actual frontier practice **deliberately
overtrains past the Chinchilla-optimal point** when a model will serve a lot of inference,
because a smaller-but-longer-trained model is cheaper to serve. This refinement is formalized in
[Sardana & Frankle, "Beyond Chinchilla-Optimal: Accounting for Inference in Language Model
Scaling Laws," arXiv:2401.00448](https://arxiv.org/abs/2401.00448) (ICML 2024), which explicitly
recommends training smaller and longer than Chinchilla-optimal when there is meaningful
inference demand, and validates model quality continuing to improve out to token/parameter
ratios as high as 10,000 in their 47-model sweep.

Wall-clock below uses `T = 6·N·D / 42 TFLOP/s` (the §3 empirical anchor; `6·N·D` is the FLOPs
estimator cross-checked in §3).

### Chinchilla-compute-optimal (D = 20·N)

| Candidate | Tokens (20×N) | FLOPs (6ND) | Wall-clock @ 42 TFLOP/s |
|---|---|---|---|
| 125M | 2.5 B | 1.875e18 | **12.5 hours** (0.52 days) |
| 350M | 7.0 B | 1.47e19 | **97.9 hours** (4.1 days) |
| 1.3B | 26 B | 2.028e20 | **1,350 hours** (56.3 days) |

Sanity check: applying the same formula to the actual 560M/11.2B-token nanochat run (§3) gives
6×560e6×11.2e9 / 4.17e13 = 902,000 s ≈ 250.6 hours — matching the real reported 234.8–261 hour
range for that run to within ~7%, which supports using this formula/anchor combination for the
extrapolations above.

### Sensitivity to overtraining (1.3B and 350M candidates, since 125M is cheap at any of these ratios)

| Tokens/param | 1.3B tokens | 1.3B wall-clock | 350M tokens | 350M wall-clock |
|---|---|---|---|---|
| 20× (Chinchilla) | 26 B | 56.3 days | 7 B | 4.1 days |
| 100× | 130 B | 281.2 days (~9.2 months) | 35 B | 20.4 days |
| 200× | 260 B | 562.5 days (~1.5 years) | 70 B | 40.8 days |

For reference, real small-LM precedents span both ends of this range, none of which were run on
a single small unit:

- [TinyLlama (arXiv:2401.02385)](https://arxiv.org/html/2401.02385v2): 1.1B params, "trained on
  these tokens across approximately three epochs, cumulatively processing 3 trillion tokens"
  (base corpus ~950B unique tokens) — ≈2,727 tokens/param — on **16× A100-40GB**, at a reported
  24,000 tok/s per A100, ≈3,456 GPU-hours per 300B tokens (per the paper's own Table 2), i.e.
  ~90 days wall-clock using all 16 GPUs in parallel. Replicating this token/param ratio on one
  DGX Spark at the §3 anchor throughput would take years (not attempted in the table above —
  clearly outside the practical envelope).
- [phi-1 (arXiv:2306.11644)](https://arxiv.org/abs/2306.11644): 1.3B params, **only 7B tokens**
  total (6B filtered "textbook-quality" code + 1B GPT-3.5-synthesized textbooks/exercises) ≈5.4
  tokens/param — **far below** Chinchilla-optimal — trained in 4 days on 8× A100s, reaching
  50.6% HumanEval pass@1. A 350M sibling ("phi-1-small," same pipeline, 20 layers/1024
  dim/16 heads) reached 45% HumanEval. This is a direct existence proof that curated,
  domain-narrow (Python-specific) data can substitute for raw token count — directly relevant
  since R5a is inventorying exactly this kind of curated Python corpus.

## 6. Operational risk: sustained multi-day/week load

A multi-week single-unit training run is an **unattended, sustained near-100%-utilization**
workload — a different profile from the burst inference/fine-tuning DGX Spark is marketed for.
Multiple 2026 sources raise thermal concerns under sustained load:

- [NVIDIA Developer Forum, "DGX Spark Thermal throttling"](https://forums.developer.nvidia.com/t/dgx-spark-thermal-throttling/349647):
  users report throttling around 100 W (well under the 240 W supply) and CPU temps hitting 95°C
  under sustained load.
- [StorageReview, "NVIDIA DGX Spark Thermal Test"](https://www.storagereview.com/review/nvidia-dgx-spark-thermal-test-how-oem-cooling-designs-stack-up)
  and [Wild Pines AI, "Your DGX Spark Is Cooking Itself"](https://www.wildpines.ai/blog/your-dgx-spark-is-cooking-itself/)
  both document sustained high-80s-°C GPU temps over many-hour windows, with one case study
  reporting 62 of 73 hours at or above 85°C; a firmware update (ASUS v0103, March 2026) and
  cooling mitigations reportedly reduced peak temps by ~15-20°C and eliminated shutdowns in that
  case.
- General guidance surfaced across these sources: DGX Spark is designed for **bursty**
  workloads (serving, short fine-tunes, dev iteration), not sustained 24/7 near-peak utilization.

This does not invalidate the §3 throughput anchor (that 560M/11-day nanochat run is itself a
real multi-day near-continuous run, so whatever thermal behavior it experienced is already
baked into the 42 TFLOP/s figure) — but it means a 1.3B-scale, ~56-day (or longer, if
overtrained) run should be treated as carrying real reliability risk, not just a compute-time
cost. Frequent checkpointing and thermal monitoring are cheap prerequisites for such a run; see
OPEN item 4 for what remains unverified here.

## 7. Go/no-go: "1B-class Python model trainable on one DGX Spark unit"

**GO, unconditionally, for ~125M and ~350M.** Both fit memory comfortably (§4) and train in
hours-to-days even at multiples of the Chinchilla-optimal token budget (§5) — cheap enough to
run repeatedly (multiple seeds, ablations) on one unit.

**CONDITIONAL GO for ~1B-class (using the 1.3B GPT-3-XL architecture as the concrete anchor):**

- Memory fits comfortably (§4: ~79 GB of 128 GB at a Chinchilla-scale-appropriate batch size).
- At a Chinchilla-compute-optimal token budget (~26B tokens), wall-clock is **~56 days**
  continuous on one unit (§5) — a genuine multi-week dedicated commitment, but within the realm
  of a single research run, not a purchase-more-hardware blocker.
- Precision realistically available today is BF16 only; NVFP4/MXFP8 training kernels are not yet
  functional on this GB10 variant (§2), so there is no available near-term speedup from lower
  precision — the 56-day figure is not going to shrink via a precision switch alone right now.
- Pushing token/parameter ratio meaningfully past Chinchilla-optimal (the "deployment-grade,
  overtrained" regime that real small LMs like TinyLlama or Llama 3 use) is **not practical** on
  a single unit within normal research timelines — 100×/200× Chinchilla pushes wall-clock to
  9–18+ months (§5 sensitivity table).
- The empirical throughput anchor was measured on a 560M model; whether it holds, improves, or
  degrades at 1.3B is unverified (OPEN item 3) — larger models often achieve *better* MFU
  (more favorable arithmetic intensity), so 56 days may be a conservative (pessimistic) estimate,
  but this is not confirmed for this specific hardware.
- Sustained multi-week near-peak utilization carries real, documented thermal/reliability risk
  (§6) that must be actively managed (checkpointing, monitoring), not assumed away.
- **The load-bearing mitigation path is data quality, not more compute**: phi-1's existence proof
  (1.3B model, only 7B tokens, curated Python-heavy data, 5.4 tokens/param — far below
  Chinchilla-optimal, let alone the overtrained regime) suggests that if R5a's curated
  Python-instruction corpus can substitute for raw scale, the realistic budget for this specific
  (narrow-domain, Python-focused) 1B-class model could land closer to single-digit days than to
  56+ days or the 9–18-month overtrained figures. This is a hypothesis carried over from a
  different model/paradigm (phi-1 is a base code-completion model, not the same architecture or
  training objective as whatever R2–R4 will specify) and is **not proven for this programme's
  target model** — it is the reason to treat the 1B-class candidate as conditional-GO rather than
  either an unqualified GO or a NO-GO.

## OPEN / unresolved

1. **"31 TFLOPS FP32" figure (carried over from the plan) has no NVIDIA-authored source.**
   NVIDIA's own pages (nvidia.com, docs.nvidia.com) publish only the FP4-sparse "1 PFLOP"/"1,000
   TOPS" marketing figure — no official BF16/FP8/FP32 dense TFLOPS table was found. The closest
   independent numbers are a community `mmapeak` benchmark (BF16 ≈213, FP8 ≈214, TF32 ≈53
   TFLOPS, via the NVIDIA Developer Forum thread in §2) which does not exactly match "31 TFLOPS
   FP32" (TF32 ≈53 is the nearest analogue, not FP32 itself). Re-verify against an authoritative
   NVIDIA architecture whitepaper if one is published, before treating any dense-precision
   TFLOPS number as load-bearing.
2. **NVFP4/MXFP8 training-kernel maturity on `sm_121` is a software-stack snapshot, not a
   hardware ceiling.** The forum/blog sources (§2) are dated within 2026 but this is fast-moving;
   re-check Transformer Engine / PyTorch support before committing to BF16-only as a long-term
   assumption.
3. **The 42 TFLOP/s throughput anchor comes from a single 560M-parameter run; it was not
   independently validated at 350M or 1B/1.3B scale on DGX Spark.** No published benchmark of a
   ~1B-parameter training run on a single DGX Spark was found this run. The wall-clock
   projections in §5 assume constant achieved TFLOP/s across model size — a stated assumption,
   not a measured fact at the 1.3B point.
4. **Whether thermal throttling meaningfully degrades throughput over a multi-week (not
   multi-day) continuous run is unconfirmed.** The 560M/11-day nanochat run is the longest
   real training benchmark found; no report of a 4–8-week continuous training run (vs. the
   thermal reports in §6, which mostly describe inference/serving or shorter stress tests) was
   located.
5. **Exact epoch count / repetition factor for phi-1 and phi-1-small's ~7B-token dataset was not
   found.** The composition (6B filtered code + 1B synthetic) is confirmed, but whether phi-1-small
   (350M) reused the identical 7B-token set (implying a different, higher, tokens/parameter ratio
   than phi-1's 1.3B) or a scaled-down variant was not resolved in the sources fetched this run.
6. **GPT-3 Table 2.1's "XL" row has an internal inconsistency** (24 heads × 128 head-dim = 3,072,
   not equal to the stated 2,048 d_model) in the source as rendered — reproduced as-quoted since
   it does not affect the params/layers/hidden-dim numbers actually used in §4–5, but worth a
   sanity check if this exact architecture is adopted for R6.

## Sources (all fetched this run, 2026-07-18)

- https://www.nvidia.com/en-us/products/workstations/dgx-spark/
- https://docs.nvidia.com/dgx/dgx-spark/hardware.html
- https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark/
- https://forums.developer.nvidia.com/t/detailed-compute-performance-metrics-for-dgx-spark/351993
- https://blog.kubesimplify.com/day-3-the-dgx-spark-unpacked-gb10-unified-memory-sm-121-and-the-one-reason-this-hardware-exists
- https://forums.developer.nvidia.com/t/transformer-engine-and-gb10-mxfp8-and-mxfp4-training-not-yet-supported/351220
- https://github.com/vllm-project/vllm/issues/43906
- https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/
- https://arxiv.org/html/2509.25149v2
- https://objectgraph.com/blog/nanochat-dgx-spark/
- https://github.com/karpathy/nanochat/discussions/28
- https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks/
- https://ar5iv.labs.arxiv.org/html/2005.14165 (GPT-3, "Language Models are Few-Shot Learners")
- https://arxiv.org/abs/2205.05198 (Korthikanti et al., activation memory formula)
- https://mbrenndoerfer.com/writing/mixed-precision-training-fp16-bf16-loss-scaling
- https://arxiv.org/pdf/2101.06840 (ZeRO-Offload)
- https://lifearchitect.ai/chinchilla/ (Chinchilla 20 tokens/param rule)
- https://arxiv.org/abs/2401.00448 (Sardana & Frankle, "Beyond Chinchilla-Optimal")
- https://arxiv.org/html/2401.02385v2 (TinyLlama)
- https://arxiv.org/abs/2306.11644 (phi-1, "Textbooks Are All You Need")
- https://forums.developer.nvidia.com/t/dgx-spark-thermal-throttling/349647
- https://www.storagereview.com/review/nvidia-dgx-spark-thermal-test-how-oem-cooling-designs-stack-up
- https://www.wildpines.ai/blog/your-dgx-spark-is-cooking-itself/
