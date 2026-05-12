# R2 paper-repro gate FAIL — root cause analysis (260511)

`tier1_simulator/r2_paper_repro.py` reports `FAIL` for both must-pass and should
targets when run against the modeled `dgx` / `dgx-attacc` paths. Re-running with
`BATCH=8` (instead of the original `BATCH=64`) did **not** close the gap.

| Target | Paper claim | Our result (BATCH=8) | Our result (BATCH=64) |
|---|---|---|---|
| FP16, vs DGX_Base | 4.84× (must) | 1.09× **FAIL** | 1.03× FAIL |
| FP16, vs DGX_Large | 2.48× (should) | n/a (DGX_Large not modeled) | n/a |
| INT8, vs DGX_Base  | 3.47× (should) | 1.08× **FAIL** | 1.03× FAIL |
| INT8, vs DGX_Large | 2.59× (should) | n/a | n/a |

Batch size is not the cause. The script is measuring the wrong metric.

---

## What the paper actually measures

Source: Park et al., "AttAcc! Unleashing the Power of PIM for Batched
Transformer-based Generative Model Inference," ASPLOS 2024
([PDF](https://scale.snu.ac.kr/papers/2024-04-Conference-ASPLOS-AttAcc.pdf),
[DOI](https://doi.org/10.1145/3620665.3640422)).

### System definitions (paper §7.1)

- `DGX_Base`: 40 HBMs, **640 GB total memory**. Standard DGX-A100 ×8.
- `DGX_Large`: 40 HBMs, **1,280 GB total memory**. Same compute/BW as
  DGX_Base, only memory capacity is doubled. (Hypothetical larger-mem
  baseline.)
- `DGX+AttAcc`: 40 HBMs in GPU side (1,280 GB) **plus** 40 AttAcc HBMs.
  AttAcc aggregate internal BW = 242 TB/s, **9× the GPU aggregate memory
  BW**. AttAcc handles the attention layer; DGX handles the FC layer.

### The 4.84× / 2.48× claim (§7.2, Fig. 14, GPT-3 175B, Lin=2048, Lout=128, no SLO)

Direct quote from the paper:

> "outperforming DGX_Base and DGX_Large, when running at the same batch size
> as DGX+AttAcc_s, in throughput by 4.84× and 2.48×, respectively"

This is a **throughput** comparison at the AttAcc-feasible batch size — not
a fixed-batch latency comparison. The paper's Fig. 14 reports the maximum
sustainable batch per system for GPT-3 175B / Lin=2048 / Lout=128 / no SLO:

| System | Max batch |
|---|---|
| DGX_Base   |   54 |
| DGX_Large  |  101 |
| DGX+AttAcc |  854 |

DGX_Base is memory-capacity-bound at batch ≈ 54. DGX+AttAcc can sustain
batch ≈ 854 because the KV cache lives on the AttAcc side (1,280 GB) and the
FC weights live on the DGX side. The 4.84× is the **steady-state throughput
ratio at each system's own maximum batch**, with AttAcc's attention
acceleration on top.

The 3.47× / 2.59× targets (paper text after Fig. 14) come from the same
methodology applied to a different `(Lin, Lout, SLO)` cell.

---

## Why r2_paper_repro.py FAILs

`tier1_simulator/r2_paper_repro.py` runs:

```python
sr.run(model="GPT-175B", system="dgx",        gpu="A100a", ngpu=8,
       num_attacc=8, num_hbm=5, batch=BATCH, lin=2048, lout=128, ...)
sr.run(model="GPT-175B", system="dgx-attacc", gpu="A100a", ngpu=8,
       num_attacc=8, num_hbm=5, batch=BATCH, lin=2048, lout=128, ...)
```

Then it computes `e2e_ms(dgx) / e2e_ms(dgx-attacc)` at the **same fixed
batch** for both systems and compares against 4.84× / 3.47×.

Three issues:

1. **Wrong metric.** The script compares end-to-end latency at a fixed
   batch. The paper claim is steady-state throughput at each system's
   *own* maximum batch (or hypothetical same-batch throughput on a system
   that physically cannot fit that batch).

2. **No DGX_Large path.** The script's CLI only has `--system dgx` and
   `--system dgx-attacc`. There is no separate `dgx-large` system that
   matches the paper's 1,280 GB-but-no-PIM baseline, so the 2.48× and
   2.59× should-targets are inevitably `skip`. The repo README already
   acknowledges this.

3. **Memory-capacity model mismatch.** The paper's DGX_Base is 640 GB
   across 40 HBMs (80 GB / GPU × 8 GPUs). DGX_Base's max-batch is limited
   by *attention-state* (KV cache) capacity, not just FC weight capacity.
   The current simulator reports per-GPU `required_cap` but
   `r2_paper_repro.py` does not enforce a system-wide capacity ceiling
   and does not back-solve for max sustainable batch. So even if we used
   throughput as the metric, BATCH would still be a hand-picked constant.

The combined effect: in the fixed-batch latency comparison, AttAcc only
helps the attention portion (a fraction of E2E at L=2048/Lout=128), so the
observed speedup is ≈1.03–1.09×, not 4.84×.

---

## What it would take to actually reproduce 4.84×

A faithful reproduction needs three changes that are out of scope for the
260511 in-flight pass:

1. **Add a `DGX_Large` system path.** Same `A100a` × `8` compute and BW as
   `dgx`, but with `MEM_CAPACITY_PER_DEVICE` doubled (or with a
   system-level cap of 1,280 GB). This unlocks the should-target row.

2. **Switch the metric to throughput at max sustainable batch.** Either:
   - First call `capacity_regime` logic per `(system, Lin, Lout)` to back
     out the SLO-feasible / memory-feasible max batch, then run E2E at
     that batch and report `batch / e2e_s` tokens-per-second.
   - Or hardcode the paper's Fig. 14 batch numbers (54 / 101 / 854 for
     GPT-3 175B Lin=2048 Lout=128 no-SLO) and run each system at its own
     batch.

3. **Compute `throughput_ratio = throughput(dgx-attacc) / throughput(dgx)`**
   instead of the current latency-at-same-batch ratio.

Both options preserve the existing `sim_runner` interface; only
`r2_paper_repro.py` and the targets table need to change.

---

## Implications for the rest of 260511

- The `multi_vlm_full_sim` results (2.13× – 5.54× per (VLM, batch)) are
  **not** affected by this issue. Those are batched VLM workloads at
  fixed `(Lin, Lout, batch)` where both dgx and dgx-attacc are run at the
  same batch — the speedup there is real, dominated by VLM-specific
  prefill / attention ratios, not by capacity-bound batch differences.
- The `capacity_regime` numbers (already in `results/`) directly show the
  max-batch differential, and are the right input to a corrected R2
  reproduction.
- The R2 gate `FAIL` should not block paper writeup as long as we cite
  this analysis and either fix `r2_paper_repro.py` per the plan above
  before the deadline, or footnote the gate as "current CLI does not
  model the throughput-at-max-batch metric used in the original paper."
