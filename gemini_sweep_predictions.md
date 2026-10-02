# Gemini-router sweep: predictions recorded before the evals

Written while BabyAI phase B (LoRA) was training, so these are committed before
any eval cell exists. The point is to test the ALFWorld three-router conclusion
out of sample rather than re-reading it off new data.

## What ALFWorld claimed

Flash-lite routes failures into *both* channels. On ALFWorld that made its SFT
the best of three routers (+12 held-out) and its memory the worst and the only
net-negative one (-4 held-out). The claim: a lesson distilled from a failure is
a good training target and a bad retrieval entry.

Two corollaries, both of which contradict what three earlier benchmarks had
suggested:
- bank size does not predict harm (34 -> +8, 45 -> -4, 65 -> +4)
- replay yield does not predict usefulness (it ranked the three arms in reverse)

## BabyAI, phase A (disk-verified, 2026-10-01 20:08)

    route_counts        neither 151, both 33, memory 10, sft 2
    bank active         38   (full bank 40)
    sft candidates      35 replayed -> 7 verified, yield 0.20
    when base FAILED    both 32, memory 8, sft 2
    when base SUCCEEDED neither 67, memory 2, both 1

The ALFWorld signature reproduces exactly: Flash-lite overwhelmingly picks
`both`, and 32 of its 33 `both` decisions come from *failed* trajectories. It
leaves successes alone (67 of 70 -> neither).

Against the earlier Qwen-35B router on the same benchmark:

    router        bank   sft pool   yield
    Qwen-35B         3          7    0.25
    Flash-lite      38          7    0.20

Same SFT pool size, a 12x larger bank.

## Predictions

1. **router-memory underperforms.** Qwen's router-memory arm scored +2 on
   test80. Flash-lite writes 12x more bank entries, almost all distilled from
   failures, so its router-memory arm should land at or below +2, and may go
   negative. This is the direct out-of-sample test of the ALFWorld claim.

2. **router-SFT roughly matches Qwen's.** Both pools hold 7 examples, so the
   SFT arms should be close. Qwen's router-SFT was +20. No strong prediction
   about which is higher; a large gap in either direction would mean pool
   *size* is not what matters, which would be news.

3. **yield stays uninformative.** 0.20 vs 0.25 is a small difference and should
   not track the outcome.

4. **force_memory stays well above router-memory.** force_memory was +9 on
   test80 with a bank built from everything. If a 38-entry failure-distilled
   bank does worse than a write-everything bank, that isolates *what* was
   written rather than how much.

Prediction 1 is the one that matters. If Flash-lite's router-memory comes in
clearly above +2 on BabyAI, the ALFWorld conclusion was ALFWorld-specific and
the report's headline finding needs weakening.

---

# Outcomes

## BabyAI -- complete (2026-10-01 21:00)

All three router-dependent cells, 0 errors each. baseline/force_memory/force_sft
do not depend on the router and are reused from the Qwen grid.

    test80, baseline 23/80 = 0.2875

    cell                succ    pass   delta   qwen delta
    router memory         27  0.3375      +4           +2
    router SFT            33  0.4125     +10          +20
    router both           28  0.3500      +5          +19
    force_memory (reused)                 +9
    force_sft    (reused)                +16

Merge verified before the driver deleted the checkpoint: target
layers.0.linear_attn.in_proj_a rel=2.00e-03, embed/visual/experts all 0.00e+00.
So the SFT cells measure a real adapter, not an unmerged base.

### Prediction 1: WRONG
Predicted router-memory <= +2, possibly negative. Got +4.

### Prediction 2: WRONG
Predicted the SFT arms would be close, both pools holding 7 examples. Got +10
against Qwen's +20 -- 8 tasks of 80 apart. The consequence written down in
advance therefore applies: **pool size is not what matters; which examples are
in it is.**

### Prediction 4: HELD
force_memory (+9) stayed above router-memory (+4).

### The two failures point in OPPOSITE directions
Memory did better than predicted, SFT worse. So this is not one systematic bias
in how ALFWorld was read; it is that ALFWorld's three-router result did not
carry to a second benchmark at all. On ALFWorld, Flash-lite beat Qwen on SFT
(+12 vs +6). Here Qwen beats Flash-lite on SFT (+20 vs +10). The ranking
reverses.

### New finding this row DOES support: an interaction
Within the Flash-lite arm:

    memory alone   +4
    SFT alone     +10
    both           +5     <- adding the bank COSTS 5 tasks

The bank helps in isolation and hurts on top of SFT. The ALFWorld grid could not
show this: there `both` was >= `sft` on both splits (+13 vs +12 held-out, +43 vs
+39 in-distribution).

Candidate mechanism: the LoRA has already absorbed the failure lessons, so
retrieving the same post-mortems at inference competes with what the weights
encode. This predicts the interaction is strongest where the SFT pool and the
bank are distilled from the SAME trajectories -- which is exactly what the
`both` route produces. Testable by building a bank from trajectories disjoint
from the SFT pool.

### What BabyAI cannot settle
`test80` is a same-distribution hold-out, so it cannot separate "failure-
distilled entries are bad retrieval entries" from "retrieval does not survive
distribution shift". SQLGym `xdb149` (cross-schema) and TextCraft `deep127`
(compositional) can.
