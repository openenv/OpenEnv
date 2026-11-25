# Recorded experiment

One base-model GRPO run, 50 final optimizer steps and 1,600 episodes; matched greedy evaluation on the first
100 HotpotQA distractor validation questions. Base exact match: 6/100. GRPO exact match: 56/100. No SFT.

- `results.png` and `results.svg`: actual reward history, exact-match counts with Wilson intervals, and recursive child counts.
- `training.csv`: 50 recorded optimizer-step metrics.
- `evaluation.csv`: 100 matched question IDs with correctness, submission flags, reward, and recursive child counts for each model. `base_submitted` and `grpo_submitted` are 1 for any submitted answer (including wrong answers), 0 for no submission; totals are 37 and 92 respectively.
- `results.json`: aggregate metrics, submitted-answer totals, paired bootstrap interval, and SHA-256 hashes of the original analysis inputs.
- `provenance.json`: model/dataset revisions, runtime versions, checkpoint continuation, and behavioral limitations.
- `compute.json`: measured phase times and runtime-based compute estimates, excluding storage and subscription fees.
- `async-api-check.json`: a manual root using the real frozen GPU child through the nonblocking API. This is not a model-generated root trajectory or a speedup benchmark.
- [`validation.md`](validation.md): committed regression-test coverage, reproduction commands, completed checks and their limits.

The trained model started zero recursive child runs in scored evaluation and also skipped delegation in the guided
checks. These results demonstrate improved REPL question answering on this slice; they do not demonstrate learned
recursive or parallel reasoning. Raw evaluation conversations and model weights were retained privately and are
not included in these numerical artifacts.

Generate these plots and tables from your own saved run with `analyze.py`, as described in the parent README.
