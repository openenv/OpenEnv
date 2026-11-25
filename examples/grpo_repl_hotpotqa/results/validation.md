# Validation record

Validation was performed on 24–25 November 2026. These are observed results, not a claim that every optional
environment or platform is supported. Numerical experiment artifacts and immutable runtime/model/dataset revisions
are recorded alongside this file in `provenance.json`, `results.json`, `evaluation.csv` and `training.csv`.

## Committed regression tests

`tests/scripts/test_hotpotqa_training.py` covers:

- Real REPL execution, answer normalization, correct/incorrect rewards, reset, iteration limits and worker cleanup.
- Direct child queries and managed nonblocking child launch/wait through a real local HTTP server with scripted
  responses; the same environment behavior is exercised in the training and evaluation scripts.
- HF token forwarding only to the intended Jobs hostname; lookalike hosts do not receive the token.
- Keeping reset/reward/cleanup controls out of the model's tool set and keeping guided-demo instructions out of
  the scored evaluation prompt.
- Reporting exact-match accuracy separately from timeout-penalized reward, including negative-reward episodes.
- Closing workers after generation failures and training success/failure, and saving the final adapter/state.
- Preserving multimodal processor metadata during merge and exporting local/bucket outputs through the SDK.

The final focused selection passed **18 tests**, including parameterized cases, after the bucket-export fix.
Some evaluation/training control-flow cases use scripted model replies or small test doubles. They are complemented
by the actual optimizer, model-serving and GPU runs below; they do not alone establish training quality.

With the repository environment and the example's REPL/processor dependencies installed:

```bash
PYTHONPATH=src:envs uv run --no-sync pytest tests/scripts/test_hotpotqa_training.py -q
bash .claude/hooks/lint.sh
uv run --no-sync ruff check examples/grpo_repl_hotpotqa
uv run --no-sync ruff format --check examples/grpo_repl_hotpotqa
git diff --check
```

Optional-dependency skips are not equivalent to exercising those cases. Full repository lint and example lint /
format checks passed. The debug-code review found no new debugger statements or core debug changes.

## Broader Linux tests

Python 3.12 on Linux, with the CI dependencies and exactly the pre-submit hook's seven ignored files and marker
selection: **6,409 passed, 148 skipped, 129 deselected, 23 passing subtests, zero failures** in 1,516.63 seconds.
The five-minute hook limit was extended to allow this selection to finish. This was not an unrestricted run of
every optional Docker, network or integration test.

```bash
PYTHONPATH=src:envs uv run --no-sync pytest tests/ \
  --ignore=tests/envs/test_browsergym_environment.py \
  --ignore=tests/envs/test_dipg_environment.py \
  --ignore=tests/envs/test_websearch_environment.py \
  --ignore=tests/envs/test_sql_optim_environment.py \
  --ignore=tests/envs/test_python_codeact_reset.py \
  --ignore=tests/envs/test_python_codeact_rewards.py \
  --ignore=tests/envs/test_textarena_environment.py \
  -m "not integration and not network and not docker" -v --tb=short
```

The Linux Jobs harness gave its source directory normal executable permissions and acted as a process-local
child subreaper, reaping only already-exited adopted children, as a normal container init would. It did not alter
test, core or environment implementations. Earlier attempts exposed permission/reaping problems or timed out;
only the completed run is counted above. This full selection preceded the final small bucket-export extension;
the subsequent 18 focused tests and actual SDK exports exercised that extension.

Broader macOS runs were not fully green: an earlier run had 5,817 passes and four failures, and a later run had
5,818 passes and eight failures. These included existing upstream/dependency issues and macOS signal/permission
behavior. Three failures from the earlier run reproduced on an untouched upstream checkout; its session-reaper
timing failure passed in isolation on both checkouts. A grading dependency was subsequently installed to match
CI and its targeted test passed. These runs are not presented as full macOS support or as a green full suite.

## Actual integration and artifact checks

| Check | Observed result and scope |
| --- | --- |
| CPU TRL integration | A tiny randomly initialized model completed an optimizer step with REPL rewards 0/1, a nonzero LoRA update, adapter save and merge. Evaluation parsing used the actual Qwen tokenizer. Repeated with the pinned runtime. Scripted generations; not a quality benchmark. |
| Worker lifecycle stress | 30 real reset/child/wait/reward/close cycles and 60 successful scripted HTTP child requests; thread, descriptor and child-process counts remained bounded. |
| Initial CUDA smoke | Two training steps plus five matched evaluation questions per model completed. Both scored 1/5; this smoke check did not establish an accuracy gain. |
| Recorded CUDA experiment | Actual Qwen3.5-4B GRPO training reached global step 50 with 1,600 saved episodes. The root used an RTX PRO 6000 96 GB; a frozen Qwen3-0.6B child was served separately on an A10G 24 GB. |
| Training continuation | After a diagnostic process crash, standard Trainer resume from checkpoint 10 retained optimizer/RNG state and the first ten metric rows. Final history contains steps 1–50; cost includes both segments and discarded work. The example adds no custom resume mechanism. |
| Adapter audit | All 692 adapter tensors were finite; 248 of 346 LoRA B tensors were nonzero. Unused vision parameters remaining zero are expected. |
| Matched evaluation | Same initial prompts/tool schema and the same first 100 validation IDs for base and trained roots; exact match 6/100 versus 56/100. One seed and a fixed slice; no SFT warmup. |
| Async API with GPU child | A manual root launched a real frozen GPU child nonblockingly, searched independently, then waited for the answer and received reward 1. This is not a model-generated delegation trajectory or a speedup benchmark. |
| Learned delegation checks | Zero child runs in trained scored evaluation; three guided HotpotQA cases and one guided synthetic case also skipped delegation. Training had 11 async starts but no complete launch/separate-search/wait sequence was found. |
| CPU SDK export | A tiny model was merged, uploaded to a private bucket, downloaded and reloaded with equal weights and usable tokenizer metadata. |
| Full 4B export | The corrected public merge script completed a direct SDK bucket export. Its safetensors header, tensor shapes/dtypes and content hash matched the evaluated weights. Processor/tokenizer files matched except the serialization-only `is_local` flag in tokenizer configuration. |
| Analysis | Plots and tables were regenerated from saved trainer state and matched evaluation JSONL. Input SHA-256 hashes are in `results.json`; uncertainty uses Wilson intervals and a seeded paired bootstrap. Figures were inspected visually. |
| Documentation | HF doc-builder processed 93 pages; HTML build, internal links, tutorial content, code fences and the actual result figure were checked. Hosted preview checks depend on PR infrastructure. |

The initial mounted merge export retained a complete temporary weights file without the expected final filename
and processor files. Its length/header were validated before content-addressed recovery. The public merge script
now serializes locally and synchronously uploads through `HfApi.sync_bucket`, with no remote deletion. Both the
tiny reload test and the full 4B export above validate that correction.

## Interpretation and review points

The results support improved REPL question answering on this slice, **not learned recursive or parallel reasoning**.
The async walkthrough validates the API mechanism separately. The paired accuracy increase is 50 percentage
points, with a 95% paired-bootstrap interval of 40–60 points; this does not remove the one-run/fixed-slice limit.

Core OpenEnv 0.14 and TRL 1.18 are released dependencies. The REPL package retains immutable commit `3659dc7`
for the separate generated-code worker and credential environment allowlist required by the current isolation
invariant. This is an explicit exception to the maintainer's release-only request and needs review.

Experiment compute was approximately $10.50; including earlier development and validation, the runtime-based
estimate was $17.06. These exclude storage/subscription and are not invoices. All experiment Jobs were terminal
when accounting was finalized. Raw conversations, model weights and diagnostic logs were retained privately;
the committed tests, numerical artifacts and this record provide the public review evidence.
