# Train a Recursive Language Model: GRPO on HotpotQA with `repl_env`

This tutorial trains [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) to answer multi-hop HotpotQA questions as a
recursive language model (RLM) against OpenEnv's [`repl_env`](../environments/repl) environment. The root model's
initial prompt omits the article library: excerpts from ten Wikipedia articles live in the REPL's `context` variable,
and the root reads them through one tool, `execute_code`, which runs Python in the REPL. Retrieved text enters the
conversation as tool feedback. Inside the REPL it searches
the library with code and hands evidence checks to a frozen child model ([Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B))
with `llm_query` or `rlm_query`. GRPO trains a LoRA adapter on the root only; the child stays frozen. The reward is
normalized answer exact match, with a -0.1 penalty when the REPL iteration limit is exhausted.

The example implements [@jucamohedano's proposal in huggingface/trl#8524](https://github.com/huggingface/trl/issues/8524),
following [@sergiopaniego's feedback](https://github.com/huggingface/trl/issues/8524#issuecomment-7109917454). The scripts are in
[`examples/grpo_repl_hotpotqa`](https://github.com/openenv/OpenEnv/tree/main/examples/grpo_repl_hotpotqa). Training
starts from the base model, with no SFT warmup. The experiment tests whether outcome-reward training improves
its use of the REPL and delegated evidence checks.

## Results

The measured run completed **50 optimizer steps (1,600 episodes)** starting from the base model, with no SFT.
Both models then answered the same first 100 questions from the HotpotQA distractor validation split, in dataset
order, using greedy decoding and identical initial prompts and budgets:

| Model | Exact match | 95% Wilson interval | Mean reward | Mean recursive child runs per question |
|---|---|---|---|---|
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | 6/100 (6%) | 2.8–12.5% | 0.058 | 0.10 |
| + GRPO | 56/100 (56%) | 46.2–65.3% | 0.560 | 0.00 |

The paired improvement is **50 percentage points** (95% paired bootstrap interval: 40–60 points; 10,000 resamples,
seed 2026). The trained model retained all six base successes and answered 50 additional questions correctly.
It submitted an answer on 92 questions versus 37 for the base, with 429 parsed tool calls versus 1,038.
This is one training run on a fixed 100-question slice, not the full validation set or a multi-seed benchmark.
The intervals describe question-sampling uncertainty for that run; they do not measure training-seed variation.

**This result demonstrates improved REPL question answering, not learned recursive delegation.** The trained
model started no recursive children in the scored evaluation. It also skipped child queries in three separately
guided HotpotQA examples and one strongly guided synthetic example. A scan of the 1,600 training episodes found
11 valid nonblocking starts but no completed child-launch → separate context-search → valid-wait sequence.
The reward gives no bonus for delegation or parallel work. Showing an accuracy gain therefore does not establish
that the model learned to use those capabilities. The managed async API is exercised separately in the manual
walkthrough below, with a real frozen GPU child.

The [recorded plots and numerical artifacts](https://github.com/openenv/OpenEnv/tree/main/examples/grpo_repl_hotpotqa/results)
contain all 50 reward points, the actual evaluation counts and uncertainty intervals, per-question results,
source-file hashes, model/dataset revisions, and runtime versions. Rebuild the figures from saved outputs with:

```bash
uv run analyze.py --trainer-state ./adapter/trainer_state.json \
    --base base.jsonl --trained trained.jsonl --output-dir ./results
```

## What you need

The observed experiment used one NVIDIA RTX PRO 6000 96GB for the root and a separately served, frozen
Qwen3-0.6B on an A10G 24GB. Times below are allocated compute time, including dependency setup, model loading,
and endpoint waits; scheduling time is free. Root and child time are listed separately because they overlap.

| Phase | Root / CPU time | Child time | Estimated compute cost |
|---|---|---|---|
| Training, 50 final steps | 121.0 min, RTX PRO 6000 | 121.0 min, A10G | $7.65 |
| Base evaluation, 100 questions | 33.5 min, RTX PRO 6000 | 28.1 min, A10G | $2.05 |
| Merge and corrected export check | 13.2 min, CPU | — | $0.007 |
| Trained evaluation and unscored demonstrations | 12.8 min, RTX PRO 6000 | 11.3 min, A10G | $0.80 |
| **Experiment total** | | | **$10.50** |

Training continued from the intact step-10 checkpoint after a diagnostic process crashed after step 11.
The final state retains steps 1–50 and the optimizer/RNG state. The training row includes both process startups
and the discarded step. An initial mounted merge left a complete temporary weight file; it was recovered and
verified, then the corrected direct bucket export was checked separately on CPU. Both exports have identical
model weights. These costs are included above; the standalone scripts have no custom resume or recovery helper.

At the measured [HF Jobs prices](https://huggingface.co/docs/hub/jobs-pricing), the root cost $2.74998/hour,
the child $1.00002/hour plus $0.01/hour for its exposed port, and CPU $0.03/hour, rounded up per job minute.
The full development session, including earlier smoke tests, failed attempts, and CPU validation, cost an estimated
**$17.06**. These are runtime-based estimates, not an invoice; bucket storage and the account subscription are
additional. The [cost artifact](https://github.com/openenv/OpenEnv/blob/main/examples/grpo_repl_hotpotqa/results/compute.json)
contains the unrounded phase values and accounting method.

The recipe uses `kernels==0.18.2` and `use_kernels=True` for FLA and causal-convolution kernels. This version
satisfies Transformers 5.24.1's loader constraint; optional normalization layers retain their reference path.
Generated trajectory lengths and child calls affect runtime, so the timeout caps in the commands are not cost
predictions.

You need a Hugging Face account with a payment method for Jobs, and [uv](https://docs.astral.sh/uv/). The scripts pin
their direct TRL, vLLM, Transformers, and OpenEnv versions (`trl[vllm,peft]==1.18.0`, `vllm==0.29.0`,
`transformers==5.24.1`, `openenv==0.14.0`). The REPL package uses
commit `3659dc73eb8114724209e642eee749cd94348ce3`, which includes isolated code execution absent from the v0.14.0
tag. Transitive dependencies still resolve at installation time. The GPU training recipe targets Linux/CUDA;
Apple Silicon inference does not validate that backend.

## 1. Play the environment

Each HotpotQA question comes with ten Wikipedia paragraphs: the two that hold the answer and eight distractors. The
environment loads them into the REPL as the `context` variable and gives the root one tool, `execute_code`. The root
searches the library with Python, delegates evidence checks to the child, and finishes with `print(FINAL(answer))`.
The reward is 1.0 for a matching answer, 0.0 for a wrong one, and -0.1 when the 30 iterations run out without an
answer.

From `examples/grpo_repl_hotpotqa`, with the child server running (step 2 below), you can play one question by hand:

```python
from grpo import HotpotQAReplEnv, load_questions

env = HotpotQAReplEnv()
question = load_questions("validation")[0]
print(env.reset(**question))
print(env.execute_code("print(context[:400])"))
print(env.execute_code("print(FINAL('Paris'))"))
print(env.get_reward())
env._close()
```

The system prompt tells the root how to work: search `context` with string operations, use `llm_query(question)` for a
quick check over a short passage, and use `rlm_query(question, blocking=False)` plus `rlm_wait(handle, timeout_s=30)`
for a deeper child run that keeps working in the background while the root keeps searching. A child sees only the
question it is given, so the root has to include the evidence it needs.

## 2. Serve the child model separately

The child model runs on its own vLLM server, so the root's vLLM stays clean and the child can be scaled or swapped
independently. Start it before training or evaluation (one GPU is enough for the 0.6B child; on the same GPU as the
root, keep the memory fraction small):

```bash
vllm serve Qwen/Qwen3-0.6B --host 127.0.0.1 --port 8001 --gpu-memory-utilization 0.15 --max-model-len 8192
```

On HF Jobs, a training job has no second terminal, so the child runs as its own Job with an exposed port. Start it
first, with a timeout that covers the training run (the endpoint disappears when the job ends):

```bash
hf jobs run --detach --expose 8000 --flavor a10g-small --timeout 8h -s HF_TOKEN \
    vllm/vllm-openai:v0.29.0 \
    vllm serve Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.15 --max-model-len 8192
```

Wait for "Application startup complete" in its logs (`hf jobs logs -f <job_id>`), then point the training job at it
with `--child-url https://<job_id>--8000.hf.jobs/v1`. The scripts send your HF token (forwarded into the training
job with `--secrets HF_TOKEN`) as the Bearer token for Jobs-hosted child servers. Cancel the child job when you are
done.

The training script points at it with `--child-url` (default `http://127.0.0.1:8001/v1`) and `--child-model`
(default `Qwen/Qwen3-0.6B`). The child runs are managed: `create_server_recursive_controller` caps them at depth 2,
10 iterations per child, 2 batch workers, 4 children in total, and 60 seconds per child, so a runaway episode cannot
spawn unbounded workers.

Child HTTP requests time out after 60 seconds with no automatic retries. Training saves a checkpoint every five
optimizer steps, retaining the latest two in addition to the final adapter and trainer state.

## 3. GRPO

`grpo.py` trains the base model against the environment with TRL's `GRPOTrainer` and `environment_factory`. The
environment class exposes one tool, `execute_code`. The GRPO reward is the exact-match outcome term of the
environment's rubric (SQuAD normalization: lowercase, no punctuation, no articles); the code-execution process term
shapes the per-step REPL feedback but does not enter the GRPO reward.

There is no bonus for delegating, making more tool calls, or finishing faster. A correct answer found without a
child earns the same reward as one found with a child. Accuracy and delegation counts therefore answer different
questions: an accuracy gain alone would not demonstrate that the model learned recursive or parallel reasoning.
Guided asynchronous walkthroughs illustrate the API and are kept outside the scored evaluation.

```python
REPLRubric(
    outcome=CustomMetricRubric(squad_exact_match),
    process=CodeExecutionRubric(),
)
```

Each step plays 4 questions x 8 episodes (32 episodes), and runs 50 steps at a constant learning rate of 1e-4 with a
LoRA adapter (r=32, alpha=64, all-linear modules). The root generates with vLLM colocated, thinking off
(`chat_template_kwargs={"enable_thinking": False}`), and curves go to Trackio:

```bash
hf jobs uv run --python 3.12 --flavor rtx-pro-6000 --timeout 4h --secrets HF_TOKEN \
    --env TRACKIO_DIR=/outputs/trackio --env TRACKIO_STORAGE_MODE=jsonl \
    -v hf://buckets/<user>/hotpotqa-grpo:/outputs:rw grpo.py \
    --model Qwen/Qwen3.5-4B --output-dir /outputs/adapter \
    --child-url https://<job_id>--8000.hf.jobs/v1
```

Create the private bucket first, as shown in the [example README](https://github.com/openenv/OpenEnv/tree/main/examples/grpo_repl_hotpotqa#hf-jobs-with-persistent-outputs).
Job-local files disappear at job termination. The final adapter and trainer state are explicitly saved to the
output directory, including for short smoke runs. Jobs authentication is required for private storage and the
child endpoint. Model publication is a separate optional step after reviewing the evaluation results; the scripts
do not upload a model repository.
The Trackio variables persist JSONL logs in the mounted bucket. The saved `trainer_state.json` also contains the
actual metric history used by `analyze.py`.

## 4. Evaluate

`evaluate.py` plays a model greedily on held-out validation questions and saves every question as one JSON line:
`id`, `question`, `answer`, `submitted`, `exact_match`, `reward`, `turns`, `delegations`, tool-call counts,
and the full `conversation`. It also writes a summary JSON file. Exact-match accuracy counts correct submitted
answers; mean reward is separate because iteration-limit failures receive -0.1. Questions run in batches of four
with cleanup after each batch, including on errors. Run it on the base
model first, then on the trained model after merging the last LoRA checkpoint into the base model (vLLM serves merged
weights, not an adapter directory). The default is 100 questions:

```bash
uv run evaluate.py --model Qwen/Qwen3.5-4B --output base.jsonl

uv run merge.py --adapter ./adapter --output ./merged
uv run evaluate.py --model ./merged --output trained.jsonl
```

Both models receive the same reset observation and use the same training-compatible chat template, thinking
disabled, a 4096-token completion budget including tool feedback, and a 6144-token context limit. `delegations` counts recursive child runs, excluding direct `llm_query` calls. The JSONL question
IDs make the comparison auditable. On HF Jobs, mount the same private output bucket on training, merging, and
evaluation jobs; the README gives complete commands. Start with two training steps and five validation questions
per model before spending on the full run. Cancel the child job when finished and record its cost too.

## Managed nonblocking children

The following small synthetic check exercises the async API by hand. With the child server configured as above,
run each `execute_code` call separately. The REPL keeps the child handle while the root reads another passage:

```python
from grpo import HotpotQAReplEnv

env = HotpotQAReplEnv()
try:
    env.reset(
        context={
            "title": ["France", "Neighbor"],
            "sentences": [["Paris is the capital of France."], ["Belgium borders France."]],
        },
        answer="Paris",
        question="What is the capital of France?",
    )
    print(env.execute_code(
        'handle = rlm_query("Read this evidence: Paris is the capital of France. '
        'What is France\'s capital? Use one Python repl block and print(FINAL(answer)) '
        'to submit the single-word answer.", blocking=False); print(handle)'
    ))
    print(env.execute_code("print(context[context.find('=== Neighbor ==='):])"))
    print(env.execute_code(
        "child_answer = rlm_wait(handle, timeout_s=30); "
        "print(child_answer.replace('FINAL', 'child_result'))"
    ))
    print(env.execute_code("print(FINAL('Paris'))"))
finally:
    env._close()
```

Against the real Qwen3-0.6B CUDA child, the nonblocking call returned a handle in 6 ms, the separate passage read
took less than 1 ms, and the later wait returned `Paris` in 3.31 seconds. The final submitted answer earned 1.0.
These are observed API timings from one manual check, not a speedup benchmark or evidence of learned delegation.
A wait can time out while a child is still running; inspect its returned status before deciding whether to wait
again. Do not print raw child output containing `FINAL(...)`, because that text can end the root episode.
The trained model did not follow this async sequence even with explicit guidance; this walkthrough validates the API
mechanism by manual orchestration. The machine-readable trace is included with the [example results](https://github.com/openenv/OpenEnv/tree/main/examples/grpo_repl_hotpotqa/results).

## Going further

- Train longer, or with more questions per step, and track held-out accuracy as well as training reward beyond 50 steps.
- Swap the child for a larger model and see whether better evidence checks lift the root, or a smaller one and see
  what the root loses.
- Try the same recipe on another multi-hop dataset, or on a library the root has to build itself instead of receiving
  ready-made.
