# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "trl[vllm,peft]==1.18.0",
#     "vllm==0.29.0",
#     "transformers==5.24.1",
#     "kernels==0.18.2",
#     "trackio",
#     "openenv==0.14.0",
#     "openenv-repl @ git+https://github.com/openenv/OpenEnv.git@3659dc73eb8114724209e642eee749cd94348ce3#subdirectory=envs/repl_env",
#     "datasets",
#     "openai",
# ]
# ///

"""
GRPO on OpenEnv's `repl_env`: train a recursive language model (RLM) on HotpotQA.

The root model answers multi-hop questions from excerpts of ten Wikipedia articles per question. Its initial prompt
omits the library: the articles live in the REPL's `context` variable, and the root reads them through one
tool, `execute_code`, which runs Python in a `LocalREPLEnv`. Inside the REPL it can search the library with code and
hand evidence checks to a frozen child model with `llm_query` or `rlm_query`. Only the root's LoRA adapter trains;
the child stays frozen. Correct submitted answers earn 1.0, wrong answers 0.0, and exhausting 30 REPL iterations earns -0.1.

Each step plays 4 questions with 8 episodes per question (32 episodes). Training curves go to Trackio.

The child model is served separately, on its own vLLM server. Locally, start it first (one GPU is enough for the
0.6B child; on the same GPU as the root, keep the memory fraction small):

```sh
vllm serve Qwen/Qwen3-0.6B --host 127.0.0.1 --port 8001 --gpu-memory-utilization 0.15 --max-model-len 8192
```

Then train locally, with vLLM colocated for the root:

```sh
python grpo.py --model Qwen/Qwen3.5-4B
```

On HF Jobs a training job has no second terminal, so the child runs as its own Job with an exposed port. Start it
first, with a timeout that covers the training run (the endpoint disappears when the job ends):

```sh
hf jobs run --detach --expose 8000 --flavor a10g-small --timeout 8h -s HF_TOKEN \
    vllm/vllm-openai:v0.29.0 \
    vllm serve Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.15 --max-model-len 8192
```

Wait for "Application startup complete" in its logs (`hf jobs logs -f <job_id>`), then train, pointing `--child-url`
at the exposed port. The script sends your HF token (forwarded into the training job with `--secrets HF_TOKEN`) as
the Bearer token for Jobs-hosted child servers:

```sh
hf jobs uv run --python 3.12 --flavor rtx-pro-6000 --timeout 4h --secrets HF_TOKEN \
    --env TRACKIO_DIR=/outputs/trackio --env TRACKIO_STORAGE_MODE=jsonl \
    -v hf://buckets/<user>/hotpotqa-grpo:/outputs:rw grpo.py \
    --model Qwen/Qwen3.5-4B --output-dir /outputs/adapter \
    --child-url https://<job_id>--8000.hf.jobs/v1
```

Cancel the child job when training is done.

Or with a separate TRL vLLM server for the root: reserve GPU 1 for the server and run training on GPU 0
with `CUDA_VISIBLE_DEVICES=0 python grpo.py --vllm-mode server`. Serve the child separately as above:

```sh
CUDA_VISIBLE_DEVICES=1 trl vllm-serve --model Qwen/Qwen3.5-4B --port 8000
```

Training saves the adapter without publishing it. Publication is a separate step after evaluation. HF Jobs
needs authentication for private storage and the child endpoint. Mount a private bucket for persistent
checkpoints; see README.md.
"""

import argparse
import os
import string
from urllib.parse import urlparse

from openai import OpenAI
from repl_env import LocalREPLEnv
from repl_env.managed_children import ManagedChildConfig
from repl_env.recursive_controller import create_server_recursive_controller
from repl_env.rubrics import CodeExecutionRubric, CustomMetricRubric, REPLRubric


# NOTE: keep this module's top-level imports light (stdlib, openai, repl_env). Managed children spawn worker
# processes with multiprocessing "spawn", and each worker re-imports this module as __mp_main__. The heavy
# imports (transformers, datasets, trl) live inside main() so a worker starts in seconds instead of
# paying for the whole training stack on every child query.

SYSTEM_PROMPT = """You are a recursive language model answering a multi-hop question. You have one tool, execute_code, which runs Python code in a REPL. The REPL holds a `context` variable: the text of ten Wikipedia articles, two of which hold the answer. The question is in the user message.

Search `context` with Python code: use string search and slicing to find the articles that matter, then follow what one article tells you to the next. When a fact needs checking, ask a child model from inside the REPL. `llm_query(question)` makes one quick call over a short passage. `rlm_query(question, blocking=False)` starts a deeper child run and returns a handle so you can keep working; collect its answer later with `rlm_wait(handle, timeout_s=30)`. These functions are already available in the REPL; do not import them from another module. A child sees only the question you give it, so include the evidence it needs. Only use the blocking `rlm_query(question)` when you genuinely cannot proceed without that answer. Do not print a child model's raw output: if it contains the text FINAL(...), the episode ends on that text.

When you are sure of the answer, run one last line of code: print(FINAL(<short answer>)). The answer is a name, a date, a number, or yes/no. Do not call any tools after that.
"""


def make_child_chat():
    """Build the chat function the child model runs on.

    Reads the child server from the environment so the managed-child workers, which are spawned processes that
    re-import this module, can build their own client: REPL_CHILD_URL (default http://127.0.0.1:8001/v1) and
    REPL_CHILD_MODEL (default Qwen/Qwen3-0.6B). Matches `make_chat` in examples/repl_model_servers. A child served
    by a second HF Job exposes its port at https://<job_id>--<port>.hf.jobs and requires the HF token as the Bearer
    token: set REPL_CHILD_API_KEY, or let it fall back to HF_TOKEN (forwarded into the training job with
    --secrets HF_TOKEN) for .hf.jobs URLs only.
    """
    base_url = os.environ.get("REPL_CHILD_URL", "http://127.0.0.1:8001/v1")
    api_key = os.environ.get("REPL_CHILD_API_KEY")
    if api_key is None and (urlparse(base_url).hostname or "").endswith(".hf.jobs"):
        api_key = os.environ.get("HF_TOKEN", "unused")
    client = OpenAI(
        base_url=base_url,
        api_key=api_key or "unused",
        timeout=60.0,
        max_retries=0,
    )
    default_model = os.environ.get("REPL_CHILD_MODEL", "Qwen/Qwen3-0.6B")

    def chat(messages, model=None):
        response = client.chat.completions.create(
            model=model or default_model,
            messages=messages,
            max_tokens=2048,
            temperature=0,
        )
        return response.choices[0].message.content or ""

    return chat


def normalize_answer(text):
    # SQuAD answer normalization: lowercase, no punctuation, no articles, single spaces
    text = "".join(char for char in text.lower() if char not in string.punctuation)
    return " ".join(word for word in text.split() if word not in ("a", "an", "the"))


def squad_exact_match(expected, predicted):
    return 1.0 if normalize_answer(expected) == normalize_answer(predicted) else 0.0


def format_library(context):
    # HotpotQA distractor context: {"title": [...], "sentences": [[...], ...]}
    parts = []
    for title, sentences in zip(context["title"], context["sentences"], strict=True):
        parts.append(f"=== {title} ===\n" + " ".join(sentences).strip())
    return "\n\n".join(parts)


def load_questions(split):
    from datasets import load_dataset

    # Each HotpotQA question comes with ten Wikipedia paragraphs: the two that hold the answer and eight distractors.
    # The prompt carries the question; the library and the answer stay in the row so reset() can load them.
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    return dataset.map(
        lambda example: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Question: " + example["question"]},
            ]
        },
        remove_columns=["type", "level", "supporting_facts"],
    )


class HotpotQAReplEnv:
    """A HotpotQA question played in OpenEnv's `repl_env` by a recursive language model.

    The root model gets one tool, `execute_code`, which runs Python in a `LocalREPLEnv`. The library is the REPL's
    `context` variable; the root searches it with code and delegates evidence checks to a frozen child model served
    on a separate vLLM server. The child runs are managed: `rlm_query(..., blocking=False)` starts a background
    child and returns a handle, `rlm_wait` collects it. Only the root's LoRA adapter trains.
    """

    # Set in main(); keep private because TRL exposes public functions as model tools.
    _child_chat_fn = None

    def __init__(self):
        self._repl = None
        self._reward = 0.0
        self._done = False
        self.submitted = None
        self.delegations = 0

    def reset(self, context, answer, question=None, **kwargs) -> str | None:
        # TRL pools and reuses environment instances, so close the previous REPL before building a new one: each
        # one owns daemon threads, and leaving them behind leaks one thread per episode.
        self._close()
        # A fresh REPL per episode: new executor worker, fresh namespace, managed children reset. The controller
        # owns the child limits, so a runaway episode cannot spawn unbounded workers.
        chat_fn = type(self)._child_chat_fn or make_child_chat()
        controller = create_server_recursive_controller(
            chat_fn,
            max_depth=2,
            max_iterations=10,
            max_batch_workers=2,
            max_children_total=4,
            per_child_timeout_s=60,
            result_truncation_limit=2000,
            managed_children=ManagedChildConfig(chat_factory=make_child_chat),
            on_subcall_start=self._count_delegation,
        )
        self._repl = LocalREPLEnv(
            recursive_controller=controller,
            rubric=REPLRubric(
                outcome=CustomMetricRubric(squad_exact_match),
                process=CodeExecutionRubric(),
            ),
        )
        self._reward = 0.0
        self._done = False
        self.submitted = None
        self.delegations = 0
        library = format_library(context)
        self._repl.reset(
            context=library,
            task_prompt=f"Answer this question: {question}" if question else "",
            max_iterations=30,
            expected_answer=answer,
        )
        # The trainer appends this to the prompt, so keep it short: the library lives in the REPL, not the prompt.
        return (
            "\n\nThe REPL is ready. `context` holds 10 Wikipedia articles "
            f"({len(library)} chars). Use the execute_code tool to search it."
        )

    def _count_delegation(self, depth, model, prompt_preview):
        # Called (from the managed-children notification thread) once per spawned child run.
        self.delegations += 1

    def _close(self):
        # Private so TRL does not expose cleanup as an agent tool.
        if self._repl is not None:
            self._repl.close()
            self._repl = None

    def execute_code(self, code: str) -> str:
        """Run Python code in the REPL.

        Args:
            code (`str`):
                Python code to run. `context` holds the ten Wikipedia articles; use `llm_query(question)` for a
                quick check over a short passage and `rlm_query(question, blocking=False)` plus
                `rlm_wait(handle, timeout_s=30)` to delegate a deeper check to a child model while you keep
                working. Finish with `print(FINAL(answer))` where the answer is a short name, a date, a number,
                or yes/no.

        Returns:
            `str`: The REPL output, or a note that the episode is finished.
        """
        if self._done:
            return "Episode finished. Do not call any more tools."
        result = self._repl.execute(code)
        if result.done:
            self._done = True
            self._reward = result.reward
            final = self._repl.state().final_answer
            self.submitted = final
            return (
                f"Episode complete. Final answer: {final}. Do not call any more tools."
            )
        output = result.observation.result.stdout or "(no output)"
        if not result.observation.result.success:
            error = (
                result.observation.result.stderr
                or result.observation.result.exception
                or "Unknown error"
            )
            output = f"{output}\n\nERROR: {error}"
        return output[:4000]

    def get_reward(self) -> float:
        # Latched on the step where the episode finished: 1.0 for a matching answer, 0.0 for a wrong one, -0.1
        # when iterations ran out without an answer, 0.0 when the trainer stopped the episode first.
        return float(self._reward)


def main():
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    parser = argparse.ArgumentParser(
        description="Run GRPO training of a recursive language model on HotpotQA, with OpenEnv's repl_env."
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3.5-4B", help="Root model to train."
    )
    parser.add_argument(
        "--child-url",
        type=str,
        default="http://127.0.0.1:8001/v1",
        help="OpenAI-compatible URL of the vLLM server for the frozen child model.",
    )
    parser.add_argument(
        "--child-model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Frozen child model for the REPL.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=50, help="Number of training steps."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to store outputs and checkpoints.",
    )
    parser.add_argument(
        "--vllm-mode",
        choices=("colocate", "server"),
        default="colocate",
        help="vLLM execution mode.",
    )
    parser.add_argument(
        "--vllm-server-url",
        type=str,
        default="http://localhost:8000",
        help="vLLM server URL (server mode only).",
    )
    args = parser.parse_args()

    # The managed-child workers are spawned processes that build their own client from these variables.
    os.environ["REPL_CHILD_URL"] = args.child_url
    os.environ["REPL_CHILD_MODEL"] = args.child_model
    HotpotQAReplEnv._child_chat_fn = make_child_chat()

    output_dir = args.output_dir or f"{args.model.split('/')[-1]}-GRPO-HotpotQA"

    environments = []

    def environment_factory():
        environment = HotpotQAReplEnv()
        environments.append(environment)
        return environment

    try:
        trainer = GRPOTrainer(
            model=args.model,
            train_dataset=load_questions("train"),
            args=GRPOConfig(
                output_dir=output_dir,
                model_init_kwargs={"dtype": "bfloat16", "use_kernels": True},
                # LoRA trains a small fraction of the parameters, so use a higher learning rate.
                learning_rate=args.learning_rate,
                lr_scheduler_type="constant",
                use_vllm=True,
                vllm_mode=args.vllm_mode,
                vllm_server_base_url=args.vllm_server_url
                if args.vllm_mode == "server"
                else None,
                vllm_gpu_memory_utilization=0.35,
                vllm_max_model_length=6144,
                report_to="trackio",
                log_completions=True,
                num_completions_to_print=1,
                logging_steps=1,
                max_steps=args.max_steps,
                # 4 questions x 8 episodes per step
                num_generations=8,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=8,
                gradient_checkpointing=True,
                bf16=True,
                max_completion_length=4096,
                mask_truncated_completions=True,
                vllm_importance_sampling_mode="token_truncate",
                max_tool_calling_iterations=30,
                save_steps=5,
                save_total_limit=2,
                push_to_hub=False,
                chat_template_kwargs={"enable_thinking": False},
            ),
            environment_factory=environment_factory,
            peft_config=LoraConfig(r=32, lora_alpha=64, target_modules="all-linear"),
        )
        trainer.train()
        trainer.save_model(output_dir)
        trainer.save_state()
    finally:
        for environment in environments:
            environment._close()


if __name__ == "__main__":
    main()
