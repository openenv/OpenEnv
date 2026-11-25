# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "trl[vllm]==1.18.0",
#     "vllm==0.29.0",
#     "transformers==5.24.1",
#     "openenv==0.14.0",
#     "openenv-repl @ git+https://github.com/openenv/OpenEnv.git@3659dc73eb8114724209e642eee749cd94348ce3#subdirectory=envs/repl_env",
#     "datasets",
#     "openai",
# ]
# ///
"""
Exact-match accuracy of a model playing the HotpotQA library game in OpenEnv's `repl_env`, on held-out validation
questions, with greedy decoding and training-aligned prompts and token limits. Tool calls are parsed with TRL's response schema
for the model's chat template, so a malformed call counts against the model, as it does in training.

The child model is served separately, on its own vLLM server. Start it first:

```sh
vllm serve Qwen/Qwen3-0.6B --host 127.0.0.1 --port 8001 --gpu-memory-utilization 0.15 --max-model-len 8192
```

On HF Jobs, serve the child as its own Job with an exposed port and pass `--child-url` (see README.md); the script
sends your HF token as the Bearer token for Jobs-hosted child servers.

Run it on the base model first, then on the trained model after merging its last LoRA checkpoint into the base model
(vLLM serves merged weights, not an adapter directory). See README.md.

    python evaluate.py --model Qwen/Qwen3.5-4B --output base.jsonl
"""

import argparse
import json
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
# imports (transformers, vllm, datasets, trl) live inside main() so a worker starts in seconds instead of
# paying for the whole eval stack on every child query.

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
    token: set REPL_CHILD_API_KEY, or let it fall back to HF_TOKEN (forwarded into the job with --secrets HF_TOKEN)
    for .hf.jobs URLs only.
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
    child and returns a handle, `rlm_wait` collects it.
    """

    # Set in main() before evaluation: the shared chat function for the child server, built by make_child_chat().
    # Shared child callback, kept private to match the training environment's tool surface.
    _child_chat_fn = None

    def __init__(self):
        self._repl = None
        self._reward = 0.0
        self._done = False
        self.submitted = None
        self.delegations = 0

    def reset(self, context, answer, question=None, **kwargs) -> str | None:
        # Close the previous REPL before building a new one: each one owns daemon threads, and leaving them
        # behind leaks one thread per episode.
        self._close()
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
        return (
            "\n\nThe REPL is ready. `context` holds 10 Wikipedia articles "
            f"({len(library)} chars). Use the execute_code tool to search it."
        )

    def _count_delegation(self, depth, model, prompt_preview):
        # Called (from the managed-children notification thread) once per spawned child run.
        self.delegations += 1

    def _close(self):
        # Private so cleanup cannot become an agent tool.
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
        return float(self._reward)


def evaluate_batch(tokenizer, llm, questions, max_turns=30):
    """Play a bounded batch, keeping the prompt and token budgets aligned with training."""
    from trl.chat_template_utils import (
        get_training_chat_template,
        is_chat_template_prefix_preserving,
        parse_response,
    )
    from vllm import SamplingParams

    template = (
        None
        if is_chat_template_prefix_preserving(tokenizer)
        else get_training_chat_template(tokenizer)
    )
    environments = [HotpotQAReplEnv() for _ in questions]
    conversations = []
    turns = [0] * len(questions)
    emitted = [0] * len(questions)
    parsed = [0] * len(questions)
    try:
        for environment, question in zip(environments, questions, strict=True):
            observation = environment.reset(**question)
            conversation = [dict(message) for message in question["prompt"]]
            conversation[-1]["content"] += observation
            conversations.append(conversation)
        tools = [environments[0].execute_code]

        def tokenize(conversation):
            return tokenizer.apply_chat_template(
                conversation,
                tools=tools,
                chat_template=template,
                enable_thinking=False,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
            )

        initial_lengths = [
            len(tokenize(conversation)) for conversation in conversations
        ]
        active = list(range(len(questions)))
        for _ in range(max_turns):
            prompts, sampling_params, ready = [], [], []
            for i in active:
                prompt = tokenize(conversations[i])
                # TRL counts generated tokens and tool feedback in the 4096-token completion budget.
                remaining = min(
                    4096 - len(prompt) + initial_lengths[i], 6144 - len(prompt)
                )
                if remaining <= 0:
                    continue
                ready.append(i)
                prompts.append(prompt)
                sampling_params.append(
                    SamplingParams(temperature=0.0, max_tokens=remaining)
                )
            if not ready:
                break
            outputs = llm.generate(
                [{"prompt_token_ids": prompt} for prompt in prompts], sampling_params
            )
            active = []
            for i, prompt, output in zip(ready, prompts, outputs, strict=True):
                completion_ids = list(output.outputs[0].token_ids)
                message = parse_response(
                    tokenizer, completion_ids, prefix=prompt, tools=tools
                )
                conversations[i].append(message)
                turns[i] += 1
                emitted[i] += tokenizer.decode(completion_ids).count("<tool_call>")
                calls = message.get("tool_calls") or []
                parsed[i] += len(calls)
                for call in calls:
                    name = call["function"]["name"]
                    try:
                        if name != "execute_code":
                            raise ValueError(f"Tool {name} not found.")
                        result = environments[i].execute_code(
                            **call["function"]["arguments"]
                        )
                    except Exception as error:
                        result = {"error": str(error)}
                    conversations[i].append(
                        {"role": "tool", "name": name, "content": str(result)}
                    )
                if calls and not environments[i]._done:
                    active.append(i)
        return [
            {
                "id": question["id"],
                "question": question["question"],
                "answer": question["answer"],
                "submitted": environment.submitted,
                "exact_match": environment.submitted is not None
                and bool(squad_exact_match(question["answer"], environment.submitted)),
                "reward": environment.get_reward(),
                "turns": num_turns,
                "delegations": environment.delegations,
                "emitted_tool_calls": num_emitted,
                "parsed_tool_calls": num_parsed,
                "conversation": conversation,
            }
            for question, conversation, environment, num_turns, num_emitted, num_parsed in zip(
                questions,
                conversations,
                environments,
                turns,
                emitted,
                parsed,
                strict=True,
            )
        ]
    finally:
        for environment in environments:
            environment._close()


def demonstrate(tokenizer, llm, questions, output_path):
    """Record guided asynchronous rollouts separately from scored evaluation."""
    instruction = (
        "\n\nThis is a guided tutorial demonstration, separate from scored evaluation. "
        "The context variable is already populated: do not overwrite it or assign an empty string. "
        "Complete the following stages in separate execute_code calls; do not combine the query and wait. "
        "First inspect context for relevant evidence. Then use "
        "rlm_query(evidence_question, blocking=False), including a short evidence passage "
        "in evidence_question, and store its handle. In a separate execute_code call, "
        "search another part of context while that child works. In a later call, collect "
        "the answer with rlm_wait(handle, timeout_s=30). The answer is a string: store it "
        "and replace the text 'FINAL' before printing it, to avoid ending the root episode. "
        "Finally use the evidence to submit your own short answer with print(FINAL(answer))."
    )
    with open(output_path, "w") as output:
        for question in questions:
            question = {
                **question,
                "prompt": [dict(message) for message in question["prompt"]],
            }
            question["prompt"][-1]["content"] += instruction
            record = evaluate_batch(tokenizer, llm, [question])[0]
            record["guided_demonstration"] = True
            record["included_in_evaluation"] = False
            output.write(json.dumps(record) + "\n")
            output.flush()


def main():
    # Greedy evaluation does not need FlashInfer's top-k/top-p sampler or its nvcc warmup.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from transformers import AutoTokenizer
    from trl.chat_template_utils import add_response_schema
    from vllm import LLM

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument(
        "--child-url",
        default="http://127.0.0.1:8001/v1",
        help="OpenAI-compatible URL of the vLLM server for the frozen child model.",
    )
    parser.add_argument("--child-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num-questions", "--num_questions", type=int, default=100)
    parser.add_argument("--max-turns", "--max_turns", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", default="eval.jsonl")
    parser.add_argument(
        "--demo-output",
        help="Optional JSONL for three guided async demonstrations on separate questions.",
    )
    args = parser.parse_args()
    if args.num_questions <= 0 or args.batch_size <= 0 or not 1 <= args.max_turns <= 30:
        parser.error(
            "question count and batch size must be positive; max turns must be in [1, 30]"
        )
    if args.demo_output and os.path.realpath(args.demo_output) in {
        os.path.realpath(args.output),
        os.path.realpath(args.output + ".summary.json"),
    }:
        parser.error(
            "demonstration output must differ from evaluation output and summary"
        )

    os.environ["REPL_CHILD_URL"] = args.child_url
    os.environ["REPL_CHILD_MODEL"] = args.child_model
    HotpotQAReplEnv._child_chat_fn = make_child_chat()
    tokenizer = add_response_schema(AutoTokenizer.from_pretrained(args.model))
    validation = load_questions("validation")
    questions = validation.select(range(args.num_questions))
    llm = LLM(
        model=args.model,
        max_model_len=6144,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True,
        max_num_seqs=args.batch_size,
    )
    correct = reward_sum = delegations = emitted = parsed = 0
    with open(args.output, "w") as output:
        for start in range(0, len(questions), args.batch_size):
            batch = list(
                questions.select(
                    range(start, min(start + args.batch_size, len(questions)))
                )
            )
            for record in evaluate_batch(tokenizer, llm, batch, args.max_turns):
                output.write(json.dumps(record) + "\n")
                correct += record["exact_match"]
                reward_sum += record["reward"]
                delegations += record["delegations"]
                emitted += record["emitted_tool_calls"]
                parsed += record["parsed_tool_calls"]
            output.flush()
    summary = {
        "model": args.model,
        "child_model": args.child_model,
        "num_questions": len(questions),
        "max_turns": args.max_turns,
        "enable_thinking": False,
        "max_completion_length": 4096,
        "max_model_length": 6144,
        "correct": correct,
        "exact_match": correct / len(questions),
        "mean_reward": reward_sum / len(questions),
        "mean_delegations": delegations / len(questions),
        "emitted_tool_calls": emitted,
        "parsed_tool_calls": parsed,
    }
    with open(args.output + ".summary.json", "w") as output:
        json.dump(summary, output, indent=2)
    print(json.dumps(summary, indent=2))
    if args.demo_output:
        demonstrations = validation.select(
            range(args.num_questions, min(args.num_questions + 3, len(validation)))
        )
        demonstrate(tokenizer, llm, list(demonstrations), args.demo_output)


if __name__ == "__main__":
    main()
