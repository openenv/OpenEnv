# SPDX-License-Identifier: BSD-3-Clause

"""Exercise the HotpotQA examples with real REPL workers and scripted model replies."""

import importlib
import inspect
import json
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "grpo_repl_hotpotqa"
QUESTION = {
    "id": "capital",
    "question": "What is the capital of France?",
    "answer": "Paris",
    "context": {"title": ["France"], "sentences": [["Paris is the capital."]]},
}


@pytest.fixture(params=["grpo", "evaluate"])
def example(request, monkeypatch):
    pytest.importorskip("smolagents")
    monkeypatch.syspath_prepend(str(EXAMPLES))
    monkeypatch.delitem(sys.modules, request.param, raising=False)
    module = importlib.import_module(request.param)
    monkeypatch.setattr(module.HotpotQAReplEnv, "_child_chat_fn", None)
    return module


@pytest.fixture
def child_server(monkeypatch):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            content = (
                "Paris"
                if payload["messages"][-1]["content"] == "quick check"
                else "```repl\nprint(FINAL('Paris'))\n```"
            )
            body = json.dumps(
                {
                    "id": "scripted",
                    "object": "chat.completion",
                    "created": 0,
                    "model": payload["model"],
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": content},
                        }
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("REPL_CHILD_URL", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv("REPL_CHILD_MODEL", "scripted-child")
    monkeypatch.setenv("REPL_CHILD_API_KEY", "test-key")
    try:
        yield requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_real_repl_rewards_reset_and_cleanup(example):
    env = example.HotpotQAReplEnv()
    try:
        env.reset(**QUESTION)
        assert "Paris is the capital" in env.execute_code("print(context)")
        env.execute_code("print(FINAL('the Paris.'))")
        assert env.submitted == "the Paris."
        assert env.get_reward() == 1.0
        assert "Episode finished" in env.execute_code("print(FINAL('wrong'))")
        env.reset(**QUESTION)
        assert env.submitted is None and env.get_reward() == 0.0
        env.execute_code("print(FINAL('London'))")
        assert env.get_reward() == 0.0
        env.reset(**QUESTION)
        for _ in range(30):
            env.execute_code("pass")
        assert env.get_reward() == -0.1
    finally:
        env._close()
    assert env._repl is None


def test_direct_and_managed_children(example, child_server):
    env = example.HotpotQAReplEnv()
    try:
        env.reset(**QUESTION)
        assert "Paris" in env.execute_code("print(llm_query('quick check'))")
        env.execute_code("handle = rlm_query('find the capital', blocking=False)")
        output = env.execute_code(
            "result = rlm_wait(handle, timeout_s=30); print(result)"
        )
        assert "Paris" in output
        assert env.delegations == 1
        env.execute_code("print(FINAL('Paris'))")
        assert env.get_reward() == 1.0
    finally:
        env._close()
    assert len(child_server) == 2
    assert all(request["model"] == "scripted-child" for request in child_server)


def test_guided_demo_does_not_modify_evaluation_prompt(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    evaluator = importlib.import_module("evaluate")
    question = {
        **QUESTION,
        "prompt": [{"role": "user", "content": QUESTION["question"]}],
    }
    seen = []

    def batch(tokenizer, llm, questions):
        seen.extend(questions)
        return [{"id": item["id"], "exact_match": False} for item in questions]

    monkeypatch.setattr(evaluator, "evaluate_batch", batch)
    output = tmp_path / "demonstration.jsonl"
    evaluator.demonstrate(None, None, [question], output)
    assert question["prompt"][-1]["content"] == QUESTION["question"]
    assert "blocking=False" in seen[0]["prompt"][-1]["content"]
    record = json.loads(output.read_text())
    assert record["guided_demonstration"] is True
    assert record["included_in_evaluation"] is False


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://job--8000.hf.jobs/v1", "hf-test"),
        ("https://hf.jobs.attacker.invalid/v1", "unused"),
        ("https://example.invalid/path/.hf.jobs", "unused"),
    ],
)
def test_hf_token_only_sent_to_jobs_host(example, monkeypatch, url, expected):
    clients = []
    monkeypatch.setenv("REPL_CHILD_URL", url)
    monkeypatch.setenv("HF_TOKEN", "hf-test")
    monkeypatch.delenv("REPL_CHILD_API_KEY", raising=False)
    monkeypatch.setattr(example, "OpenAI", lambda **kwargs: clients.append(kwargs))
    example.make_child_chat()
    assert clients[0]["api_key"] == expected
    assert clients[0]["timeout"] == 60.0
    assert clients[0]["max_retries"] == 0


@pytest.fixture
def evaluator(monkeypatch):
    pytest.importorskip("smolagents")
    monkeypatch.syspath_prepend(str(EXAMPLES))
    monkeypatch.delitem(sys.modules, "evaluate", raising=False)
    module = importlib.import_module("evaluate")
    instances = []
    env_type = module.HotpotQAReplEnv

    def factory():
        env = env_type()
        instances.append(env)
        return env

    monkeypatch.setattr(module, "HotpotQAReplEnv", factory)
    parsing = ModuleType("trl.chat_template_utils")
    parsing.get_training_chat_template = lambda tokenizer: None
    parsing.is_chat_template_prefix_preserving = lambda tokenizer: True
    parsing.parse_response = lambda tokenizer, ids, **kwargs: json.loads(bytes(ids))
    monkeypatch.setitem(sys.modules, "trl.chat_template_utils", parsing)
    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(SamplingParams=SimpleNamespace)
    )
    return module, instances


class Tokenizer:
    def apply_chat_template(self, conversation, **kwargs):
        assert kwargs["enable_thinking"] is False
        assert "The REPL is ready" in conversation[1]["content"]
        return [1] * (len(json.dumps(conversation)) // 10)

    def decode(self, ids):
        return bytes(ids).decode()


def tool_reply(name, arguments):
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": name, "arguments": arguments}}],
    }
    return SimpleNamespace(
        outputs=[SimpleNamespace(token_ids=list(json.dumps(message).encode()))]
    )


def test_evaluation_keeps_controls_private_and_reports_accuracy(evaluator):
    module, instances = evaluator
    replies = iter(
        [
            tool_reply("reset", QUESTION),
            tool_reply("execute_code", {"code": "print(FINAL('Paris'))"}),
        ]
    )

    def generate(prompts, params):
        assert 0 < params[0].max_tokens <= 4096
        return [next(replies)]

    question = {
        **QUESTION,
        "prompt": [
            {"role": "system", "content": "test"},
            {"role": "user", "content": QUESTION["question"]},
        ],
    }
    records = module.evaluate_batch(
        Tokenizer(), SimpleNamespace(generate=generate), [question]
    )
    record = records[0]
    assert record["exact_match"] is True and record["reward"] == 1.0
    assert "Tool reset not found" in record["conversation"][3]["content"]
    assert record["turns"] == 2
    assert all(env._repl is None for env in instances)


@pytest.mark.parametrize("bucket_output", [False, True])
def test_merge_preserves_multimodal_processor(monkeypatch, tmp_path, bucket_output):
    pytest.importorskip("torchvision")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    monkeypatch.syspath_prepend(str(EXAMPLES))
    module = importlib.import_module("merge")
    adapter, output = tmp_path / "adapter", tmp_path / "merged"
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizers.Tokenizer(
            tokenizers.models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")
        ),
        unk_token="[UNK]",
    )
    processor = transformers.Qwen3VLProcessor(
        image_processor=transformers.Qwen2VLImageProcessor(merge_size=3),
        video_processor=transformers.Qwen3VLVideoProcessor(),
        tokenizer=tokenizer,
    )
    processor.save_pretrained(adapter)

    class Model:
        def merge_and_unload(self):
            return self

        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(
            AutoPeftModelForCausalLM=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: Model()
            )
        ),
    )
    destination = (
        "hf://buckets/example/artifacts/merged" if bucket_output else str(output)
    )

    def sync_bucket(self, source, dest):
        assert dest == destination
        shutil.copytree(source, output)

    monkeypatch.setattr("huggingface_hub.HfApi.sync_bucket", sync_bucket)
    monkeypatch.setattr(
        sys, "argv", ["merge.py", "--adapter", str(adapter), "--output", destination]
    )
    module.main()
    restored = transformers.AutoProcessor.from_pretrained(output)
    assert isinstance(restored, transformers.Qwen3VLProcessor)
    assert restored.image_processor.merge_size == 3


def test_evaluation_closes_workers_on_generation_failure(evaluator):
    module, instances = evaluator

    def generate(*args):
        raise RuntimeError("generation failed")

    question = {
        **QUESTION,
        "prompt": [
            {"role": "system", "content": "test"},
            {"role": "user", "content": QUESTION["question"]},
        ],
    }
    with pytest.raises(RuntimeError, match="generation failed"):
        module.evaluate_batch(
            Tokenizer(), SimpleNamespace(generate=generate), [question]
        )
    assert all(env._repl is None for env in instances)


def test_iteration_penalty_is_not_negative_accuracy(evaluator):
    module, instances = evaluator

    def generate(prompts, params):
        return [tool_reply("execute_code", {"code": "pass"}) for _ in prompts]

    question = {
        **QUESTION,
        "prompt": [
            {"role": "system", "content": "test"},
            {"role": "user", "content": QUESTION["question"]},
        ],
    }
    record = module.evaluate_batch(
        Tokenizer(), SimpleNamespace(generate=generate), [question]
    )[0]
    assert record["reward"] == -0.1
    assert record["exact_match"] is False
    assert record["turns"] == 30
    assert all(env._repl is None for env in instances)


@pytest.mark.parametrize("fail", [False, True])
def test_training_saves_final_adapter_and_always_closes(monkeypatch, tmp_path, fail):
    pytest.importorskip("smolagents")
    monkeypatch.syspath_prepend(str(EXAMPLES))
    module = importlib.import_module("grpo")
    instances, saved = [], []

    class Trainer:
        def __init__(self, **kwargs):
            assert kwargs["args"].push_to_hub is False
            env = kwargs["environment_factory"]()
            instances.append(env)
            # TRL discovers tools on the class after main configures the child callback.
            tools = {
                name
                for name, _ in inspect.getmembers(type(env), inspect.isfunction)
                if not name.startswith("_") and name not in {"reset", "get_reward"}
            }
            assert tools == {"execute_code"}
            env.reset(**QUESTION)

        def train(self):
            if fail:
                raise RuntimeError("training failed")

        def save_model(self, output):
            saved.append(output)

        def save_state(self):
            saved.append("state")

    monkeypatch.setitem(
        sys.modules,
        "trl",
        SimpleNamespace(GRPOConfig=SimpleNamespace, GRPOTrainer=Trainer),
    )
    monkeypatch.setitem(
        sys.modules, "peft", SimpleNamespace(LoraConfig=SimpleNamespace)
    )
    monkeypatch.setattr(module, "load_questions", lambda split: [QUESTION])
    monkeypatch.setattr(
        sys, "argv", ["grpo.py", "--max-steps", "2", "--output-dir", str(tmp_path)]
    )
    if fail:
        with pytest.raises(RuntimeError, match="training failed"):
            module.main()
        assert not saved
    else:
        module.main()
        assert saved == [str(tmp_path), "state"]
    assert all(env._repl is None for env in instances)
