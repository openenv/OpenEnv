"""Run a local RLM against a vLLM chat completion server."""

import argparse
import os

from openai import OpenAI
from repl_env import LocalRLMRunner


def make_chat(client: OpenAI, default_model: str):
    def chat(messages, model=None):
        response = client.chat.completions.create(
            model=model or default_model,
            messages=messages,
            max_tokens=2048,
            temperature=0,
        )
        return response.choices[0].message.content or ""

    return chat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    with OpenAI(
        base_url=args.base_url, api_key=os.environ.get("REPL_API_KEY", "unused")
    ) as client:
        runner = LocalRLMRunner(make_chat(client, args.model), max_iterations=10)
        result = runner.run("alpha beta gamma", "Count the words in context.")
        print(result.final_answer)


if __name__ == "__main__":
    main()
