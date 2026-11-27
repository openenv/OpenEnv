"""Use separate models for the parent RLM loop and child queries."""

import argparse
import os

from openai import OpenAI
from repl_env import LocalRLMRunner
from repl_env.recursive_backends import BackendLimits, LocalChildRLMBackend
from vllm_runner import make_chat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-url", required=True)
    parser.add_argument("--parent-model", required=True)
    parser.add_argument("--child-url", required=True)
    parser.add_argument("--child-model", required=True)
    args = parser.parse_args()

    with (
        OpenAI(
            base_url=args.parent_url,
            api_key=os.environ.get("REPL_PARENT_API_KEY", "unused"),
        ) as parent_client,
        OpenAI(
            base_url=args.child_url,
            api_key=os.environ.get("REPL_CHILD_API_KEY", "unused"),
        ) as child_client,
    ):
        parent_chat = make_chat(parent_client, args.parent_model)
        child_chat = make_chat(child_client, args.child_model)

        def backend_factory(
            _parent_chat,
            *,
            system_prompt,
            max_iterations,
            max_depth,
            depth,
            env_max_iterations_multiplier,
        ):
            return LocalChildRLMBackend(
                child_chat,
                runner_factory=LocalRLMRunner,
                system_prompt=system_prompt,
                max_iterations=max_iterations,
                env_max_iterations_multiplier=env_max_iterations_multiplier,
                depth=depth,
                limits=BackendLimits(max_depth=max_depth),
            )

        runner = LocalRLMRunner(
            parent_chat, backend_factory=backend_factory, max_iterations=10, max_depth=2
        )
        result = runner.run(
            "alpha beta gamma\ndelta epsilon",
            "Use rlm_query to count the words in each line, then add the counts. "
            "Return the total as an integer.",
        )
        print(result.final_answer)


if __name__ == "__main__":
    main()
