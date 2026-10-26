# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
RLM system prompts and parsing utilities for the REPL environment.

Prompt structure is kept close to the official RLM implementation:
- OpenAI-style message lists in the core loop
- model/provider-specific formatting handled by the chat client
- explicit guidance around `llm_query` vs `rlm_query`
"""

import re
import textwrap
from dataclasses import dataclass
from typing import List, Optional


# =============================================================================
# Query Metadata (for context info)
# =============================================================================


@dataclass
class QueryMetadata:
    """Metadata about the context for building prompts."""

    context_lengths: List[int]
    context_total_length: int
    context_type: str = "str"  # "str" or "List[str]"


_RLM_SYSTEM_PROMPT_BASE = """You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. A `context` variable that contains extremely important information about your query. You should check the content of the `context` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. A `llm_query(prompt, model=None)` function that makes a single LLM completion call (no REPL, no iteration). Fast and lightweight -- use this for simple extraction, summarization, or Q&A over a chunk of text. The sub-LLM can handle around 500K chars.
3. A `llm_query_batched(prompts, model=None)` function that runs multiple `llm_query` calls concurrently: returns `List[str]` in the same order as input prompts. Much faster than sequential `llm_query` calls for independent queries.
4. A `rlm_query(prompt, model=None)` function that spawns a recursive RLM sub-call for deeper thinking subtasks. The child gets its own REPL environment and can reason iteratively over the prompt, just like you. Use this when a subtask requires multi-step reasoning, code execution, or its own iterative problem-solving. Falls back to `llm_query` if recursion is not available.
5. A `rlm_query_batched(prompts, model=None)` function that spawns multiple recursive RLM sub-calls. Each prompt gets its own child RLM. Falls back to `llm_query_batched` if recursion is not available.
6. A `SHOW_VARS()` function that returns all variables you have created in the REPL. Use this to check what variables exist before using FINAL_VAR.
7. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning.

{cost_guidance}

When to use `llm_query` vs `rlm_query`:
- Use `llm_query` for simple, one-shot tasks: extracting info from a chunk, summarizing text, answering a factual question, or classifying content.
- Use `rlm_query` when the subtask itself requires deeper thinking: multi-step reasoning, solving a sub-problem that needs its own REPL and iteration, or tasks where a single LM call might not be enough.

Breaking down problems: you must break problems into more digestible components. Use the REPL to write a programmatic strategy that chunks context, asks targeted sub-questions, branches on results, and combines answers in code.

REPL for computation: use the REPL for programmatic steps and then chain those results into an LLM call when needed.
```repl
import math
v_parallel = pitch * (q * B) / (2 * math.pi * m)
v_perp = R * (q * B) / m
theta_rad = math.atan2(v_perp, v_parallel)
theta_deg = math.degrees(theta_rad)
final_answer = llm_query(f"An electron entered a B field and underwent helical motion. Computed entry angle: {{theta_deg:.2f}} deg. State the answer clearly for the user.")
```

You will only be able to see truncated outputs from the REPL environment, so use the query functions on variables you want to analyze. Make sure to explicitly look through the entire context in REPL before answering your query. Break the context and the problem into digestible pieces: figure out a chunking strategy, query sub-LLMs over useful batches, save answers to buffers, and aggregate them.

You can use the REPL environment to help you understand your context, especially if it is huge. Remember that your sub-LLMs are powerful, so do not be afraid to put a lot of context into them when batching makes sense.

When you want to execute Python code in the REPL environment, wrap it in triple backticks with the `repl` language identifier. For example:
```repl
chunk = context[:10000]
answer = llm_query(f"What is the magic number in the context? Here is the chunk: {{chunk}}")
print(answer)
```

For independent queries over chunks, use batching:
```repl
query = "A man became famous for his book 'The Great Gatsby'. How many jobs did he have?"
chunk_size = len(context) // 10
chunks = []
for i in range(10):
    if i < 9:
        chunk_str = "\\n".join(context[i * chunk_size:(i + 1) * chunk_size])
    else:
        chunk_str = "\\n".join(context[i * chunk_size:])
    chunks.append(chunk_str)

prompts = [
    f"Try to answer the following query: {{query}}. Here are the documents:\\n{{chunk}}. Only answer if you are confident in your answer based on the evidence."
    for chunk in chunks
]
answers = llm_query_batched(prompts)
final_answer = llm_query(
    f"Aggregating all the answers per chunk, answer the original query about total number of jobs: {{query}}\\n\\nAnswers:\\n" + "\\n".join(answers)
)
```

For subtasks that require deeper reasoning, use `rlm_query`:
```repl
trend = rlm_query(f"Analyze this dataset and conclude with one word: up, down, or stable: {{data}}")
if "up" in trend.lower():
    recommendation = "Consider increasing exposure."
elif "down" in trend.lower():
    recommendation = "Consider hedging."
else:
    recommendation = "Hold position."
final_answer = llm_query(
    f"Given trend={{trend}} and recommendation={{recommendation}}, provide a one-sentence summary for the user."
)
```

IMPORTANT: When you are done with the iterative process, you must provide a final answer using one of the FINAL functions. Do not use these unless you have completed your task. You have two options:
1. Use FINAL(value) to provide the answer directly.
2. Use FINAL_VAR("variable_name") to return a variable by name.

WARNING: FINAL_VAR retrieves an existing variable. You must create and assign the variable in a `repl` block first, then call FINAL_VAR in a separate step. If you are unsure what variables exist, call SHOW_VARS().

Think step by step carefully, plan, and execute this plan immediately in your response. Do not just say what you will do. Use the REPL environment and sub-LLM calls as much as needed, and explicitly answer the original query in your final answer.
"""


RLM_SYSTEM_PROMPT = textwrap.dedent(_RLM_SYSTEM_PROMPT_BASE.format(cost_guidance=""))


# =============================================================================
# System Prompt for Qwen3-Coder-480B (with IMPORTANT cost warning from paper)
# Adds cost warning after the "sub LLMs are powerful" paragraph
# =============================================================================

RLM_SYSTEM_PROMPT_QWEN = textwrap.dedent(
    _RLM_SYSTEM_PROMPT_BASE.format(
        cost_guidance=(
            "IMPORTANT: Be careful about using `llm_query` because it can incur high runtime costs. "
            "Batch as much information as reasonably possible into each call and prefer `llm_query_batched` "
            "for independent chunk-level work.\n"
        )
    )
)


# =============================================================================
# User Prompt Templates (from official RLM repo)
# =============================================================================

USER_PROMPT = """Think step-by-step on what to do using the REPL environment (which contains the context) to answer the prompt.\n\nContinue using the REPL environment, which has the `context` variable, and querying sub-LLMs by writing to ```repl``` tags, and determine your answer. Your next action:"""

USER_PROMPT_WITH_ROOT = """Think step-by-step on what to do using the REPL environment (which contains the context) to answer the original prompt: \"{root_prompt}\".\n\nContinue using the REPL environment, which has the `context` variable, and querying sub-LLMs by writing to ```repl``` tags, and determine your answer. Your next action:"""


# =============================================================================
# Prompt Building Functions (from official RLM repo)
# =============================================================================


def build_rlm_system_prompt(
    system_prompt: str,
    query_metadata: QueryMetadata,
) -> List[dict]:
    """
    Build the initial system prompt for the REPL environment based on extra prompt metadata.

    Args:
        system_prompt: The system prompt to use
        query_metadata: QueryMetadata object containing context metadata

    Returns:
        List of message dictionaries [system, user(metadata)]
    """
    context_lengths = query_metadata.context_lengths
    context_total_length = query_metadata.context_total_length
    context_type = query_metadata.context_type

    # If there are more than 100 chunks, truncate to the first 100 chunks.
    if len(context_lengths) > 100:
        others = len(context_lengths) - 100
        context_lengths_str = (
            str(context_lengths[:100]) + "... [" + str(others) + " others]"
        )
    else:
        context_lengths_str = str(context_lengths)

    metadata = f"\n\nYour context is a {context_type} with {context_total_length} total characters, and is broken up into chunks of char lengths: {context_lengths_str}."

    return [
        {"role": "system", "content": system_prompt + metadata},
    ]


def build_user_prompt(
    root_prompt: Optional[str] = None,
    iteration: int = 0,
    context_count: int = 1,
    history_count: int = 0,
) -> dict:
    """
    Build the user prompt for a given iteration.

    Args:
        root_prompt: The original query/task
        iteration: Current iteration number (0 = first)
        context_count: Number of context variables available
        history_count: Number of prior conversation histories

    Returns:
        User message dict
    """
    if iteration == 0:
        safeguard = "You have not interacted with the REPL environment or seen your prompt / context yet. Your next action should be to look through and figure out how to answer the prompt, so don't just provide a final answer yet.\n\n"
        prompt = safeguard + (
            USER_PROMPT_WITH_ROOT.format(root_prompt=root_prompt)
            if root_prompt
            else USER_PROMPT
        )
    else:
        prompt = (
            "The history before is your previous interactions with the REPL environment. "
            + (
                USER_PROMPT_WITH_ROOT.format(root_prompt=root_prompt)
                if root_prompt
                else USER_PROMPT
            )
        )

    # Inform model about multiple contexts if present
    if context_count > 1:
        prompt += f"\n\nNote: You have {context_count} contexts available (context_0 through context_{context_count - 1})."

    # Inform model about prior conversation histories if present
    if history_count > 0:
        if history_count == 1:
            prompt += "\n\nNote: You have 1 prior conversation history available in the `history` variable."
        else:
            prompt += f"\n\nNote: You have {history_count} prior conversation histories available (history_0 through history_{history_count - 1})."

    return {"role": "user", "content": prompt}


# =============================================================================
# Parsing Utilities
# =============================================================================


def extract_code_blocks(text: str, language: str = "python") -> List[str]:
    """Extract code blocks from LLM response.

    Supports both ```repl``` (official RLM) and ```python``` style blocks.

    Args:
        text: The LLM response text
        language: Language identifier to match (default "python")

    Returns:
        List of code strings extracted from the response
    """
    # Match 'repl' (official) and 'python' (common alternative)
    patterns = [
        r"```repl\s*(.*?)```",
        rf"```{language}\s*(.*?)```",
    ]

    all_matches = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        all_matches.extend(m.strip() for m in matches if m.strip())

    return all_matches


def format_observations(
    observations,
    code_blocks: Optional[List[str]] = None,
    max_character_length: int = 20000,
) -> str:
    """Format REPL observations into observation text for the LLM.

    Follows the official RLM format: each observation includes the executed code
    and its output, so the LLM sees what it wrote alongside the result.

    Args:
        observations: List of REPL observations from env.step()
        code_blocks: Optional list of code strings that produced these observations
        max_character_length: Truncate output beyond this length

    Returns:
        Formatted observation string
    """
    formatted = []
    for i, observation in enumerate(observations, 1):
        output = (
            observation.result.stdout.strip()
            if observation.result.stdout
            else "(no output)"
        )
        if not observation.result.success:
            error = (
                observation.result.stderr
                or observation.result.exception
                or "Unknown error"
            )
            output = f"{output}\n\nERROR: {error}"

        # Truncate large outputs
        if len(output) > max_character_length:
            output = (
                output[:max_character_length]
                + f"... [{len(output) - max_character_length} chars truncated]"
            )

        # Echo back executed code (matches official RLM format)
        if code_blocks and i <= len(code_blocks):
            formatted.append(
                f"Code executed:\n```python\n{code_blocks[i - 1]}\n```\n\nREPL output:\n{output}"
            )
        else:
            formatted.append(f"Code block {i} output:\n{output}")

    return "\n\n".join(formatted)
