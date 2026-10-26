# envs/finqa_env/server/rewards.py
"""
Reward computation for the FinQA environment.

Uses fuzzy numerical matching to compare predicted answers against ground truth.
Handles various formats: \boxed{}, percentages, fractions, decimals.
"""

import re
from fractions import Fraction
from typing import Optional, Tuple


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract answer from \boxed{...} format.

    Args:
        text: Text potentially containing \boxed{answer}

    Returns:
        The extracted answer or None if not found
    """
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    return None


def extract_all_boxed_answers(text: str) -> list:
    """
    Extract all answers from \boxed{...} format.

    Args:
        text: Text potentially containing multiple \boxed{answer}

    Returns:
        List of extracted answers
    """
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    return [m.strip() for m in matches]


def parse_number(text: str, convert_percent: bool = True) -> Optional[float]:
    """
    Parse a string into a float, handling various formats.

    Handles:
    - Plain numbers: "6.118", "-3.14"
    - Percentages: "20.9%", "20.9 %"
    - Fractions: "1/2", "3/4"
    - Thousands separators: "1,234.56"
    - Negative numbers in parens: "(100)"

    Args:
        text: String to parse
        convert_percent: If True, divide percentages by 100. If False, just strip the % sign.

    Returns:
        Float value or None if parsing fails
    """
    if text is None:
        return None

    text = text.strip()

    if not text:
        return None

    try:
        # Remove LaTeX annotations like \text{million}, \text{%}, etc.
        text = re.sub(r"\\text\{[^}]*\}", "", text)

        # Remove currency symbols ($ and \$)
        text = text.replace("\\$", "").replace("$", "").strip()

        # Handle percentage (including LaTeX escaped \%)
        if "%" in text or "\\%" in text:
            text = text.replace("\\%", "").replace("%", "").strip()
            if convert_percent:
                return float(text.replace(",", "")) / 100
            else:
                return float(text.replace(",", ""))

        # Handle parentheses for negative numbers
        if text.startswith("(") and text.endswith(")"):
            text = "-" + text[1:-1]

        # Handle fractions (e.g., "1/2", "3/4")
        if "/" in text and not text.startswith("-"):
            try:
                return float(Fraction(text))
            except (ValueError, ZeroDivisionError):
                pass

        # Handle negative fractions
        if text.startswith("-") and "/" in text:
            try:
                return -float(Fraction(text[1:]))
            except (ValueError, ZeroDivisionError):
                pass

        # Remove thousands separators and parse
        text = text.replace(",", "")
        return float(text)

    except (ValueError, TypeError):
        return None


def normalize_answer(
    answer: str, convert_percent: bool = True
) -> Tuple[Optional[float], str]:
    """
    Normalize an answer string to a comparable format.

    Args:
        answer: Raw answer string
        convert_percent: If True, divide percentages by 100. If False, just strip the % sign.

    Returns:
        Tuple of (parsed_number, cleaned_string)
    """
    if answer is None:
        return None, ""

    # Try to extract from \boxed{} first
    boxed = extract_boxed_answer(answer)
    if boxed:
        answer = boxed

    # Clean up whitespace
    answer = answer.strip()

    # Try to parse as number
    num = parse_number(answer, convert_percent)

    return num, answer.lower()


def extract_numbers_from_multi_value(text: str) -> list:
    """
    Extract all numbers from a comma/semicolon separated string.
    Handles formats like "2022: 0.933, 2023: 0.930" or "0.933, 0.931, 0.930".
    """
    parts = _split_multi_value(text)
    return [num for _, num in parts]


def _split_multi_value(text: str) -> list:
    """
    Extract (key, number) pairs from a comma/semicolon separated string.

    Returns list of (key, float) tuples. Key is a year string like "2022"
    if found, otherwise None.
    """
    # Split by comma or semicolon (with optional LaTeX spacing like \; or \ )
    parts = re.split(r"[,;]\s*|\\[;,]\s*", text)
    results = []
    for part in parts:
        # Strip LaTeX whitespace commands (\ , \;, \,)
        part = re.sub(r"\\[;, ]", " ", part).strip()
        if not part:
            continue
        # Try to extract a year label (e.g. "2022:", "2022 to 2023:", "2022→2023:")
        # Normalize \rightarrow and similar to "to" before matching
        part_normalized = re.sub(r"\\rightarrow|→|->|−>", " to ", part)
        year_match = re.search(r"(20\d{2}(?:\s*to\s*20\d{2})?)", part_normalized)
        key = year_match.group(1) if year_match else None
        # Remove label prefix like "2022:" or "2022:\"
        cleaned = re.sub(r"^[^:]*:\s*\\?\s*", "", part)
        num = parse_number(cleaned)
        if num is not None:
            results.append((key, num))
    return results


def compare_single_values(
    pred_num: Optional[float],
    truth_num: Optional[float],
    pred_str: str,
    truth_str: str,
    tolerance: float = 0.01,
    max_absolute_diff: float = 1.0,
) -> bool:
    """Compare two single values."""
    # If both are numbers, compare numerically with tolerance
    if pred_num is not None and truth_num is not None:
        # Handle zero case
        if truth_num == 0:
            return abs(pred_num) < 0.001

        # Calculate both errors
        abs_diff = abs(pred_num - truth_num)
        relative_error = abs_diff / abs(truth_num)

        # BOTH conditions must pass
        return relative_error <= tolerance and abs_diff <= max_absolute_diff

    # If one is a number and other isn't, not equal
    if (pred_num is None) != (truth_num is None):
        return False

    # Fall back to string comparison
    return pred_str == truth_str


def compute_reward(
    predicted: str,
    ground_truth: str,
    tolerance: float = 0.01,
    max_absolute_diff: float = 1.0,
) -> float:
    """
    Compute reward based on answer correctness.

    Uses fuzzy numerical matching with BOTH relative and absolute tolerance checks.
    A prediction is correct only if it passes BOTH conditions.

    Handles multiple values (e.g., ground truth with multiple \boxed{} values).

    Args:
        predicted: The predicted answer from the agent
        ground_truth: The expected correct answer
        tolerance: Relative tolerance for numerical comparison (default 1%)
        max_absolute_diff: Maximum absolute difference allowed (default 1.0)

    Returns:
        1.0 if correct, 0.0 if incorrect
    """
    # Check for multiple boxed answers in ground truth
    truth_boxed = extract_all_boxed_answers(ground_truth)

    if len(truth_boxed) > 1:
        # Multiple ground truth values - split prediction by comma/semicolon
        pred_values = re.split(r"[,;]\s*", predicted.strip())

        if len(pred_values) != len(truth_boxed):
            return 0.0  # Different number of values

        # Compare each pair
        for pred_val, truth_val in zip(pred_values, truth_boxed):
            # Strip year/label prefix (e.g. "2024: -4" -> "-4")
            pred_val_cleaned = (
                re.sub(r"^[^:]*:\s*", "", pred_val) if ":" in pred_val else pred_val
            )
            pred_num, pred_str = normalize_answer(pred_val_cleaned)
            truth_num, truth_str = normalize_answer(truth_val)

            if not compare_single_values(
                pred_num, truth_num, pred_str, truth_str, tolerance, max_absolute_diff
            ):
                # Fallback: try without % conversion (for percentage points like "4.5%" vs "4.5")
                pred_num_no_pct, _ = normalize_answer(pred_val, convert_percent=False)
                if not compare_single_values(
                    pred_num_no_pct,
                    truth_num,
                    pred_str,
                    truth_str,
                    tolerance,
                    max_absolute_diff,
                ):
                    return 0.0

        return 1.0  # All values matched

    # Single value comparison
    pred_num, pred_str = normalize_answer(predicted)
    truth_num, truth_str = normalize_answer(ground_truth)

    if compare_single_values(
        pred_num, truth_num, pred_str, truth_str, tolerance, max_absolute_diff
    ):
        return 1.0

    pred_num_no_pct, _ = normalize_answer(predicted, convert_percent=False)
    if compare_single_values(
        pred_num_no_pct, truth_num, pred_str, truth_str, tolerance, max_absolute_diff
    ):
        return 1.0

    # Fallback: multi-value inside single \boxed{} (only if truth didn't parse as single number)
    if len(truth_boxed) == 1 and truth_num is None:
        truth_pairs = _split_multi_value(truth_boxed[0])
        pred_pairs = _split_multi_value(predicted)
        if len(truth_pairs) > 1 and len(pred_pairs) == len(truth_pairs):
            # If both sides have year keys, match by key (order-independent)
            truth_keys = {k for k, _ in truth_pairs if k is not None}
            pred_keys = {k for k, _ in pred_pairs if k is not None}
            if truth_keys and pred_keys and truth_keys == pred_keys:
                truth_map = {k: v for k, v in truth_pairs}
                pred_map = {k: v for k, v in pred_pairs}
                for key in truth_map:
                    p, t = pred_map[key], truth_map[key]
                    abs_diff = abs(p - t)
                    rel_err = (
                        abs_diff / abs(t) if t != 0 else (0 if p == 0 else float("inf"))
                    )
                    if not (rel_err <= tolerance and abs_diff <= max_absolute_diff):
                        return 0.0
                return 1.0

            # Otherwise fall back to positional matching
            for (_, p), (_, t) in zip(pred_pairs, truth_pairs):
                abs_diff = abs(p - t)
                rel_err = (
                    abs_diff / abs(t) if t != 0 else (0 if p == 0 else float("inf"))
                )
                if not (rel_err <= tolerance and abs_diff <= max_absolute_diff):
                    return 0.0
            return 1.0

    return 0.0
