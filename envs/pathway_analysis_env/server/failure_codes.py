# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stable ``failure_code`` strings for observation metadata (pathway_analysis_env v1)."""

# Session / episode
EPISODE_ALREADY_DONE = "episode_already_done"
UNKNOWN_ACTION_TYPE = "unknown_action_type"

# understand_experiment_design
DESIGN_PARTIAL_CONTRAST = "design_partial_contrast"
DESIGN_INVALID_CONTRAST_NAMES = "design_invalid_contrast_names"
DESIGN_INSUFFICIENT_SAMPLES_PER_ARM = "design_insufficient_samples_per_arm"

# run_differential_expression
DE_MISSING_CONTRAST = "de_missing_contrast"
DE_PYDESeq2_UNAVAILABLE = "de_pydeseq2_unavailable"
DE_DESEQ2_FAILED = "de_deseq2_failed"
DE_INVALID_COUNTS_MATRIX = "de_invalid_counts_matrix"
DE_INVALID_AUTHOR_TABLE = "de_invalid_author_table"
DE_TOO_FEW_GENES_AFTER_FILTER = "de_too_few_genes_after_filter"
CASE_SAMPLE_METADATA_MISMATCH = "case_sample_metadata_mismatch"

DE_UNSUPPORTED_PRECOMPUTED_CONTRAST = "de_unsupported_precomputed_contrast"

# run_pathway_enrichment
ORA_DE_PREREQUISITE = "ora_de_prerequisite"
ORA_NO_PATHWAY_DEFINITIONS = "ora_no_pathway_definitions"

ORA_GSEAPY_UNAVAILABLE = "ora_gseapy_unavailable"
ORA_ENRICHR_FAILED = "ora_enrichr_failed"

# compare_pathways
COMPARE_MISSING_PATHWAY_NAMES = "compare_missing_pathway_names"

# ask_expert
EXPERT_DISABLED = "expert_disabled"
EXPERT_BUDGET_EXHAUSTED = "expert_budget_exhausted"

# submit (analytics)
SUBMIT_INCORRECT_HYPOTHESIS = "submit_incorrect_hypothesis"

# strict mode umbrella (specific code still preferred when set)
STRICT_TERMINATION = "strict_termination"
