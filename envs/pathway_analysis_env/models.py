# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for Pathway Analysis Environment.

Supports pipeline-style episodes (count matrix + sample metadata + gene sets)
with PyDESeq2 differential expression, Fisher ORA, overlap-aware summaries,
optional expert calls, and HTML step traces.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core.env_server import Action, Observation, State
from pydantic import Field


class PathwayAction(Action):
    """
    Action for the Pathway Analysis environment.

    action_type:
        - ``inspect_dataset``: describe available samples and conditions.
        - ``understand_experiment_design``: **(1)** Summarize groups (conditions, sample counts);
          optionally **(2)** validate ``condition_a``/``condition_b`` as reference/alternate for
          DGE (does not run DESeq2). Valid pairs feed ``run_differential_expression`` when DE omits
          conditions. **(3)** Pathway steps follow DE.
        - ``run_differential_expression``: PyDESeq2 contrast (needs ``condition_a`` /
          ``condition_b`` when using count-matrix cases).
        - ``run_pathway_enrichment``: ORA on DE genes vs pathway gene sets.
        - ``compare_pathways``: contrast exclusive vs shared DE support between two
          pathways (``pathway_a``, ``pathway_b``).
        - ``ask_expert``: curriculum / ablation hint with budget and penalty.
        - ``submit_answer``: submit ``hypothesis`` pathway name and end episode.
    """

    action_type: str
    condition_a: Optional[str] = None
    condition_b: Optional[str] = None
    gene_list: Optional[List[str]] = None
    hypothesis: Optional[str] = None
    pathway_a: Optional[str] = None
    pathway_b: Optional[str] = None
    expert_question: Optional[str] = None


class PathwayObservation(Observation):
    """Observation with optional rich DE / ORA structures (JSON-serializable)."""

    message: str = ""
    available_conditions: List[str] = Field(default_factory=list)
    top_genes: List[str] = Field(default_factory=list)
    top_pathways: List[str] = Field(default_factory=list)
    de_genes: List[Dict[str, Any]] = Field(default_factory=list)
    pathway_enrichment: List[Dict[str, Any]] = Field(default_factory=list)
    pathway_comparison: Optional[Dict[str, Any]] = None
    overlap_summary: Optional[Dict[str, Any]] = None
    statistical_ambiguity: Optional[Dict[str, Any]] = None
    expert_message: Optional[str] = None
    trace_path: Optional[str] = None
    experiment_design: Optional[Dict[str, Any]] = None


class PathwayState(State):
    """Episode state including pipeline mode flags and expert budget."""

    true_pathway: str = ""
    conditions: List[str] = Field(default_factory=list)
    de_run: bool = False
    enrichment_run: bool = False
    is_done: bool = False
    pipeline_mode: bool = False
    strict_mode: bool = False
    expert_calls_used: int = 0
    expert_budget: int = 0
    legacy_mode: bool = False
    design_understood: bool = False
    validated_reference: Optional[str] = None
    validated_alternate: Optional[str] = None
