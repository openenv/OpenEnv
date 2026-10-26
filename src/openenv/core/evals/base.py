# SPDX-License-Identifier: BSD-3-Clause

"""Base class for evaluation harnesses."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from openenv.core.evals.types import EvalConfig, EvalResult


class EvalHarness(ABC):
    """Abstract base class for evaluation harnesses.

    Subclasses implement run() to define evaluation logic.
    """

    @abstractmethod
    def run(
        self,
        harness_version: str,
        library_versions: Dict[str, str],
        dataset: str,
        eval_parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run the evaluation and return scores.

        Args:
            harness_version: Version of the evaluation harness.
            library_versions: Versions of libraries used in the evaluation.
            dataset: Name of the dataset to evaluate on.
            eval_parameters: Parameters for the evaluation.

        Returns:
            Dictionary of scores from the evaluation.
        """
        raise NotImplementedError

    def run_from_config(self, config: EvalConfig) -> EvalResult:
        """Run evaluation from an EvalConfig and return an EvalResult.

        Args:
            config: Configuration for the evaluation.

        Returns:
            EvalResult containing the config and scores.
        """
        scores = self.run(
            harness_version=config.harness_version,
            library_versions=config.library_versions,
            dataset=config.dataset,
            eval_parameters=config.eval_parameters,
        )
        return EvalResult(config=config, scores=scores)

    @property
    def name(self) -> str:
        """Return the name of the harness (class name)."""
        return self.__class__.__name__
