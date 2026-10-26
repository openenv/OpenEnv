# SPDX-License-Identifier: BSD-3-Clause

"""Base Rubric class for reward computation.

Rubrics compute rewards from actions and observations. The API is modeled
after PyTorch's nn.Module: users implement forward(), and the framework
handles child registration and hooks.

See RFC 004 for full design: rfcs/004-rubrics.md
"""

import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple


class Rubric(ABC):
    """Abstract base class for reward computation.

    A Rubric computes a reward signal from an action and observation.
    Subclasses implement forward() to define the reward logic.

    Examples:

        ```python
        class MyRubric(Rubric):
            def forward(self, action, observation) -> float:
                return 1.0 if action.valid else 0.0

        rubric = MyRubric()
        reward = rubric(action, observation)
        ```

    Child rubrics are auto-registered when assigned as attributes,
    enabling hierarchical composition and introspection.
    """

    _rubric_children: Dict[str, "Rubric"]
    _forward_hooks: List[Callable]
    _forward_pre_hooks: List[Callable]
    last_score: Optional[float]

    def __init__(self):
        # Use object.__setattr__ to avoid triggering __setattr__ during init
        object.__setattr__(self, "_rubric_children", {})
        object.__setattr__(self, "_forward_hooks", [])
        object.__setattr__(self, "_forward_pre_hooks", [])
        object.__setattr__(self, "last_score", None)

    def __setattr__(self, name: str, value: Any) -> None:
        # Auto-register child rubrics when assigned as attributes
        if isinstance(value, Rubric):
            self._rubric_children[name] = value
        object.__setattr__(self, name, value)

    def __call__(self, action: Any, observation: Any):
        """Evaluate the rubric with hooks.

        Args:
            action: The action taken by the agent.
            observation: The resulting observation.

        Returns:
            `float`: Reward value (typically 0.0 to 1.0).
        """
        # Check if forward method is async BEFORE calling it
        if inspect.iscoroutinefunction(self.forward):
            # Async path - pre-hooks will be called in _call_async
            result = self.forward(action, observation)
            return self._call_async(action, observation, result)
        else:
            # Sync path - call pre-hooks BEFORE forward()
            for hook in self._forward_pre_hooks:
                hook(self, action, observation)
            result = self.forward(action, observation)
            if inspect.iscoroutine(result):
                # forward() handed back an async child's call unawaited (e.g. a
                # RubricDict dispatch): await it before recording the score
                return self._finish_awaited(action, observation, result)
            return self._call_sync(action, observation, result)

    def _call_sync(self, action: Any, observation: Any, result: float) -> float:
        """Synchronous call path."""
        return self._finish_forward(action, observation, result)

    def _run_forward_pre_hooks(
        self, action: Any, observation: Any, start: int = 0
    ) -> None:
        """Run pre-forward hooks synchronously, from index ``start`` on."""
        for hook in self._forward_pre_hooks[start:]:
            hook(self, action, observation)

    async def _run_forward_pre_hooks_async(
        self, action: Any, observation: Any, start: int = 0
    ) -> None:
        """Run pre-forward hooks from an async call path, from index ``start`` on."""
        for hook in self._forward_pre_hooks[start:]:
            if inspect.iscoroutinefunction(hook):
                await hook(self, action, observation)
            else:
                hook(self, action, observation)

    def _run_leading_sync_pre_hooks(self, action: Any, observation: Any) -> int:
        """Run pre-forward hooks up to the first async one; return how many ran.

        Containers call this before evaluating any child: they learn whether
        they are on the async path only by calling a child, but pre-hooks must
        run first. The rest are passed as ``start`` to the hook runners above.
        """
        ran = 0
        for hook in self._forward_pre_hooks:
            if inspect.iscoroutinefunction(hook):
                break
            hook(self, action, observation)
            ran += 1
        return ran

    def _finish_forward(self, action: Any, observation: Any, result: float) -> float:
        """Store the result and run post-forward hooks synchronously."""
        self.last_score = result

        # Post-forward hooks
        for hook in self._forward_hooks:
            hook(self, action, observation, result)

        return result

    async def _finish_forward_async(
        self, action: Any, observation: Any, result: float
    ) -> float:
        """Store the result and run post-forward hooks from an async call path."""
        self.last_score = result

        # Post-forward hooks
        for hook in self._forward_hooks:
            if inspect.iscoroutinefunction(hook):
                await hook(self, action, observation, result)
            else:
                hook(self, action, observation, result)

        return result

    async def _finish_awaited(
        self, action: Any, observation: Any, result_coro
    ) -> float:
        """Await a coroutine returned by a sync forward(); pre-hooks already ran."""
        result = await result_coro
        return await self._finish_forward_async(action, observation, result)

    async def _call_async(self, action: Any, observation: Any, result_coro) -> float:
        """Asynchronous call path."""
        # Pre-forward hooks
        await self._run_forward_pre_hooks_async(action, observation)

        # Await the forward result
        result = await result_coro
        return await self._finish_forward_async(action, observation, result)

    @abstractmethod
    def forward(self, action: Any, observation: Any) -> float:
        """Compute the reward. Implement this in subclasses.

        Args:
            action: The action taken by the agent.
            observation: The resulting observation.

        Returns:
            `float`: Reward value (typically 0.0 to 1.0).
        """
        raise NotImplementedError

    def register_forward_hook(
        self, hook: Callable[["Rubric", Any, Any, float], None]
    ) -> None:
        """Register a hook called after forward().

        Args:
            hook (`Callable`):
                Callable with signature (rubric, action, observation, result).
        """
        self._forward_hooks.append(hook)

    def register_forward_pre_hook(
        self, hook: Callable[["Rubric", Any, Any], None]
    ) -> None:
        """Register a hook called before forward().

        Args:
            hook (`Callable`):
                Callable with signature (rubric, action, observation).
        """
        self._forward_pre_hooks.append(hook)

    def children(self) -> Iterator["Rubric"]:
        """Iterate over immediate child rubrics."""
        yield from self._rubric_children.values()

    def named_children(self) -> Iterator[Tuple[str, "Rubric"]]:
        """Iterate over immediate child rubrics with names."""
        yield from self._rubric_children.items()

    def rubrics(self) -> Iterator["Rubric"]:
        """Iterate over all descendant rubrics (depth-first)."""
        for child in self._rubric_children.values():
            yield child
            yield from child.rubrics()

    def named_rubrics(self, prefix: str = "") -> Iterator[Tuple[str, "Rubric"]]:
        """Iterate over all descendant rubrics with dot-separated names."""
        for name, child in self._rubric_children.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, child
            yield from child.named_rubrics(full_name)

    def get_rubric(self, path: str) -> "Rubric":
        """Access a nested rubric by dot-separated path.

        Args:
            path (`str`):
                Dot-separated path (e.g., "code.syntax").

        Returns:
            `Rubric`: The rubric at the specified path.

        Raises:
            KeyError: If the path does not exist.
        """
        parts = path.split(".")
        current = self
        for part in parts:
            if part not in current._rubric_children:
                raise KeyError(f"Rubric path not found: {path}")
            current = current._rubric_children[part]
        return current

    def reset(self) -> None:
        """Reset any internal state. Override in subclasses if needed."""
        pass

    def state_dict(self) -> Dict[str, Any]:
        """Serialize rubric configuration for checkpointing."""
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load rubric configuration from checkpoint."""
        pass
