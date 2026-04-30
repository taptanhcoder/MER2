from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Registry:
    name: str
    _items: dict[str, Any] = field(default_factory=dict)

    def register(self, key: str) -> Callable[[Any], Any]:
        def decorator(obj: Any) -> Any:
            if key in self._items:
                raise KeyError(f"{self.name} already contains key: {key}")
            self._items[key] = obj
            return obj

        return decorator

    def add(self, key: str, obj: Any) -> None:
        if key in self._items:
            raise KeyError(f"{self.name} already contains key: {key}")
        self._items[key] = obj

    def get(self, key: str) -> Any:
        if key not in self._items:
            raise KeyError(
                f"Unknown key '{key}' in registry '{self.name}'. "
                f"Available: {sorted(self._items)}"
            )
        return self._items[key]

    def has(self, key: str) -> bool:
        return key in self._items

    def available(self) -> list[str]:
        return sorted(self._items.keys())


MODEL_REGISTRY = Registry("models")
LOSS_REGISTRY = Registry("losses")
OPTIMIZER_REGISTRY = Registry("optimizers")
SCHEDULER_REGISTRY = Registry("schedulers")
FUSION_REGISTRY = Registry("fusion")