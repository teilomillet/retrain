"""Candidate archive for test-time discovery."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class DiscoverEntry:
    """One archived candidate solution."""

    entry_id: int
    text: str
    reward: float
    parent_id: int | None
    created_step: int
    depth: int
    expansions: int = 0
    best_child_reward: float | None = None
    children: list[int] = field(default_factory=list)

    @property
    def q_value(self) -> float:
        if self.best_child_reward is None:
            return self.reward
        return max(self.reward, self.best_child_reward)

    @property
    def is_leaf(self) -> bool:
        return not self.children


class DiscoverArchive:
    """Archive of reusable states for a single discovery problem."""

    def __init__(self, empty_reward: float) -> None:
        self.entries: dict[int, DiscoverEntry] = {
            0: DiscoverEntry(
                entry_id=0,
                text="",
                reward=float(empty_reward),
                parent_id=None,
                created_step=-1,
                depth=0,
            )
        }
        self.total_expansions = 0
        self._next_id = 1

    def __len__(self) -> int:
        return len(self.entries)

    def get(self, entry_id: int) -> DiscoverEntry:
        return self.entries[entry_id]

    def best_entry(self) -> DiscoverEntry:
        return max(
            self.entries.values(),
            key=lambda e: (e.reward, e.q_value, -e.depth, -e.entry_id),
        )

    def record_selection(self, entry_id: int) -> None:
        self.total_expansions += 1
        current = entry_id
        while current is not None:
            entry = self.entries[current]
            entry.expansions += 1
            current = entry.parent_id

    def add_attempt(
        self,
        *,
        parent_id: int,
        text: str,
        reward: float,
        step: int,
    ) -> DiscoverEntry:
        parent = self.entries[parent_id]
        entry = DiscoverEntry(
            entry_id=self._next_id,
            text=text,
            reward=float(reward),
            parent_id=parent_id,
            created_step=step,
            depth=parent.depth + 1,
        )
        self.entries[entry.entry_id] = entry
        self._next_id += 1
        parent.children.append(entry.entry_id)
        self._update_best_child_chain(parent_id, entry.reward)
        return entry

    def _update_best_child_chain(self, parent_id: int | None, reward: float) -> None:
        current = parent_id
        while current is not None:
            entry = self.entries[current]
            if entry.best_child_reward is None or reward > entry.best_child_reward:
                entry.best_child_reward = reward
            current = entry.parent_id

    def _recompute_best_child_chain(self, parent_id: int | None) -> None:
        current = parent_id
        while current is not None:
            entry = self.entries[current]
            best: float | None = None
            for child_id in entry.children:
                child = self.entries[child_id]
                candidate = child.q_value
                if best is None or candidate > best:
                    best = candidate
            entry.best_child_reward = best
            current = entry.parent_id

    def prune(self, max_entries: int) -> None:
        while len(self.entries) > max_entries:
            candidates = [
                entry
                for entry in self.entries.values()
                if entry.entry_id != 0 and entry.is_leaf
            ]
            if not candidates:
                return
            victim = min(
                candidates,
                key=lambda e: (e.reward, e.expansions, e.created_step, e.depth, e.entry_id),
            )
            parent_id = victim.parent_id
            if parent_id is not None:
                parent = self.entries[parent_id]
                parent.children = [cid for cid in parent.children if cid != victim.entry_id]
            del self.entries[victim.entry_id]
            self._recompute_best_child_chain(parent_id)

    def _rank_priors(self) -> dict[int, float]:
        ordered = sorted(
            self.entries.values(),
            key=lambda e: (e.reward, e.q_value, -e.depth, -e.entry_id),
            reverse=True,
        )
        n = len(ordered)
        denom = n * (n + 1) / 2.0
        priors: dict[int, float] = {}
        for idx, entry in enumerate(ordered):
            priors[entry.entry_id] = (n - idx) / denom
        return priors

    def select(self, batch_size: int, exploration: float) -> list[DiscoverEntry]:
        if batch_size <= 0:
            return []
        priors = self._rank_priors()

        def _score(entry: DiscoverEntry) -> tuple[float, float, float, int]:
            prior = priors.get(entry.entry_id, 0.0)
            bonus = (
                exploration
                * prior
                * math.sqrt(1.0 + float(self.total_expansions))
                / (1.0 + float(entry.expansions))
            )
            return (entry.q_value + bonus, entry.reward, -float(entry.depth), -entry.entry_id)

        ordered = sorted(self.entries.values(), key=_score, reverse=True)
        if not ordered:
            return []
        selected = ordered[:batch_size]
        while len(selected) < batch_size:
            selected.append(ordered[len(selected) % len(ordered)])
        return selected

    def context_entries(self, start_id: int, limit: int) -> list[DiscoverEntry]:
        if limit <= 0:
            return []
        others = [
            entry for entry in self.entries.values() if entry.entry_id != start_id and entry.text
        ]
        return sorted(
            others,
            key=lambda e: (e.reward, e.q_value, -e.depth, -e.entry_id),
            reverse=True,
        )[:limit]
