from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    observation: np.ndarray
    action: int
    logprob: float
    reward: float
    done: bool
    value: float
    head: str


class RolloutBuffer:
    def __init__(self) -> None:
        self.transitions: list[Transition] = []
        self.advantages: list[float] = []
        self.returns: list[float] = []

    def __len__(self) -> int:
        return len(self.transitions)

    def clear(self) -> None:
        self.transitions.clear()
        self.advantages.clear()
        self.returns.clear()

    def add(
        self,
        *,
        observation: np.ndarray,
        action: int,
        logprob: float,
        reward: float,
        done: bool,
        value: float,
        head: str,
    ) -> None:
        self.transitions.append(
            Transition(
                observation=np.asarray(observation, dtype=np.float32),
                action=int(action),
                logprob=float(logprob),
                reward=float(reward),
                done=bool(done),
                value=float(value),
                head=str(head),
            )
        )

    def set_last_reward(self, reward: float, *, done: bool = True) -> None:
        if not self.transitions:
            return
        last = self.transitions[-1]
        self.transitions[-1] = Transition(
            observation=last.observation,
            action=last.action,
            logprob=last.logprob,
            reward=float(last.reward + reward),
            done=bool(done),
            value=last.value,
            head=last.head,
        )

    def finish_trajectory(self, *, last_value: float, gamma: float, gae_lambda: float) -> None:
        if not self.transitions:
            self.advantages = []
            self.returns = []
            return

        rewards = np.asarray([t.reward for t in self.transitions], dtype=np.float32)
        values = np.asarray([t.value for t in self.transitions] + [float(last_value)], dtype=np.float32)
        dones = np.asarray([t.done for t in self.transitions], dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for index in reversed(range(len(self.transitions))):
            nonterminal = 1.0 - dones[index]
            delta = rewards[index] + gamma * values[index + 1] * nonterminal - values[index]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            advantages[index] = gae

        returns = advantages + values[:-1]
        self.advantages = advantages.tolist()
        self.returns = returns.tolist()
