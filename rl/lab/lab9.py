"""
10-Armed Testbed: UCB vs ε-Greedy Action Selection
Sutton & Barto, Reinforcement Learning: An Introduction

Pure stdlib implementation (math, random, statistics) + matplotlib for plotting.
"""

from __future__ import annotations

import math
import os
import random
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class BanditEnvironment:
    """Stationary 10-armed bandit.

    True action values q*(a) ~ N(0,1); rewards R_t ~ N(q*(a), 1).
    """

    NUM_ARMS: int = 10

    def __init__(self) -> None:
        self.q_star: list[float] = [
            random.gauss(0.0, 1.0) for _ in range(self.NUM_ARMS)
        ]
        self.optimal_action: int = self.q_star.index(max(self.q_star))

    def step(self, action: int) -> float:
        """Return a stochastic reward for *action*."""
        return random.gauss(self.q_star[action], 1.0)

    def reset(self) -> None:
        """Re-sample a fresh problem instance."""
        self.q_star = [random.gauss(0.0, 1.0) for _ in range(self.NUM_ARMS)]
        self.optimal_action = self.q_star.index(max(self.q_star))


# ---------------------------------------------------------------------------
# Agents (abstract base + concrete implementations)
# ---------------------------------------------------------------------------


class Agent(ABC):
    """Base class for bandit agents."""

    def __init__(self, n_arms: int = 10) -> None:
        self.n_arms: int = n_arms
        self.Q: list[float] = [0.0] * n_arms  # action-value estimates
        self.N: list[int] = [0] * n_arms  # action counts
        self.t: int = 0  # global step counter

    def reset(self) -> None:
        self.Q = [0.0] * self.n_arms
        self.N = [0] * self.n_arms
        self.t = 0

    def update(self, action: int, reward: float) -> None:
        """Incremental sample-average update."""
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]

    @abstractmethod
    def select_action(self) -> int:
        """Return the chosen arm index."""

    @property
    @abstractmethod
    def label(self) -> str:
        """Human-readable identifier for legend / logging."""


class EpsilonGreedyAgent(Agent):
    """ε-greedy with incremental sample-average estimates."""

    def __init__(self, epsilon: float, n_arms: int = 10) -> None:
        super().__init__(n_arms)
        self.epsilon = epsilon

    @property
    def label(self) -> str:
        return f"ε-greedy  ε={self.epsilon}"

    def select_action(self) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_arms)
        # argmax with random tie-breaking
        max_q = max(self.Q)
        ties = [a for a, q in enumerate(self.Q) if q == max_q]
        return random.choice(ties)


class UCBAgent(Agent):
    """Upper Confidence Bound action selection (UCB1).

    A_t = argmax_a [ Q_t(a) + c * sqrt( ln(t) / N_t(a) ) ]

    Unvisited arms (N=0) are always considered maximising.
    """

    def __init__(self, c: float, n_arms: int = 10) -> None:
        super().__init__(n_arms)
        self.c = c

    @property
    def label(self) -> str:
        return f"UCB  c={self.c}"

    def select_action(self) -> int:
        self.t += 1  # increment *before* computing bonus (t >= 1 always)

        # Any arm with N=0 is an unvisited arm → treat as maximising
        unvisited = [a for a in range(self.n_arms) if self.N[a] == 0]
        if unvisited:
            return random.choice(unvisited)

        log_t = math.log(self.t)
        ucb_values = [
            self.Q[a] + self.c * math.sqrt(log_t / self.N[a])
            for a in range(self.n_arms)
        ]
        max_ucb = max(ucb_values)
        ties = [a for a, v in enumerate(ucb_values) if v == max_ucb]
        return random.choice(ties)


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Aggregated statistics over many independent runs."""

    label: str
    avg_rewards: list[float] = field(default_factory=list)
    pct_optimal: list[float] = field(default_factory=list)


def run_experiment(
    agent: Agent,
    n_runs: int = 2_000,
    n_steps: int = 1_000,
) -> RunResult:
    """Run *n_runs* independent episodes of *n_steps* each.

    Returns per-step averages across all runs.
    """
    # Accumulators: sum over runs so we can divide at the end
    reward_sum: list[float] = [0.0] * n_steps
    optimal_sum: list[int] = [0] * n_steps

    env = BanditEnvironment()

    for _ in range(n_runs):
        env.reset()
        agent.reset()

        for t in range(n_steps):
            action = agent.select_action()
            reward = env.step(action)
            agent.update(action, reward)

            reward_sum[t] += reward
            optimal_sum[t] += int(action == env.optimal_action)

    avg_rewards = [s / n_runs for s in reward_sum]
    pct_optimal = [100.0 * s / n_runs for s in optimal_sum]

    return RunResult(
        label=agent.label,
        avg_rewards=avg_rewards,
        pct_optimal=pct_optimal,
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_STYLE = {
    "figure.facecolor": "#0f1117",
    "axes.facecolor": "#181c27",
    "axes.edgecolor": "#3a3f55",
    "axes.labelcolor": "#c8ccd8",
    "axes.titlecolor": "#e8ecf4",
    "xtick.color": "#7a7f94",
    "ytick.color": "#7a7f94",
    "grid.color": "#2a2f44",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "legend.facecolor": "#1e2233",
    "legend.edgecolor": "#3a3f55",
    "legend.labelcolor": "#c8ccd8",
    "text.color": "#c8ccd8",
    "font.family": "monospace",
}

# Colour palette: 4 warm tones for UCB, 3 cool tones for ε-greedy
_COLORS_UCB = ["#f9c74f", "#f4845f", "#f3722c", "#c1121f"]
_COLORS_EPS = ["#4cc9f0", "#7b2d8b", "#90e0ef"]


def _assign_colors(results: list[RunResult]) -> list[str]:
    ucb_idx = eps_idx = 0
    colors: list[str] = []
    for r in results:
        if r.label.startswith("UCB"):
            colors.append(_COLORS_UCB[ucb_idx % len(_COLORS_UCB)])
            ucb_idx += 1
        else:
            colors.append(_COLORS_EPS[eps_idx % len(_COLORS_EPS)])
            eps_idx += 1
    return colors


def _save_plot(
    results: list[RunResult],
    colors: list[str],
    y_key: str,  # "avg_rewards" | "pct_optimal"
    y_label: str,
    title: str,
    out_path: str,
) -> None:
    steps = list(range(1, len(results[0].avg_rewards) + 1))

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor(_STYLE["figure.facecolor"])

        for result, color in zip(results, colors):
            series = getattr(result, y_key)
            lw = 1.8 if result.label.startswith("UCB") else 1.4
            ls = "-" if result.label.startswith("UCB") else "--"
            ax.plot(
                steps,
                series,
                label=result.label,
                color=color,
                linewidth=lw,
                linestyle=ls,
                alpha=0.92,
            )

        ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
        ax.set_xlabel("Steps", fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.85)
        ax.grid(True)

        # Minor tick marks for readability
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(4))
        ax.tick_params(which="minor", length=3, color=_STYLE["xtick.color"])

        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    N_RUNS = 2_000
    N_STEPS = 1_000

    # ── Agents ──────────────────────────────────────────────────────────────
    agents: list[Agent] = [
        UCBAgent(c=1),
        UCBAgent(c=2),
        UCBAgent(c=3),
        UCBAgent(c=4),
        EpsilonGreedyAgent(epsilon=0.1),
        EpsilonGreedyAgent(epsilon=0.01),
        EpsilonGreedyAgent(epsilon=0.001),
    ]

    # ── Run experiments ──────────────────────────────────────────────────────
    results: list[RunResult] = []
    total = len(agents)

    print(f"\n10-Armed Testbed  |  {N_RUNS:,} runs × {N_STEPS:,} steps\n")
    print("─" * 52)

    for idx, agent in enumerate(agents, 1):
        print(f"  [{idx}/{total}]  {agent.label} …", end=" ", flush=True)
        result = run_experiment(agent, n_runs=N_RUNS, n_steps=N_STEPS)
        results.append(result)

        final_reward = result.avg_rewards[-1]
        final_optimal = result.pct_optimal[-1]
        print(f"done  |  reward={final_reward:+.4f}  optimal={final_optimal:.1f}%")

    print("─" * 52)
    print()

    # ── Summary table ────────────────────────────────────────────────────────
    col_w = 26
    print(
        f"{'Agent':<{col_w}}  {'Avg Reward (last step)':>22}  {'% Optimal (last step)':>22}"
    )
    print("─" * (col_w + 50))
    for r in results:
        print(
            f"  {r.label:<{col_w - 2}}"
            f"  {r.avg_rewards[-1]:>22.4f}"
            f"  {r.pct_optimal[-1]:>21.1f}%"
        )
    print()

    # ── Output directory ─────────────────────────────────────────────────────
    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    colors = _assign_colors(results)

    print("Saving plots …")
    _save_plot(
        results=results,
        colors=colors,
        y_key="avg_rewards",
        y_label="Average Reward",
        title="10-Armed Testbed — Average Reward: UCB vs ε-Greedy",
        out_path=str(plots_dir / "ucb_vs_egreedy_reward.png"),
    )
    _save_plot(
        results=results,
        colors=colors,
        y_key="pct_optimal",
        y_label="% Optimal Action",
        title="10-Armed Testbed — % Optimal Action: UCB vs ε-Greedy",
        out_path=str(plots_dir / "ucb_vs_egreedy_optimal.png"),
    )
    print("\nAll done ✓")


if __name__ == "__main__":
    main()
