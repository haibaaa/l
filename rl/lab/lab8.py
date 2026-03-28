"""
20-Armed Bandit Problem: Comparing exploration strategies.
Demonstrates why Optimistic Greedy (init=40) is optimal for deterministic rewards.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from dataclasses import dataclass, field
from typing import Literal

# ── Reproducibility ───────────────────────────────────────────────────────────
RNG_SEED = 42

# ── Experiment hyper-parameters ───────────────────────────────────────────────
N_ARMS = 20
N_STEPS = 1000
N_TRIALS = 500
REWARD_LOW, REWARD_HIGH = 0, 20  # Uniform(0, 20) deterministic rewards
OPTIMISTIC_INIT = 40  # Above any reachable reward → forces exploration


# ══════════════════════════════════════════════════════════════════════════════
# Bandit environment
# ══════════════════════════════════════════════════════════════════════════════


class Bandit:
    """
    A k-armed bandit with *deterministic* rewards.

    Each arm's true value is sampled once from Uniform(low, high) at
    construction time and never changes during the trial.
    """

    def __init__(
        self,
        n_arms: int = N_ARMS,
        low: float = REWARD_LOW,
        high: float = REWARD_HIGH,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.n_arms = n_arms
        self._rng = rng or np.random.default_rng()
        # True (deterministic) reward for every arm
        self.true_values: np.ndarray = self._rng.uniform(low, high, size=n_arms)
        self.optimal_arm: int = int(np.argmax(self.true_values))

    def pull(self, arm: int) -> float:
        """Return the deterministic reward for *arm*."""
        return float(self.true_values[arm])


# ══════════════════════════════════════════════════════════════════════════════
# Agent
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class Agent:
    """
    Action-value agent that supports both ε-greedy and optimistic-greedy
    strategies via sample-average updates.

    Parameters
    ----------
    n_arms        : Number of bandit arms.
    epsilon       : Exploration rate (0 → pure greedy).
    init_estimate : Initial Q-value for all arms.
    rng           : Optional random generator for reproducibility.
    """

    n_arms: int = N_ARMS
    epsilon: float = 0.0
    init_estimate: float = 0.0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def __post_init__(self) -> None:
        self._q: np.ndarray = np.full(self.n_arms, self.init_estimate, dtype=float)
        self._counts: np.ndarray = np.zeros(self.n_arms, dtype=int)

    # ── action selection ──────────────────────────────────────────────────────

    def select_action(self) -> int:
        """ε-greedy action selection (ε=0 → always greedy)."""
        if self.epsilon > 0 and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_arms))
        # Break ties randomly among maximal arms
        max_q = np.max(self._q)
        best_arms = np.where(self._q == max_q)[0]
        return int(self.rng.choice(best_arms))

    # ── update rule ───────────────────────────────────────────────────────────

    def update(self, arm: int, reward: float) -> None:
        """
        Incremental sample-average update:
            Q(a) ← Q(a) + 1/N(a) · [R − Q(a)]

        Starting from a high init_estimate, each pull *reduces* Q toward the
        true value, so all arms are tried at least once before the agent
        commits — pure forced exploration without a stochastic ε.
        """
        self._counts[arm] += 1
        self._q[arm] += (reward - self._q[arm]) / self._counts[arm]

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def estimates(self) -> np.ndarray:
        return self._q.copy()


# ══════════════════════════════════════════════════════════════════════════════
# Strategy registry
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class StrategyConfig:
    label: str
    epsilon: float
    init_estimate: float
    color: str
    linestyle: str = "-"


STRATEGIES: list[StrategyConfig] = [
    StrategyConfig("ε-greedy (ε=0.10)", epsilon=0.10, init_estimate=0, color="#E63946"),
    StrategyConfig("ε-greedy (ε=0.01)", epsilon=0.01, init_estimate=0, color="#F4A261"),
    StrategyConfig(
        "Pure Greedy (init=0)", epsilon=0.00, init_estimate=0, color="#457B9D"
    ),
    StrategyConfig(
        "Optimistic Greedy (init=40)",
        epsilon=0.00,
        init_estimate=OPTIMISTIC_INIT,
        color="#2A9D8F",
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# Simulation runner
# ══════════════════════════════════════════════════════════════════════════════


def run_experiment(
    strategy: StrategyConfig,
    n_trials: int = N_TRIALS,
    n_steps: int = N_STEPS,
    seed: int = RNG_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run *n_trials* independent trials for a single strategy.

    Returns
    -------
    avg_reward   : shape (n_steps,)  – mean reward at each step
    pct_optimal  : shape (n_steps,)  – fraction of trials that chose the
                                       optimal arm at each step (0–100 %)
    """
    rng = np.random.default_rng(seed)
    rewards = np.zeros((n_trials, n_steps))
    optimal = np.zeros((n_trials, n_steps), dtype=bool)

    for trial in range(n_trials):
        bandit = Bandit(rng=rng)
        agent = Agent(
            n_arms=N_ARMS,
            epsilon=strategy.epsilon,
            init_estimate=strategy.init_estimate,
            rng=rng,
        )

        for step in range(n_steps):
            action = agent.select_action()
            reward = bandit.pull(action)
            agent.update(action, reward)

            rewards[trial, step] = reward
            optimal[trial, step] = action == bandit.optimal_arm

    avg_reward = rewards.mean(axis=0)
    pct_optimal = optimal.mean(axis=0) * 100.0
    return avg_reward, pct_optimal


# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

PLOT_DIR = "plots"
SMOOTHING_WINDOW = 20  # steps – light rolling mean for readability


def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple rolling-mean smoother (valid convolution, then zero-pad front)."""
    kernel = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode="valid")
    pad = np.full(window - 1, smoothed[0])
    return np.concatenate([pad, smoothed])


def _apply_plot_style(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Steps", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=10, framealpha=0.85)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)


def plot_results(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    strategies: list[StrategyConfig],
) -> None:
    os.makedirs(PLOT_DIR, exist_ok=True)
    steps = np.arange(1, N_STEPS + 1)

    # ── Figure 1 : Average Reward ─────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(11, 5))
    for cfg in strategies:
        avg_r, _ = results[cfg.label]
        ax1.plot(
            steps,
            _smooth(avg_r, SMOOTHING_WINDOW),
            color=cfg.color,
            linestyle=cfg.linestyle,
            linewidth=2,
            label=cfg.label,
        )
    _apply_plot_style(
        ax1, "Average Reward vs. Steps (20-Armed Bandit)", "Average Reward"
    )
    fig1.tight_layout()
    path1 = os.path.join(PLOT_DIR, "average_reward.png")
    fig1.savefig(path1, dpi=150)
    print(f"  Saved → {path1}")
    plt.close(fig1)

    # ── Figure 2 : % Optimal Action ───────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(11, 5))
    for cfg in strategies:
        _, pct_opt = results[cfg.label]
        ax2.plot(
            steps,
            _smooth(pct_opt, SMOOTHING_WINDOW),
            color=cfg.color,
            linestyle=cfg.linestyle,
            linewidth=2,
            label=cfg.label,
        )
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
    _apply_plot_style(
        ax2, "% Optimal Action vs. Steps (20-Armed Bandit)", "% Optimal Action"
    )
    fig2.tight_layout()
    path2 = os.path.join(PLOT_DIR, "optimal_action.png")
    fig2.savefig(path2, dpi=150)
    print(f"  Saved → {path2}")
    plt.close(fig2)


# ══════════════════════════════════════════════════════════════════════════════
# Numerical analysis
# ══════════════════════════════════════════════════════════════════════════════


def print_analysis(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    strategies: list[StrategyConfig],
) -> None:
    """Print cumulative rewards and a brief justification table."""
    header = (
        f"\n{'Strategy':<35} {'Cumul. Reward':>15} {'Avg/Step':>10} {'Final %Opt':>12}"
    )
    print(header)
    print("─" * len(header))

    cumulative: dict[str, float] = {}
    for cfg in strategies:
        avg_r, pct_opt = results[cfg.label]
        cum = float(avg_r.sum())
        cumulative[cfg.label] = cum
        print(
            f"{cfg.label:<35} {cum:>15,.2f} {cum/N_STEPS:>10.4f}"
            f" {pct_opt[-SMOOTHING_WINDOW:].mean():>11.1f}%"
        )

    # ── Justify optimistic greedy ─────────────────────────────────────────────
    best_label = max(cumulative, key=cumulative.__getitem__)
    opt_label = "Optimistic Greedy (init=40)"
    print(
        f"\n{'─'*70}\n"
        f"JUSTIFICATION — Why '{opt_label}' is optimal\n"
        f"{'─'*70}\n"
        f"  1. SYSTEMATIC EXPLORATION (no wasted random pulls):\n"
        f"     Starting Q(a)=40 > max possible reward ({REWARD_HIGH}), every arm looks\n"
        f"     'promising' until pulled.  The agent is *forced* to try all {N_ARMS}\n"
        f"     arms before it can commit — no randomness needed.\n\n"
        f"  2. FAST CONVERGENCE to the true best arm:\n"
        f"     After each arm is pulled once, Q(a) drops to its deterministic\n"
        f"     true value.  Because rewards are deterministic, a single pull\n"
        f"     suffices — the agent then greedily exploits the best arm for the\n"
        f"     remaining ~{N_STEPS - N_ARMS} steps, maximising cumulative reward.\n\n"
        f"  3. ZERO WASTED EXPLORATION after the initial sweep:\n"
        f"     ε-greedy continues exploring randomly for *all* {N_STEPS} steps,\n"
        f"     sacrificing reward indefinitely.  Optimistic greedy stops exploring\n"
        f"     as soon as every arm has been tried exactly once (~{N_ARMS} steps).\n\n"
        f"  Best strategy by cumulative reward: '{best_label}'\n"
        f"  Optimistic Greedy cumulative reward: {cumulative[opt_label]:,.2f}\n"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    print(
        f"Running 20-Armed Bandit experiment\n"
        f"  Trials={N_TRIALS}, Steps={N_STEPS}, Arms={N_ARMS}, Seed={RNG_SEED}\n"
    )

    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cfg in STRATEGIES:
        print(f"  Simulating: {cfg.label} …")
        results[cfg.label] = run_experiment(cfg)

    print("\nGenerating plots …")
    plot_results(results, STRATEGIES)

    print_analysis(results, STRATEGIES)


if __name__ == "__main__":
    main()
