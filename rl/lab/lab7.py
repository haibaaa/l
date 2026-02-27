import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Dict, List

matplotlib.use("Agg")  # Non-interactive backend (Wayland-safe)


def main():
    from typing import Dict, List

    # -----------------------------
    # Configuration
    # -----------------------------
    n_bandits: int = 2000
    n_arms: int = 10
    n_steps: int = 1000

    epsilons: List[float] = [0.0, 0.2]  # greedy, epsilon-greedy
    initial_values: List[float] = [0.0, 0.7, 2]  # zero init, optimistic init

    np.random.seed(42)

    # -----------------------------
    # Bandit Simulator
    # -----------------------------
    def run_bandit(epsilon: float, init_value: float) -> np.ndarray:
        """
        Run the 10-armed bandit experiment.

        Args:
            epsilon (float): exploration probability
            init_value (float): initial action-value estimate

        Returns:
            np.ndarray: average reward per step (shape: [n_steps])
        """
        avg_rewards: np.ndarray = np.zeros(n_steps, dtype=np.float64)

        for _ in range(n_bandits):
            # true action values q*(a)
            q_true: np.ndarray = np.random.normal(loc=0.0, scale=1.0, size=n_arms)

            # estimated action values q(a)
            Q: np.ndarray = np.full(n_arms, init_value, dtype=np.float64)

            # action counts n(a)
            N: np.ndarray = np.zeros(n_arms, dtype=np.int64)

            for t in range(n_steps):
                action: int
                reward: float

                # ε-greedy action selection
                if np.random.rand() < epsilon:
                    action = int(np.random.randint(n_arms))
                else:
                    action = int(np.argmax(Q))

                # Reward generation
                reward = float(q_true[action] + np.random.normal(0.0, 1.0))

                # Update count
                N[action] += 1

                # Sample-average update
                Q[action] += (reward - Q[action]) / float(N[action])

                # Accumulate rewards
                avg_rewards[t] += reward

        # Average across bandits
        avg_rewards /= float(n_bandits)
        return avg_rewards

    # -----------------------------
    # Run Experiments
    # -----------------------------
    results: Dict[str, np.ndarray] = {}

    for epsilon in epsilons:
        for init in initial_values:
            label: str = f"ε={epsilon}, Q₀={init}"
            results[label] = run_bandit(epsilon, init)

    # -----------------------------
    # Plot & Save (NO DISPLAY)
    # -----------------------------
    plt.figure(figsize=(10, 6))

    for label, rewards in results.items():
        plt.plot(rewards, label=label)

    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title("10-Armed Bandit Testbed (2000 Bandits)")
    plt.legend()
    plt.grid(True)

    plt.savefig("outputs/out.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
