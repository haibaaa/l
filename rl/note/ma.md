# Study Note: Multi-Armed Bandit (MAB) & Reinforcement Learning

This note breaks down the provided Python code and the core concepts of the Multi-Armed Bandit problem to help you prepare for your viva.

---

## 1. What is the Multi-Armed Bandit Problem?
The MAB is a classic Reinforcement Learning (RL) puzzle that exemplifies the **Exploration vs. Exploitation trade-off**.
* **The Scenario:** You are in front of 10 slot machines ("one-armed bandits"). Each has a different probability of paying out. 
* **The Goal:** Maximize your total reward over a specific number of turns (steps).
* **The Dilemma:** * **Exploitation:** Pull the arm that has given you the best reward so far.
    * **Exploration:** Pull a different arm to see if it might be even better than your current favorite.

---

## 2. Key Code Components & Variables

### The "Testbed" Setup
* `n_bandits = 2000`: To get statistically significant results, the code runs the entire experiment 2,000 times and averages the results.
* `n_arms = 10`: There are 10 different actions the agent can take.
* `q_true`: The **actual** mean reward for each arm (hidden from the agent). These are drawn from a Normal distribution (Mean=0, SD=1).

### The Agent's Internal Logic
* `Q`: The agent’s **estimate** of how good each arm is. It starts at `init_value` and updates over time.
* `N`: A counter tracking how many times each specific arm has been pulled.

---

## 3. Action Selection Strategies (The "How")

The code compares two main strategies for choosing an arm:

### A. $\epsilon$-Greedy (Epsilon-Greedy)
Controlled by the `epsilon` variable:
* **If $rand < \epsilon$:** The agent **Explores** (picks a random arm).
* **Otherwise:** The agent **Exploits** (picks the arm with the highest `Q` value using `np.argmax(Q)`).
* *In the code:* $\epsilon=0.1$ means the agent explores 10% of the time. $\epsilon=0.0$ is "Pure Greedy."

### B. Optimistic Initial Values
Controlled by `init_value`:
* Standard agents start with `Q = 0.0`.
* An **Optimistic** agent starts with `Q = 5.0`.
* **Why?** Since the true rewards are around 0, a starting value of 5.0 is "disappointingly high." When the agent pulls an arm and gets a reward of ~0, it updates `Q` downward. This forces the agent to try **every** arm at least once because the unexplored arms (still at 5.0) look better than the ones already tried. It's a clever way to force exploration.

---

## 4. The Mathematical Update Rule
The core of the learning happens in this line:
`Q[action] += (reward - Q[action]) / float(N[action])`

This is the **Incremental Update Rule**. It calculates the new average reward for an arm without storing every previous reward:
$$NewEstimate = OldEstimate + \frac{1}{n} [Target - OldEstimate]$$
* **Target:** The reward just received.
* **Error:** `(reward - Q[action])`.
* **Step Size:** `1 / N`. As we pull an arm more often, we become more "certain," and new rewards change our estimate less.

---

## 5. Potential Viva Questions

**Q: Why do we use 2000 bandits instead of just 1?**
**A:** Because RL involves randomness (stochasticity). One single run might be lucky or unlucky. Averaging over 2000 runs shows the true performance of the *strategy* rather than just luck.

**Q: What is the downside of a Pure Greedy ($\epsilon=0$) strategy?**
**A:** It often gets stuck on a sub-optimal arm. If the first arm it pulls gives a reward of 0.5, and all other arms are currently 0.0, it will never try the other arms—even if another arm has a true mean of 2.0.

**Q: What is the benefit of the Optimistic Initial Value ($Q_0=5$)?**
**A:** it encourages early exploration. Even with $\epsilon=0$, the agent will explore all options because it starts with high expectations that are gradually "corrected" downward until it finds the best arm.

**Q: What happens to the Reward vs. Steps graph over time?**
**A:** The average reward should increase and then plateaus. The $\epsilon$-greedy approach will usually converge faster but might stay slightly below the "perfect" reward because it continues to explore (waste turns) even after finding the best arm.
