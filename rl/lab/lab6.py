import numpy as np

# -----------------------------
# 1. define states and actions
# -----------------------------
states = [0, 1, 2, 3, 4]  # S1–S5
actions = [0, 1, 2, 3, 4, 5]  # a1–a6

n_states = len(states)
n_actions = len(actions)
gamma: float = 0.1


# random reward function
rewards = np.random.randint(1, 10, size=n_states)


# ------------------------------------------------
# 2. generate dataset of (s, a, s') samples
# ------------------------------------------------
def generate_samples(num_samples):
    data = []
    for _ in range(num_samples):
        s = np.random.choice(states)
        a = np.random.choice(actions)
        s_next = np.random.choice(states)
        data.append((s, a, s_next))
    return data


# ------------------------------------------------
# 3. Estimate transition probabilities
# ------------------------------------------------
def estimate_transition_matrix(data):
    counts = np.zeros((n_states, n_actions, n_states))
    sa_counts = np.zeros((n_states, n_actions))

    # Count occurrences
    for s, a, s_next in data:
        counts[s][a][s_next] += 1
        sa_counts[s][a] += 1

    # Convert counts to probabilities
    P = np.zeros((n_states, n_actions, n_states))

    for s in states:
        for a in actions:
            if sa_counts[s][a] > 0:
                P[s][a] = counts[s][a] / sa_counts[s][a]

    return P


# ------------------------------------------------
# 4. Compute value function for a random policy
# ------------------------------------------------
def compute_value(P):
    V = np.zeros(n_states)

    # Random fixed policy
    policy = np.random.choice(actions, size=n_states)

    for _ in range(1000):  # iterative policy evaluation
        V_new = np.copy(V)
        for s in states:
            a = policy[s]
            V_new[s] = rewards[s] + gamma * np.dot(P[s][a], V)
        if np.max(np.abs(V_new - V)) < 1e-6:
            break
        V = V_new

    return V


# ==================================================
# (i) Using 1000 samples
# ==================================================
data_1000 = generate_samples(1000)
P_1000 = estimate_transition_matrix(data_1000)
V_1000 = compute_value(P_1000)

print("Value V(S1) with 1000 samples:", V_1000[0])


# ==================================================
# (ii) Using 10^5 samples
# ==================================================
data_100k = generate_samples(100000)
P_100k = estimate_transition_matrix(data_100k)
V_100k = compute_value(P_100k)

print("Value V(S1) with 10^5 samples:", V_100k[0])
