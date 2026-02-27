import numpy as np

# States and rewards
states = [0, 1, 2]
rewards = np.array([4, 7, 1])

# Transition matrix
P = {
    "a1": np.array([[0.3, 0.4, 0.3], [0.6, 0.2, 0.2], [0.2, 0.2, 0.6]]),
    "a2": np.array([[0.5, 0.4, 0.1], [0.3, 0.2, 0.5], [0.4, 0.2, 0.4]]),
    "a3": np.array([[0.25, 0.25, 0.5], [0.4, 0.3, 0.3], [0.2, 0.5, 0.3]]),
}

# Policy representation
policy = {0: "a1", 1: "a3", 2: "a2"}


def policy_evaluation(gamma, epsilon=1e-6):
    V = np.zeros(3)
    iterations = 0

    while True:
        delta = 0
        V_new = np.copy(V)

        for s in states:
            a = policy[s]
            V_new[s] = rewards[s] + gamma * (P[a][s] @ V)
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new
        iterations += 1

        if delta < epsilon:
            break

    return V, iterations


# Discount factors.
gammas = [0, 0.1, 0.01, 0.001, 0.3]

for gamma in gammas:
    V, iters = policy_evaluation(gamma)
    print(f"\nγ = {gamma}")
    print(f"V(S0) = {V[0]:.4f}")
    print(f"V(S1) = {V[1]:.4f}")
    print(f"V(S2) = {V[2]:.4f}")
    print(f"took {iters} iterations")
