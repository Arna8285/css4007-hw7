
import csv
import os
import random
import gymnasium as gym

SEEDS = [11, 22, 33]
EVAL_SEEDS = [1011, 1022, 1033]
EPSILONS = [0.2, 0.0]

GAME_COUNT = 10000
MAX_STEPS = 100

OUTPUT_DIR = "outputs/part-d"


def load_q_table(seed, epsilon):
    filename = (
        f"{OUTPUT_DIR}/q_table_seed_{seed}_epsilon_{epsilon}.csv"
    )

    q_values = {}

    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            state = (
                int(row["player_sum"]),
                int(row["dealer_card"]),
                int(row["usable_ace"])
            )

            q_values[state] = [
                float(row["q_stick"]),
                float(row["q_hit"])
            ]

    return q_values


def choose_greedy_action(state, q_values, rng):
    values = q_values.get(state, [0.0, 0.0])

    best_value = max(values)

    best_actions = [
        action for action in [0, 1]
        if values[action] == best_value
    ]

    return rng.choice(best_actions)


def evaluate_policy(policy_name, eval_seed, q_values=None):
    env = gym.make("Blackjack-v1")
    rng = random.Random(eval_seed)

    wins = 0
    losses = 0
    draws = 0
    cutoffs = 0
    total_reward = 0.0
    completed_games = 0

    observation, info = env.reset(seed=eval_seed)

    for game in range(GAME_COUNT):
        episode_reward = 0.0
        finished = False

        for step in range(MAX_STEPS):
            state = tuple(observation)

            if policy_name == "baseline":
                action = 1 if state[0] < 20 else 0
            else:
                action = choose_greedy_action(
                    state, q_values, rng
                )

            observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            episode_reward += reward

            if terminated or truncated:
                finished = True
                break

        if finished:
            completed_games += 1
            total_reward += episode_reward

            if episode_reward > 0:
                wins += 1
            elif episode_reward < 0:
                losses += 1
            else:
                draws += 1
        else:
            cutoffs += 1

        observation, info = env.reset()

    env.close()

    mean_reward = (
        total_reward / completed_games
        if completed_games > 0 else 0.0
    )

    return {
        "policy": policy_name,
        "eval_seed": eval_seed,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "completed_games": completed_games,
        "cutoffs": cutoffs,
        "mean_reward": mean_reward
    }


def main():
    results = []

    for seed, eval_seed in zip(SEEDS, EVAL_SEEDS):
        print("\n==============================")
        print("Training seed:", seed)
        print("Evaluation seed:", eval_seed)
        print("==============================")

        baseline_result = evaluate_policy(
            "baseline", eval_seed
        )

        baseline_result["training_seed"] = seed
        results.append(baseline_result)

        print("Baseline:", baseline_result)

        for epsilon in EPSILONS:
            q_values = load_q_table(seed, epsilon)

            result = evaluate_policy(
                "q_learning_epsilon_" + str(epsilon),
                eval_seed,
                q_values
            )

            result["training_seed"] = seed
            results.append(result)

            print("Q-learning epsilon", epsilon, ":", result)

    filename = f"{OUTPUT_DIR}/evaluation_results.csv"

    with open(filename, "w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "training_seed",
            "eval_seed",
            "policy",
            "wins",
            "losses",
            "draws",
            "completed_games",
            "cutoffs",
            "mean_reward"
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\nEvaluation completed.")
    print("Results saved to:", filename)


if __name__ == "__main__":
    main()
