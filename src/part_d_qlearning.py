
import csv
import os
import random
import gymnasium as gym
from collections import defaultdict

GAMMA = 1.0
ALPHA = 0.1
SEEDS = [11, 22, 33]
EPSILONS = [0.2, 0.0]
GAME_COUNT = 50000
MAX_STEPS = 100

OUTPUT_DIR = "outputs/part-d"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def choose_action(state, q_values, epsilon, rng):
    if rng.random() < epsilon:
        return rng.choice([0, 1])

    values = q_values[state]
    best_value = max(values)

    best_actions = [
        action for action in [0, 1]
        if values[action] == best_value
    ]

    return rng.choice(best_actions)


def train_q_learning(seed, epsilon):
    env = gym.make("Blackjack-v1")
    rng = random.Random(seed)

    q_values = defaultdict(lambda: [0.0, 0.0])
    action_counts = defaultdict(lambda: [0, 0])

    block_rewards = []
    block_total_reward = 0.0
    cutoffs = 0

    observation, info = env.reset(seed=seed)

    for game in range(1, GAME_COUNT + 1):
        episode_reward = 0.0
        finished = False

        for step in range(MAX_STEPS):
            state = tuple(observation)

            action = choose_action(
                state, q_values, epsilon, rng
            )

            next_observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            next_state = tuple(next_observation)

            if terminated or truncated:
                target = reward
                finished = True
            else:
                target = (
                    reward
                    + GAMMA * max(q_values[next_state])
                )

            old_value = q_values[state][action]

            q_values[state][action] = (
                old_value + ALPHA * (target - old_value)
            )

            action_counts[state][action] += 1
            episode_reward += reward

            observation = next_observation

            if terminated or truncated:
                break

        if not finished:
            cutoffs += 1

        block_total_reward += episode_reward

        if game % 1000 == 0:
            block_rewards.append({
                "game": game,
                "mean_reward": block_total_reward / 1000
            })

            block_total_reward = 0.0

            print(
                "Seed:", seed,
                "Epsilon:", epsilon,
                "Game:", game,
                "Cutoffs:", cutoffs
            )

        observation, info = env.reset()

    env.close()

    return q_values, action_counts, block_rewards, cutoffs


def save_q_table(seed, epsilon, q_values, action_counts):
    filename = (
        f"{OUTPUT_DIR}/q_table_seed_{seed}_epsilon_{epsilon}.csv"
    )

    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "player_sum",
            "dealer_card",
            "usable_ace",
            "q_stick",
            "q_hit",
            "stick_visits",
            "hit_visits"
        ])

        for state in sorted(q_values):
            writer.writerow([
                *state,
                q_values[state][0],
                q_values[state][1],
                action_counts[state][0],
                action_counts[state][1]
            ])


def save_policy_tables(seed, epsilon, q_values, action_counts):
    for usable_ace in [0, 1]:
        filename = (
            f"{OUTPUT_DIR}/policy_seed_{seed}_epsilon_{epsilon}_"
            f"usable_ace_{usable_ace}.csv"
        )

        with open(filename, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            writer.writerow([
                "player_sum",
                "dealer_card",
                "usable_ace",
                "best_action",
                "stick_visits",
                "hit_visits"
            ])

            states = [
                state for state in q_values
                if state[2] == usable_ace
            ]

            for state in sorted(states):
                counts = action_counts[state]

                if sum(counts) == 0:
                    best_action = "UNVISITED"
                elif q_values[state][0] > q_values[state][1]:
                    best_action = "STICK"
                elif q_values[state][1] > q_values[state][0]:
                    best_action = "HIT"
                else:
                    best_action = "TIE"

                writer.writerow([
                    *state,
                    best_action,
                    counts[0],
                    counts[1]
                ])


def save_training_rewards(seed, epsilon, block_rewards):
    filename = (
        f"{OUTPUT_DIR}/training_rewards_seed_{seed}_epsilon_{epsilon}.csv"
    )

    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(["game", "mean_reward"])

        for item in block_rewards:
            writer.writerow([
                item["game"],
                item["mean_reward"]
            ])


def main():
    for seed in SEEDS:
        for epsilon in EPSILONS:
            print("\n==============================")
            print("Q-learning")
            print("Seed:", seed)
            print("Epsilon:", epsilon)
            print("==============================")

            q_values, counts, rewards, cutoffs = (
                train_q_learning(seed, epsilon)
            )

            save_q_table(seed, epsilon, q_values, counts)
            save_policy_tables(seed, epsilon, q_values, counts)
            save_training_rewards(seed, epsilon, rewards)

            print("States:", len(q_values))
            print("Cutoffs:", cutoffs)
            print("Training completed.")


if __name__ == "__main__":
    main()
