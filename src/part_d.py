
import csv
import os
import json
import gymnasium as gym

SEEDS = [11, 22, 33]
GAME_COUNT = 10000
MAX_STEPS = 100

OUTPUT_DIR = "outputs/part-d"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_environment():
    return gym.make("Blackjack-v1")


def fixed_policy(observation):
    player_sum, dealer_card, usable_ace = observation

    if player_sum < 20:
        return 1

    return 0


def collect_games(seed, number_of_games):
    env = make_environment()
    records = []

    observation, info = env.reset(seed=seed)

    completed_games = 0
    cutoffs = 0
    environment_truncations = 0

    for game_number in range(1, number_of_games + 1):
        step_number = 0
        terminated = False
        truncated = False

        while (
            not terminated
            and not truncated
            and step_number < MAX_STEPS
        ):
            action = fixed_policy(observation)

            next_observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            step_number += 1

            cutoff = (
                step_number >= MAX_STEPS
                and not terminated
                and not truncated
            )

            records.append({
                "seed": seed,
                "game": game_number,
                "step": step_number,
                "observation": observation,
                "action": action,
                "reward": reward,
                "next_observation": next_observation,
                "terminal": terminated,
                "truncated": truncated,
                "cutoff": cutoff
            })

            observation = next_observation

            if cutoff:
                cutoffs += 1

        if terminated:
            completed_games += 1

        if truncated:
            environment_truncations += 1

        observation, info = env.reset()

    env.close()

    return records, completed_games, cutoffs, environment_truncations


def save_games(records, seed):
    filename = f"{OUTPUT_DIR}/games_seed_{seed}.csv"

    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "seed",
                "game",
                "step",
                "observation",
                "action",
                "reward",
                "next_observation",
                "terminal",
                "truncated",
                "cutoff"
            ]
        )

        writer.writeheader()

        for record in records:
            row = record.copy()
            row["observation"] = json.dumps(row["observation"])
            row["next_observation"] = json.dumps(
                row["next_observation"]
            )
            writer.writerow(row)


def main():
    for seed in SEEDS:
        print("\nCollecting games. Seed:", seed)

        (
            records,
            completed_games,
            cutoffs,
            environment_truncations
        ) = collect_games(seed, GAME_COUNT)

        save_games(records, seed)

        print("Games requested:", GAME_COUNT)
        print("Completed games:", completed_games)
        print("Safety cutoffs:", cutoffs)
        print("Environment truncations:", environment_truncations)
        print("Transitions:", len(records))


if __name__ == "__main__":
    main()

