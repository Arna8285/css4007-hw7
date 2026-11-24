
import csv
import json
import os
from collections import defaultdict

GAMMA = 1.0
ALPHA = 0.1
SEEDS = [11, 22, 33]
CHECKPOINTS = [100, 1000, 10000]
OUTPUT_DIR = "outputs/part-d"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_games(seed):
    filename = f"{OUTPUT_DIR}/games_seed_{seed}.csv"
    episodes = defaultdict(list)

    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            item = {
                "state": tuple(json.loads(row["observation"])),
                "action": int(row["action"]),
                "reward": float(row["reward"]),
                "next_state": (
                    "END" if row["terminal"] == "True"
                    else tuple(json.loads(row["next_observation"]))
                )
            }

            episodes[int(row["game"])].append(item)

    return episodes


def save_values(method, seed, game_count, values, visits):
    filename = (
        f"{OUTPUT_DIR}/{method}_seed_{seed}_games_{game_count}.csv"
    )

    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "player_sum",
            "dealer_card",
            "usable_ace",
            "value",
            "visits"
        ])

        for state in sorted(values):
            writer.writerow([
                *state,
                values[state],
                visits[state]
            ])


def save_action_coverage(seed, game_count, states, pair_visits):
    filename = (
        f"{OUTPUT_DIR}/model_action_coverage_seed_{seed}_games_{game_count}.csv"
    )

    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "player_sum",
            "dealer_card",
            "usable_ace",
            "action",
            "action_name",
            "visits",
            "status"
        ])

        for state in sorted(states):
            for action in [0, 1]:
                count = pair_visits.get((state, action), 0)

                status = "OBSERVED" if count > 0 else "UNOBSERVED"
                action_name = "STICK" if action == 0 else "HIT"

                writer.writerow([
                    *state,
                    action,
                    action_name,
                    count,
                    status
                ])


def model_based_learning(episodes):
    transitions = defaultdict(int)
    reward_sums = defaultdict(float)
    visits = defaultdict(int)
    states = set()

    for episode in episodes.values():
        for item in episode:
            state = item["state"]
            action = item["action"]
            next_state = item["next_state"]

            states.add(state)

            if next_state != "END":
                states.add(next_state)

            key = (state, action)

            transitions[(state, action, next_state)] += 1
            reward_sums[key] += item["reward"]
            visits[key] += 1

    probabilities = {}

    for (state, action, next_state), count in transitions.items():
        probabilities[(state, action, next_state)] = (
            count / visits[(state, action)]
        )

    values = {state: 0.0 for state in states}
    converged = False
    sweeps = 0

    for sweep in range(10000):
        new_values = {}

        for state in states:
            action = 1 if state[0] < 20 else 0
            key = (state, action)

            if visits[key] == 0:
                new_values[state] = 0.0
                continue

            average_reward = reward_sums[key] / visits[key]
            expected_value = 0.0

            for (s, a, next_state), probability in probabilities.items():
                if s == state and a == action:
                    if next_state != "END":
                        expected_value += (
                            probability * values[next_state]
                        )

            new_values[state] = (
                average_reward + GAMMA * expected_value
            )

        difference = max(
            abs(new_values[state] - values[state])
            for state in states
        )

        values = new_values
        sweeps = sweep + 1

        if difference < 0.000001:
            converged = True
            break

    state_visits = defaultdict(int)

    for (state, action), count in visits.items():
        state_visits[state] += count

    return values, state_visits, visits, states, converged, sweeps


def monte_carlo_learning(episodes):
    returns_sum = defaultdict(float)
    visits = defaultdict(int)
    values = {}
    game_count = 0

    for game_number in sorted(episodes):
        episode = episodes[game_number]
        total_return = 0.0
        episode_returns = [0.0] * len(episode)

        for i in range(len(episode) - 1, -1, -1):
            total_return = (
                episode[i]["reward"] + GAMMA * total_return
            )
            episode_returns[i] = total_return

        for i, item in enumerate(episode):
            state = item["state"]

            returns_sum[state] += episode_returns[i]
            visits[state] += 1

            values[state] = returns_sum[state] / visits[state]

        game_count += 1

        if game_count in CHECKPOINTS:
            save_values(
                "mc", current_seed, game_count, values, visits
            )

    return values, visits


def td_learning(episodes):
    values = defaultdict(float)
    visits = defaultdict(int)
    game_count = 0

    for game_number in sorted(episodes):
        for item in episodes[game_number]:
            state = item["state"]
            next_state = item["next_state"]
            reward = item["reward"]

            if next_state == "END":
                next_value = 0.0
            else:
                next_value = values[next_state]

            target = reward + GAMMA * next_value

            values[state] += ALPHA * (target - values[state])
            visits[state] += 1

        game_count += 1

        if game_count in CHECKPOINTS:
            save_values(
                "td", current_seed, game_count, values, visits
            )

    return values, visits


def main():
    global current_seed

    for seed in SEEDS:
        current_seed = seed

        print("\n================================")
        print("Seed:", seed)
        print("================================")

        episodes = load_games(seed)

        print("Loaded games:", len(episodes))

        print("\nModel-Based Learning")

        for checkpoint in CHECKPOINTS:
            selected_episodes = dict(
                list(sorted(episodes.items()))[:checkpoint]
            )

            (
                model_values,
                state_visits,
                pair_visits,
                states,
                converged,
                sweeps
            ) = model_based_learning(selected_episodes)

            save_values(
                "model",
                seed,
                checkpoint,
                model_values,
                state_visits
            )

            save_action_coverage(
                seed,
                checkpoint,
                states,
                pair_visits
            )

            print(
                f"Games: {checkpoint}, "
                f"States: {len(model_values)}, "
                f"Visited pairs: {len(pair_visits)}, "
                f"Converged: {converged}, "
                f"Sweeps: {sweeps}"
            )

        print("\nMonte Carlo Learning")

        mc_values, mc_visits = monte_carlo_learning(episodes)

        print("States:", len(mc_values))
        print("Total visits:", sum(mc_visits.values()))

        print("\nTD(0) Learning")

        td_values, td_visits = td_learning(episodes)

        print("States:", len(td_values))
        print("Total visits:", sum(td_visits.values()))

    print("\nAll three methods completed.")


if __name__ == "__main__":
    main()
