

import csv
from collections import defaultdict

GAMMA = 1.0
ALPHA = 0.1
STATES = ["A", "B", "C", "D", "E"]
ACTIONS = [0, 1]

# Загружаем игровые истории
with open("data/paper-games.csv", "r", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

games = defaultdict(list)

for row in rows:
    games[int(row["game"])].append({
        "state": row["state"],
        "action": int(row["action"]),
        "next_state": row["next_state"],
        "reward": float(row["reward"]),
        "terminated": row["terminated"].lower() == "true"
    })

# Загружаем названия состояний
with open("data/states.csv", "r", encoding="utf-8") as file:
    state_info = list(csv.DictReader(file))

print("PART C — PAPER REPLAY")

# =====================================
# 1. MODEL-BASED LEARNING
# =====================================

print("\n========== MODEL-BASED LEARNING ==========")

counts = defaultdict(int)
reward_sums = defaultdict(float)
state_action_counts = defaultdict(int)

for game in games.values():
    for item in game:
        key = (item["state"], item["action"], item["next_state"])
        counts[key] += 1
        reward_sums[key] += item["reward"]
        state_action_counts[(item["state"], item["action"])] += 1

print("\nTransition counts, probabilities and rewards:")

for key, count in counts.items():
    state, action, next_state = key
    probability = count / state_action_counts[(state, action)]
    average_reward = reward_sums[key] / count

    print(
        state, action, next_state,
        "Count:", count,
        "Probability:", probability,
        "Average reward:", average_reward
    )

# Определяем действия фиксированной политики
policy = {}

for game in games.values():
    for item in game:
        policy[item["state"]] = item["action"]

# Выполняем два синхронных прохода
values = {state: 0.0 for state in STATES}

for sweep in range(1, 3):
    new_values = {}

    for state in STATES:
        action = policy[state]
        total = 0.0

        for key, count in counts.items():
            s, a, next_state = key

            if s == state and a == action:
                probability = count / state_action_counts[(s, a)]
                average_reward = reward_sums[key] / count

                next_value = 0.0 if next_state == "END" else values[next_state]

                total += probability * (
                    average_reward + GAMMA * next_value
                )

        new_values[state] = total

    values = new_values.copy()

    print("\nValue sweep", sweep)
    for state in STATES:
        print(state, values[state])


# =====================================
# 2. MONTE CARLO
# =====================================

print("\n========== MONTE CARLO ==========")

returns = defaultdict(list)

for game_number in sorted(games):
    episode = games[game_number]
    G = 0.0
    game_returns = [0.0] * len(episode)

    # Рассчитываем возвраты с конца игры
    for i in range(len(episode) - 1, -1, -1):
        G = episode[i]["reward"] + GAMMA * G
        game_returns[i] = G

    for i, item in enumerate(episode):
        returns[item["state"]].append(game_returns[i])

        print(
            "Game:", game_number,
            "State:", item["state"],
            "Return:", game_returns[i]
        )

print("\nMonte Carlo final results:")

for state in STATES:
    values_list = returns[state]

    if values_list:
        average = sum(values_list) / len(values_list)
        print(
            state,
            "Count:", len(values_list),
            "Returns:", values_list,
            "Average:", average
        )
    else:
        print(state, "No observations")


# =====================================
# 3. TD(0)
# =====================================

print("\n========== TD(0) ==========")

td_values = {state: 0.0 for state in STATES}

for pass_number in range(1, 3):
    print("\nTD pass", pass_number)

    for game_number in sorted(games):
        for item in games[game_number]:
            state = item["state"]
            next_state = item["next_state"]
            reward = item["reward"]

            old_value = td_values[state]

            next_value = 0.0 if next_state == "END" else td_values[next_state]

            target = reward + GAMMA * next_value
            new_value = old_value + ALPHA * (target - old_value)

            td_values[state] = new_value

            print(
                "Game:", game_number,
                "State:", state,
                "Old:", old_value,
                "Reward:", reward,
                "Next value:", next_value,
                "Target:", target,
                "New:", new_value
            )

print("\nTD(0) final values:")

for state in STATES:
    print(state, td_values[state])


# =====================================
# 4. Q-LEARNING
# =====================================

print("\n========== Q-LEARNING ==========")

q_values = {
    state: {action: 0.0 for action in ACTIONS}
    for state in STATES
}

for pass_number in range(1, 3):
    print("\nQ-learning pass", pass_number)

    for game_number in sorted(games):
        for item in games[game_number]:
            state = item["state"]
            action = item["action"]
            next_state = item["next_state"]
            reward = item["reward"]

            old_q = q_values[state][action]

            if next_state == "END":
                max_next_q = 0.0
            else:
                max_next_q = max(q_values[next_state].values())

            target = reward + GAMMA * max_next_q
            new_q = old_q + ALPHA * (target - old_q)

            q_values[state][action] = new_q

            print(
                "Game:", game_number,
                "State:", state,
                "Action:", action,
                "Old Q:", old_q,
                "Reward:", reward,
                "Max next Q:", max_next_q,
                "Target:", target,
                "New Q:", new_q
            )

print("\nQ-learning final table:")

for state in STATES:
    print(state, q_values[state])

    best_value = max(q_values[state].values())
    best_actions = [
        action for action, value in q_values[state].items()
        if value == best_value
    ]

    print("Greedy action(s):", best_actions)

print("\n========== END OF PART C ==========")
