
import csv
from collections import defaultdict

# Основные параметры
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

# Загружаем информацию о состояниях
with open("data/states.csv", "r", encoding="utf-8") as file:
    state_info = list(csv.DictReader(file))

state_labels = {row["label"] for row in state_info}

# Проверяем данные
for game in games.values():
    for item in game:
        if item["state"] not in STATES:
            raise ValueError("Unknown state: " + item["state"])

        if item["next_state"] != "END" and item["next_state"] not in STATES:
            raise ValueError("Unknown next state: " + item["next_state"])

        if item["action"] not in ACTIONS:
            raise ValueError("Unknown action: " + str(item["action"]))

        if item["terminated"] != (item["next_state"] == "END"):
            raise ValueError("Terminal flag does not match next state")

if state_labels != set(STATES):
    raise ValueError("State labels do not match STATES")

print("PART C - PAPER REPLAY")
print("Gamma:", GAMMA)
print("Alpha:", ALPHA)
print("Games:", len(games))


# 1. MODEL-BASED LEARNING

print("\n========== MODEL-BASED LEARNING ==========")

counts = defaultdict(int)
reward_sums = defaultdict(float)
state_action_counts = defaultdict(int)

# Считаем переходы и награды
for game in games.values():
    for item in game:
        key = (
            item["state"],
            item["action"],
            item["next_state"]
        )

        counts[key] += 1
        reward_sums[key] += item["reward"]

        state_action_counts[
            (item["state"], item["action"])
        ] += 1

# Выводим переходы
print("\nTransition counts, probabilities and rewards:")

for key, count in counts.items():
    state, action, next_state = key

    probability = count / state_action_counts[(state, action)]
    average_reward = reward_sums[key] / count

    print(
        "State:", state,
        "Action:", action,
        "Next state:", next_state,
        "Count:", count,
        "Probability:", probability,
        "Average reward:", average_reward
    )

# Показываем наблюдаемые и ненаблюдаемые действия
print("\nObserved and unobserved state-action pairs:")

for state in STATES:
    for action in ACTIONS:
        count = state_action_counts[(state, action)]

        if count > 0:
            print(
                "State:", state,
                "Action:", action,
                "Observed:", count
            )
        else:
            print(
                "State:", state,
                "Action:", action,
                "Unobserved"
            )

# Определяем фиксированную политику
policy = {}

for game in games.values():
    for item in game:
        state = item["state"]
        action = item["action"]

        if state in policy and policy[state] != action:
            raise ValueError(
                "Different policy actions found for state " + state
            )

        policy[state] = action

print("\nFixed policy:")

for state in STATES:
    if state in policy:
        print(state, "Action:", policy[state])
    else:
        print(state, "No observed policy action")

# Выполняем два синхронных прохода
values = {state: 0.0 for state in STATES}

for sweep in range(1, 3):
    new_values = {}

    for state in STATES:
        if state not in policy:
            new_values[state] = 0.0
            continue

        action = policy[state]
        total = 0.0

        for key, count in counts.items():
            s, a, next_state = key

            if s == state and a == action:
                probability = count / state_action_counts[(s, a)]
                average_reward = reward_sums[key] / count

                if next_state == "END":
                    next_value = 0.0
                else:
                    next_value = values[next_state]

                total += probability * (
                    average_reward + GAMMA * next_value
                )

        new_values[state] = total

    # Обновляем значения одновременно
    values = new_values.copy()

    print("\nValue sweep", sweep)

    for state in STATES:
        print(state, values[state])


# 2. MONTE CARLO


print("\n========== MONTE CARLO ==========")

returns = defaultdict(list)

# Обрабатываем все игры по порядку
for game_number in sorted(games):
    episode = games[game_number]

    G = 0.0
    game_returns = [0.0] * len(episode)

    # Рассчитываем возвраты с конца игры
    for i in range(len(episode) - 1, -1, -1):
        G = episode[i]["reward"] + GAMMA * G
        game_returns[i] = G

    # Сохраняем возвраты каждого посещения
    for i, item in enumerate(episode):
        returns[item["state"]].append(game_returns[i])

        print(
            "Game:", game_number,
            "State:", item["state"],
            "Return:", game_returns[i]
        )

# Выводим итоговые средние значения
print("\nMonte Carlo final results:")

for state in STATES:
    values_list = returns[state]

    if values_list:
        average = sum(values_list) / len(values_list)

        print(
            "State:", state,
            "Count:", len(values_list),
            "Returns:", values_list,
            "Average:", average
        )
    else:
        print("State:", state, "No observations")



# 3. TD(0)

print("\n========== TD(0) ==========")

td_values = {state: 0.0 for state in STATES}

# Выполняем два прохода по всем играм
for pass_number in range(1, 3):
    print("\nTD pass", pass_number)

    for game_number in sorted(games):
        print("\nGame:", game_number)

        for item in games[game_number]:
            state = item["state"]
            next_state = item["next_state"]
            reward = item["reward"]

            old_value = td_values[state]

            # Для терминального состояния продолжение равно нулю
            if item["terminated"]:
                next_value = 0.0
            else:
                next_value = td_values[next_state]

            target = reward + GAMMA * next_value

            new_value = old_value + ALPHA * (
                target - old_value
            )

            # Обновляем значение сразу
            td_values[state] = new_value

            print(
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



# 4. Q-LEARNING


print("\n========== Q-LEARNING ==========")

# Инициализируем все Q-значения нулями
q_values = {
    state: {
        action: 0.0
        for action in ACTIONS
    }
    for state in STATES
}

# Выполняем два прохода по всем играм
for pass_number in range(1, 3):
    print("\nQ-learning pass", pass_number)

    for game_number in sorted(games):
        print("\nGame:", game_number)

        for item in games[game_number]:
            state = item["state"]
            action = item["action"]
            next_state = item["next_state"]
            reward = item["reward"]

            old_q = q_values[state][action]

            # Для терминального состояния продолжение равно нулю
            if item["terminated"]:
                max_next_q = 0.0
            else:
                max_next_q = max(
                    q_values[next_state].values()
                )

            target = reward + GAMMA * max_next_q

            new_q = old_q + ALPHA * (
                target - old_q
            )

            # Обновляем Q-значение сразу
            q_values[state][action] = new_q

            print(
                "State:", state,
                "Action:", action,
                "Old Q:", old_q,
                "Reward:", reward,
                "Max next Q:", max_next_q,
                "Target:", target,
                "New Q:", new_q
            )

# Выводим итоговую таблицу
print("\nQ-learning final table:")

for state in STATES:
    print("\nState:", state)
    print("Q-values:", q_values[state])

    best_value = max(q_values[state].values())

    best_actions = [
        action
        for action, value in q_values[state].items()
        if value == best_value
    ]

    print("Greedy action(s):", best_actions)

print("\n========== END OF PART C ==========")
