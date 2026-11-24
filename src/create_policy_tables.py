import csv
import os

OUTPUT_DIR = "outputs/part-d"

for seed in [11, 22, 33]:
    for epsilon in [0.2, 0.0]:

        q_file = os.path.join(
            OUTPUT_DIR,
            f"q_table_seed_{seed}_epsilon_{epsilon}.csv"
        )

        with open(q_file, "r", newline="") as file:
            rows = list(csv.DictReader(file))

        for usable_ace in [0, 1]:
            output_file = os.path.join(
                OUTPUT_DIR,
                f"policy_seed_{seed}_epsilon_{epsilon}_usable_ace_{usable_ace}.csv"
            )

            with open(output_file, "w", newline="") as file:
                writer = csv.writer(file)

                writer.writerow([
                    "state",
                    "best_action",
                    "counts"
                ])

                for row in rows:
                    if int(row["usable_ace"]) != usable_ace:
                        continue

                    stick = float(row["q_stick"])
                    hit = float(row["q_hit"])

                    stick_visits = int(row["stick_visits"])
                    hit_visits = int(row["hit_visits"])
                    total_visits = stick_visits + hit_visits

                    if total_visits == 0:
                        action = "UNVISITED"
                    elif stick > hit:
                        action = "STICK"
                    elif hit > stick:
                        action = "HIT"
                    else:
                        action = "TIE"

                    state = row["player_sum"]

                    writer.writerow([
                        state,
                        action,
                        total_visits
                    ])

print("All policy tables created.")