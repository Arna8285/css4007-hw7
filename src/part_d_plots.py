import csv
import os
import matplotlib.pyplot as plt

OUTPUT_DIR = "outputs/part-d"

plt.figure(figsize=(10, 6))

for seed in [11, 22, 33]:
    for epsilon in [0.2, 0.0]:

        filename = os.path.join(
            OUTPUT_DIR,
            f"training_rewards_seed_{seed}_epsilon_{epsilon}.csv"
        )

        games = []
        rewards = []

        with open(filename, "r", newline="") as file:
            reader = csv.DictReader(file)

            for row in reader:
                games.append(int(row["game"]))
                rewards.append(float(row["mean_reward"]))

        plt.plot(
            games,
            rewards,
            label=f"Seed {seed}, epsilon {epsilon}"
        )

plt.xlabel("Training games")
plt.ylabel("Mean reward per 1000 games")
plt.title("Q-learning Training Performance")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "training_rewards.png"),
    dpi=300
)

plt.show()

print("Training plot saved.")