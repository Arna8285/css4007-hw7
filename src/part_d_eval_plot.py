
import csv
import os
import matplotlib.pyplot as plt

OUTPUT_DIR = "outputs/part-d"

labels = [
    "Baseline",
    "Q-learning epsilon 0.2",
    "Q-learning epsilon 0.0"
]

rewards = {
    "Baseline": [],
    "Q-learning epsilon 0.2": [],
    "Q-learning epsilon 0.0": []
}

filename = os.path.join(
    OUTPUT_DIR,
    "evaluation_results.csv"
)

with open(filename, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        policy = row["policy"]

        if policy == "baseline":
            rewards["Baseline"].append(
                float(row["mean_reward"])
            )

        elif policy == "q_learning_epsilon_0.2":
            rewards["Q-learning epsilon 0.2"].append(
                float(row["mean_reward"])
            )

        elif policy == "q_learning_epsilon_0.0":
            rewards["Q-learning epsilon 0.0"].append(
                float(row["mean_reward"])
            )

averages = [
    sum(rewards[label]) / len(rewards[label])
    for label in labels
]

plt.figure(figsize=(9, 5))
plt.bar(labels, averages)

plt.xlabel("Policy")
plt.ylabel("Average reward")
plt.title("Q-learning Evaluation Results")
plt.grid(axis="y")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "evaluation_rewards.png"),
    dpi=300
)

plt.show()

print("Evaluation plot saved.")
print("Average rewards:", averages)
