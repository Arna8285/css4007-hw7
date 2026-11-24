# HW7 Submission — Blackjack Reinforcement Learning

## AI-use declaration

ChatGPT was used to help understand the assignment requirements, explain reinforcement learning algorithms, debug Python code, and improve project documentation.

The implementation was tested locally. The supplied game histories were used as fixed input for Part C. The simulated experiments in Part D were generated using Gymnasium.

## Setup and reproduction

- **Language:** Python
- **Environment:** Windows PowerShell
- **Python:** 3.13.2
- **Gymnasium:** 1.4.0
- **NumPy:** 2.5.5
- **Matplotlib:** 3.11.2
- **Repository:** https://github.com/Arna8285/css4007-hw7

### Project structure

- `paper-work/PART-A.pdf` — original Part A calculations.
- `paper-work/PART-B.pdf` — original Part B calculations.
- `data/paper-games.csv` — fixed game histories.
- `data/states.csv` — state descriptions.
- `src/paper_replay.py` — Part C implementation.
- `src/part_d.py` — Part D fixed-policy game collection.
- `src/part_d_learning.py` — Model-Based, Monte Carlo and TD(0).
- `src/part_d_qlearning.py` — Q-learning training.
- `src/part_d_evaluation.py` — policy evaluation.
- `src/part_d_plots.py` — training plot generation.
- `src/part_d_eval_plot.py` — evaluation plot generation.
- `outputs/paper-replay/paper-replay.txt` — Part C output.
- `outputs/part-d/` — Part D logs, tables, results and plots.

### Reproduction commands

Run commands from the project root.

Part C:

```powershell
python .\src\paper_replay.py
```

Part D — collect games:

```powershell
python .\src\part_d.py
```

Fixed-policy learning:

```powershell
python .\src\part_d_learning.py
```

Q-learning:

```powershell
python .\src\part_d_qlearning.py
```

Evaluation:

```powershell
python .\src\part_d_evaluation.py
```

Generate plots:

```powershell
python .\src\part_d_plots.py
python .\src\part_d_eval_plot.py
```

## Part A — Count states and actions

Original calculations: `paper-work/PART-A.pdf`.

The paper calculations cover Blackjack, Durak, tic-tac-toe and Nim. The original work is preserved in the PDF file.

## Part B — Paper calculations

Original calculations: `paper-work/PART-B.pdf`.

The original paper work was committed before implementing the Part C learners. It is preserved separately from the Python implementation.

## Part C — Reproduce the calculations

The fixed-history replay output is saved in `outputs/paper-replay/paper-replay.txt`.

The implementation is available in `src/paper_replay.py`.

The program uses exactly the supplied histories from `data/paper-games.csv` and state labels from `data/states.csv`.

### Parameters

- Discount factor: gamma = 1.0
- TD(0) learning rate: alpha = 0.1
- Q-learning learning rate: alpha = 0.1
- Initial state values: 0
- Initial Q-values: 0

### Results

| Method | State / action | Sweep / pass | Program result |
|---|---|---|---:|
| Model-Based | A | Sweep 1 | -0.333333 |
| Model-Based | A | Sweep 2 | -0.333333 |
| Model-Based | B | Sweep 1 and 2 | 0 |
| Monte Carlo | A | Final average | -0.333333 |
| Monte Carlo | B | Final average | 0 |
| TD(0) | A | Pass 2 final | -0.140030 |
| TD(0) | B | Pass 2 final | -0.018100 |
| Q-learning | A, action 1 | Pass 2 final | -0.139220 |
| Q-learning | B, action 0 | Pass 2 final | -0.018100 |

The model-based estimate for state B is zero because the two observed terminal rewards, +1 and -1, average to zero.

Monte Carlo uses returns from every visit. TD(0) updates values immediately after each transition. Q-learning updates action values using the maximum successor Q-value.

The program keeps full precision internally. The paper-result comparison and any differences should be checked against the original Part B calculations in `paper-work/PART-B.pdf`.

## Part D — Simulated Blackjack games

### 1. Fixed-policy game collection

New Blackjack games were generated using Gymnasium.

The fixed policy was:

- Hit when the player's sum is below 20.
- Stick when the player's sum is 20 or higher.

Three random seeds were used: 11, 22 and 33.

Each seed generated 10,000 games.

| Seed | Requested games | Completed games | Safety cutoffs | Environment truncations | Transitions |
|---:|---:|---:|---:|---:|---:|
| 11 | 10,000 | 10,000 | 0 | 0 | 17,958 |
| 22 | 10,000 | 10,000 | 0 | 0 | 18,036 |
| 33 | 10,000 | 10,000 | 0 | 0 | 17,912 |
| **Total** | **30,000** | **30,000** | **0** | **0** | **53,906** |

All requested games were completed. No safety cutoffs or environment truncations occurred.

The saved game logs were reused as input for Model-Based, Monte Carlo and TD(0), ensuring that the three methods used the same game histories.

### 2. Fixed-policy value estimation

The following methods were implemented:

- **Model-Based:** estimates state values using observed transition counts, probabilities and rewards.
- **Monte Carlo:** estimates values using complete episode returns.
- **TD(0):** updates estimates after each transition using the next state's value.

The algorithms were evaluated after 100, 1,000 and 10,000 games for each seed.

#### Model-Based convergence

| Seed | Games | Observed states | Observed policy pairs | Converged | Sweeps |
|---:|---:|---:|---:|---|---:|
| 11 | 100 | 96 | 96 | Yes | 6 |
| 11 | 1,000 | 243 | 243 | Yes | 12 |
| 11 | 10,000 | 280 | 280 | Yes | 11 |
| 22 | 100 | 116 | 116 | Yes | 6 |
| 22 | 1,000 | 245 | 245 | Yes | 10 |
| 22 | 10,000 | 280 | 280 | Yes | 11 |
| 33 | 100 | 118 | 118 | Yes | 5 |
| 33 | 1,000 | 247 | 247 | Yes | 11 |
| 33 | 10,000 | 280 | 280 | Yes | 11 |

The Model-Based method converged in every reported experiment using a tolerance of 1e-6.

At 10,000 games, Monte Carlo and TD(0) each processed all observed transitions:

| Seed | Monte Carlo visits | TD(0) visits |
|---:|---:|---:|
| 11 | 17,958 | 17,958 |
| 22 | 18,036 | 18,036 |
| 33 | 17,912 | 17,912 |

#### State coverage and missing actions

The value tables include player sum, dealer card and usable-ace status.

At 10,000 games, 280 policy state-action pairs were observed for each seed. The alternative actions were not observed under the fixed policy.

Unobserved state-action pairs are marked as `UNOBSERVED` in the action coverage files. Their values cannot be reliably estimated from these fixed-policy histories because the alternative actions were not taken.

This is an important limitation of policy evaluation: the data describes the selected policy, not every possible action.

#### Common and rare states

For seed 11 after 10,000 games:

| State (player sum, dealer card, usable ace) | Method | Estimated value | Visits |
|---|---|---:|---:|
| (20, 10, 0) | Model-Based | 0.415842 | 606 |
| (20, 10, 0) | Monte Carlo | 0.415842 | 606 |
| (20, 10, 0) | TD(0) | 0.439296 | 606 |
| (4, 6, 0) | Model-Based | -0.692830 | 1 |
| (4, 6, 0) | Monte Carlo | -1.000000 | 1 |
| (4, 6, 0) | TD(0) | 0.000000 | 1 |
| (21, 10, 1) | Model-Based | 0.947368 | 190 |
| (21, 10, 1) | Monte Carlo | 0.947368 | 190 |
| (21, 10, 1) | TD(0) | 0.980968 | 190 |

Common states have more visits and generally more stable estimates. Rare states have very few observations, so estimates can differ considerably between methods.

The usable-ace flag is part of the state representation. States with and without a usable ace are treated separately because they can have different outcomes and values.

### 3. Q-learning

Q-learning was trained using new Gymnasium Blackjack games.

Parameters:

- Training seeds: 11, 22 and 33.
- Games per training run: 50,000.
- Learning rate: alpha = 0.1.
- Discount factor: gamma = 1.0.
- Exploration rates: epsilon = 0.2 and epsilon = 0.0.
- Maximum episode length: 100 actions.
- Q-tables were initialized to zero.
- Each training run used a fresh Q-table.

For epsilon = 0.2, the agent sometimes selected a random action to explore. For epsilon = 0.0, the agent selected greedy actions.

Random tie-breaking was used when actions had equal Q-values.

The Q-tables, action counts, training reward logs and learned HIT/STICK policy tables were saved in `outputs/part-d/`.

The policy tables are separated by usable-ace status and include information about unvisited states.

### 4. Evaluation against the baseline

Each learned policy was evaluated on 10,000 new games. The evaluation seeds were 1011, 1022 and 1033.

The baseline and learned policies used the same evaluation seed in each comparison. During evaluation, Q-learning did not explore or update its Q-table.

| Training seed | Evaluation seed | Policy | Wins | Losses | Draws | Completed | Cutoffs | Mean reward |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 11 | 1011 | Baseline | 3,001 | 6,462 | 537 | 10,000 | 0 | -0.3461 |
| 11 | 1011 | Q-learning, epsilon 0.2 | 4,270 | 4,809 | 921 | 10,000 | 0 | -0.0539 |
| 11 | 1011 | Q-learning, epsilon 0.0 | 4,187 | 4,851 | 962 | 10,000 | 0 | -0.0664 |
| 22 | 1022 | Baseline | 2,965 | 6,481 | 554 | 10,000 | 0 | -0.3516 |
| 22 | 1022 | Q-learning, epsilon 0.2 | 4,230 | 4,907 | 863 | 10,000 | 0 | -0.0677 |
| 22 | 1022 | Q-learning, epsilon 0.0 | 4,223 | 4,895 | 882 | 10,000 | 0 | -0.0672 |
| 33 | 1033 | Baseline | 2,984 | 6,496 | 520 | 10,000 | 0 | -0.3512 |
| 33 | 1033 | Q-learning, epsilon 0.2 | 4,200 | 4,974 | 826 | 10,000 | 0 | -0.0774 |
| 33 | 1033 | Q-learning, epsilon 0.0 | 4,133 | 5,034 | 833 | 10,000 | 0 | -0.0901 |

#### Average evaluation results

| Policy | Average mean reward |
|---|---:|
| Baseline | -0.3496 |
| Q-learning, epsilon 0.2 | -0.0663 |
| Q-learning, epsilon 0.0 | -0.0746 |

Both learned policies performed better than the baseline in these evaluation experiments. The epsilon 0.2 policy had a slightly better average reward than the epsilon 0.0 policy.

All evaluation runs completed 10,000 games with zero cutoffs.

### 5. Plots and saved outputs

The following plots were generated:

- `outputs/part-d/training_rewards.png` — training mean reward in blocks of 1,000 games.
- `outputs/part-d/evaluation_rewards.png` — comparison of average evaluation rewards.

The training plot shows training performance. The evaluation plot shows performance on separate evaluation games. These results are reported separately.

Other saved outputs include:

- Game logs for all three fixed-policy seeds.
- Model-Based value tables and action coverage tables.
- Monte Carlo value tables.
- TD(0) value tables.
- Q-tables for each training seed and epsilon.
- HIT/STICK policy tables for usable-ace and no-usable-ace states.
- Training reward CSV files.
- Evaluation results in `evaluation_results.csv`.

### 6. Explanation of results

#### Prediction versus control

Model-Based, Monte Carlo and TD(0) estimate the value of a given fixed policy. They are prediction methods.

Q-learning is a control method. It learns action values and uses them to select actions that aim to maximize future reward.

#### Terminal updates

When a game ends, the terminal reward is used directly. No value is bootstrapped from a terminal state.

The 100-action safety limit prevents an episode from running indefinitely. Cutoff games are counted separately and are not included in completed-game win/loss/draw statistics.

#### Usable ace

A usable ace can count as either 1 or 11 without immediately causing a bust. Therefore, usable-ace status is included in the state representation.

This allows the algorithms to learn different values for states with and without a usable ace.

#### Exploration and exploitation

Epsilon-greedy Q-learning balances exploration and exploitation.

With epsilon = 0.2, the agent sometimes explores random actions. With epsilon = 0.0, it always selects a greedy action, apart from random tie-breaking.

Exploration can help the agent discover better actions. However, the final policy is evaluated without exploration.

#### State and action coverage

More games generally lead to more observed states and visits. Common states have more reliable estimates than rare states.

Fixed-policy data does not contain the alternative action in every state. Therefore, missing action values must not be treated as observed estimates.

#### Differences between random seeds

The three seeds produced slightly different game histories and evaluation rewards. This is expected because the games are stochastic.

However, the learned policies performed better than the baseline for all three evaluation seeds.

#### Did Q-learning improve the baseline?

Yes. The baseline average mean reward was -0.3496. Q-learning achieved -0.0663 with epsilon = 0.2 and -0.0746 with epsilon = 0.0.

Both learned policies improved the average evaluation reward in these experiments.

## Conclusion

Part C reproduced the supplied fixed-history calculations using Model-Based, Monte Carlo, TD(0) and Q-learning methods.

Part D collected 30,000 fixed-policy games, evaluated state values using three prediction methods, trained Q-learning policies and compared them against a baseline on separate evaluation games.

The experiments show that state coverage affects value estimates and that the learned Q-learning policies performed better than the baseline in the tested evaluation runs.

