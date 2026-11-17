# HW7 submission — Blackjack

## AI-use declaration

Which tools did you use, and for what?

## Setup and reproduction

Record dependency versions, installation steps and exact commands for the
paper replay, training and frozen-policy evaluation.

## Part A — Count states and actions

Link your paper answers and comparison table for Blackjack, Durak, tic-tac-toe
and Nim. Show formulas, assumptions, example legal actions and hidden information.

## Part B — Paper calculations

Link scans/photos of all four methods. Commit the original paper work before
implementing learners. Keep originals alongside any corrections.

## Part C — Reproduce the calculations

Link the fixed-history replay output in outputs/paper-replay/.

| Method | State / action | Sweep / pass / update | Paper result | Program result | Difference | Explanation |
| --- | --- | --- | --- | --- | --- | --- |

## Part D — Simulated games

### Fixed-policy value estimates

Link the shared logs, checkpoint value tables and visit counts. Explain
missing states and differences among model-based, Monte Carlo and TD estimates.

### Q-learning and baseline evaluation

| Training seed | Policy / epsilon | Evaluation seed | Wins | Losses | Draws | Cut-offs | Mean reward |
| --- | --- | --- | --- | --- | --- | --- | --- |

### Plots and learned policies

Link training curves, Q-tables, hit/stick policy tables and state-action counts.
Separate usable-ace and no-usable-ace results.

### Explain your results

Address prediction versus control, terminal updates, usable aces, exploration,
coverage, variation across seeds and whether the learned policy improved.
