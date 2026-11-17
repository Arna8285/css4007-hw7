# HW7 — Learn Blackjack four ways

**CSS-4007 · Artificial Intelligence · Fall 2026**

Fork [ardakshalkar/css4007-hw7](https://github.com/ardakshalkar/css4007-hw7), keep your fork public as in earlier homeworks, and work in your fork. Do not open a pull request to the starter repository.

Implement model-based learning, direct evaluation (Monte Carlo), TD(0) and
Q-learning in a new setting: **Blackjack**.

**Work in order: paper calculations → exact reproduction in code → simulated games.**
Complete the paper work for all four methods before implementing the learners.
Save a scan/photo and commit it before your first learning implementation.
Keep the original calculations visible when correcting them.

**Outcome:** LO-07 — Formulate problems as MDPs and implement value- and policy-based RL.

**Recorded total:** 4 marks. **Recorded course weight:** 4%.
Deadline, destination for handing in the fork URL, rubric and mark allocation remain unspecified.
The starter provides the assignment, supplied histories and a blank submission template; you write the learning programs.

## The game

Use Gymnasium **`Blackjack-v1`, `natural=False`, `sab=False`** and record your
installed version. This is a simulation, with no money or betting.

- Try to beat the dealer without exceeding 21.
- Face cards count as 10. An ace counts as 11 when possible without busting,
  otherwise as 1; other cards have their numerical values.
- **Hit** draws another card; **stick** ends your turn. The dealer then draws
  until reaching at least 17, and the game is resolved.
- Cards are drawn with replacement. Observe only **(player sum, dealer showing
  card, usable ace)**. A usable ace is currently counted as 11. The dealer's
  hidden card is not observed.
- Rewards are **+1 win, -1 loss, 0 draw**. A nonterminal hit gives zero reward.
  The specified settings give no natural-blackjack bonus.
- Gymnasium actions: **0 = stick, 1 = hit**.

Use **discount 1.0**, since the objective is the final game result, and
**learning rate 0.1** for TD and Q-learning. Initialize value/Q tables to zero.
For the first three methods, evaluate this fixed policy: **hit below 20;
stick at 20 or 21**. This is a baseline, not a claim of optimal play.

First explain states, actions, rewards and termination in your own words.
Why must the usable-ace flag be part of the state?

## Part A — How many states and actions?

Answer on paper. Show formulas and reasoning, not just final numbers. For
each game distinguish **full game state**, **the player's observation**,
**all action labels** and **actions legal in a particular position**. Include
whose turn it is when needed. Label each count as exact for the stated
representation, an upper bound, or a count of only part of the state.

### 1. Blackjack

Consider only decision observations with player sums **12–21**, dealer
showing cards **1–10**, and the usable-ace flag **yes/no**.

- How many observation entries does this table contain? Show the product.
- How many action labels are there? How many Q-table entries are needed for
  this restricted representation?
- Does this count include decision observations below 12 or the terminal
  marker? Explain why it is not a count of the entire environment's state space.
- Give two different underlying card situations with the same observation.
  Why should the learner not receive the dealer's hidden card?

### 2. Durak

For this question use **two-player, 36-card, non-transfer Durak**, ranks 6
through ace. Define an elementary move as playing **one card**, picking up
when defending, or ending an attack when permitted. Do not count a sequence
of several card plays as one action.

- Before any cards are exposed, how many different unordered six-card hands
  can one player receive? Use a combination and explain why order is ignored.
  Is a hand alone enough to describe a game state?
- You know your six cards and one exposed trump card outside your hand. The
  opponent has six cards. With no other information about their hand, how many
  candidate opponent hands are consistent with these observations? Explain
  why this still does not count the full hidden states: the order of the
  remaining stock can affect future draws. Ignore extra information that
  first-attacker selection might reveal for this counting question.
- On an empty table, the attacker must open with one card. How many legal
  opening actions do they have with six distinct cards?
- **Concrete defense:** hearts are trumps; the attack is **10♣**. Your hand is
  **6♣, 9♣, A♣, 7♥, Q♥, 8♠**. List every legal defense card and the pick-up
  option, then count the legal actions. A defense must use a higher card of
  the same suit, or a trump against a non-trump attack. Transfers are excluded.
- Describe what a full simulator state must store and what the player can
  observe: hands, attack/defense pairs, trump, stock, discarded cards, phase
  and whose turn it is. Can the same visible table conceal different opponent
  hands and different future outcomes?

Do not give an unsupported single number for all reachable Durak states.
Your hand counts are components of that larger problem.

### 3. Tic-tac-toe

Use a 3×3 board, X moves first, players alternate, and play stops at the first
win or full board.

- Each square is empty, X or O. Count all board encodings. Is that also the
  exact number of legally reachable positions? Give two reasons for your answer.
- How many action labels can an implementation use? For the position below,
  determine whose turn it is, list legal placements and count them.
- Can board contents determine the turn in a legal game under these rules?
  Would multiplying every board encoding by two necessarily count legal states?

```text
X O X
. O .
. . .
```

### 4. A small Nim game

There are **three labeled piles**, initially containing **2, 3 and 4 stones**.
Players alternate. A move chooses one nonempty pile and removes any positive
number of stones from it. Taking the last stone wins; the empty position is
terminal.

- Count pile configurations within the starting limits. Then count encodings
  that also record one of two players to move. Explain why this product is
  an encoding count, without claiming that every encoding is reachable from
  the specified initial position and starting player.
- At `(2,3,4)`, list actions as **(pile, number removed)** and count them.
  Repeat for `(0,1,2)` and the terminal position.
- Why is “choose a pile” alone an incomplete action description? How would
  treating piles as unlabeled change what you count?

Finish with a comparison table: **game, representation, state-count formula,
type of count, action representation, legal-action count in the example,
hidden information**. Which games would need an especially large Q-table?
Explain why a count of action labels need not equal the number legal now.

In your submission include this paper work. No Durak, tic-tac-toe or Nim
implementation is required; the programming task remains Blackjack.

## Part B — Calculate all four methods on paper

These are **four constructed game histories** consistent with the rules.
They are illustrative samples, not representative game probabilities.
The dealer shows **6** throughout.

| Label | Player sum | Dealer shows | Usable ace? |
| --- | ---: | ---: | --- |
| A | 18 | 6 | no |
| B | 20 | 6 | no |
| C | 17 | 6 | yes |
| D | 17 | 6 | no |
| E | 21 | 6 | no |

`END` is a terminal marker with zero continuation value, not a playable state.
At C, drawing a ten makes the ace count as 1: the total stays 17 but the
usable-ace flag changes.

| Game | Step | State | Action | Next state | Reward | What happened |
| --- | --- | --- | --- | --- | ---: | --- |
| 1 | 1 | A | hit | B | 0 | Drew 2 |
| 1 | 2 | B | stick | END | +1 | Dealer finishes at 19 |
| 2 | 1 | A | hit | END | -1 | Drew 5 and busted |
| 3 | 1 | C | hit | D | 0 | Drew a ten; ace now counts as 1 |
| 3 | 2 | D | hit | E | 0 | Drew 4 |
| 3 | 3 | E | stick | END | 0 | Dealer finishes at 21: draw |
| 4 | 1 | A | hit | B | 0 | Drew 2 |
| 4 | 2 | B | stick | END | -1 | Dealer finishes at 21 |

For each method write the rule, substitute numbers, show intermediate
calculations and finish with a table. A final table without working is insufficient.

1. **Model-based learning:** Count outcomes for every observed state-action
   pair. Estimate transition probabilities and mean rewards for each observed
   `(state, action, next state)` triple. Include both observed outcomes of
   sticking at B in its mean reward. Use this learned model and the fixed
   policy to calculate **two synchronous policy-evaluation sweeps** from
   `V0 = 0`, showing V1 and V2 for A–E. Use only the previous sweep's values;
   do not maximize over actions. Mark other pairs unobserved. Explain why
   sample probabilities need not match true game probabilities.
2. **Direct evaluation:** Calculate the return from every nonterminal visit
   until that game ends. Use every-visit Monte Carlo: average returns for each
   state and show sample counts. Process all four games once.
3. **TD(0):** Process games 1–4 in order, updating immediately after each
   transition. Show old value, reward, successor value, target and new value
   for **every update**. Retain values between games. Replay games 1–4 once
   more without resetting and show both passes. Terminal continuation is zero.
4. **Q-learning:** Start with a fresh zero Q-table and replay the same four
   games twice. Show old Q-value, reward, maximum successor Q-value, target
   and new Q-value for every update. Maximize over **both hit and stick**,
   including unobserved actions initialized to zero. Terminal continuation is
   zero. Finish with both Q-values at A–E and identify greedy actions/ties.
   Why does an untried action's zero estimate not establish that it is good?

Explain why the four final tables need not agree after this limited experience
and different amounts of processing. Prescribed replay checks updates; it does
not test action selection or exploration.

## Part C — Reproduce the paper calculations in the system

Implement the algorithms yourself and load **exactly the supplied histories**
as fixed input. The starter supplies them in `data/paper-games.csv`, with state
labels in `data/states.csv`. Do not randomly deal replacement histories in Gymnasium.

- Match parameters, initialization, game order, two model-based sweeps, one
  Monte Carlo pass and two TD/Q-learning passes.
- Print learned counts/probabilities/rewards, both value sweeps, Monte Carlo
  returns/counts/averages, every TD/Q update and the final tables.
- Compare **method, state/state-action, sweep/pass/update, paper result,
  program result and difference**. Keep full precision internally.
- Explain discrepancies and preserve the original work beside corrections.
- Provide one command for this reproduction. Save outputs in `outputs/paper-replay/`.

## Part D — Learn from simulated games

Use Gymnasium to deal and resolve new games. Write the four learning methods
yourself; the environment library supplies the game. No GPU or API key is needed.
Learners receive observations and observed rewards/transitions, and must not
read hidden cards or the environment's internal state.

### Evaluate a fixed policy

For each seed **11, 22 and 33**, collect **10,000 games** using hit-below-20.
Seed the environment at the first reset of each run; subsequent resets continue
its random sequence. Log seed, game, step, observation, action, reward, next
observation and terminal flag. Convert terminal transitions to `END` for the
learners even when Gymnasium returns an observation at game end.

Feed the **same saved games in the same order** to the learned model with
fixed-policy evaluation, every-visit Monte Carlo and TD(0). Save estimates
after **100, 1,000 and 10,000 games**. For model-based evaluation use tolerance
**0.000001** and a maximum of **10,000 synchronous sweeps**; report nonconvergence.
Keep unobserved pairs marked as such, explain their treatment and identify
values affected by missing successor information. Do not fill gaps from hidden
game rules or present initialized zeros as observed estimates.

Show state-value tables by player sum and dealer showing card, **separately
for usable ace and no usable ace**, with visit counts and missing estimates.
Compare methods at several frequently visited and rarely visited states and
explain how estimates change with more games. No exact value reference is supplied.

### Learn actions with Q-learning

Train fresh tables for **50,000 games per seed**, using seeds **11, 22 and 33**,
learning rate **0.1**, discount **1.0** and epsilon **0.2**. Exploration chooses
uniformly between the two actions. Break greedy ties randomly using a seeded
action generator. Save Q-values, state-action visit counts and greedy hit/stick
tables, separated by usable-ace flag.

Repeat with **epsilon 0**, keeping other settings unchanged. This removes the
exploration branch; random tie-breaking can still vary actions.

Freeze each trained table. Evaluate it and the fixed-policy baseline for
**10,000 new games each**, using evaluation seeds **1011, 1022 and 1033** paired
with the corresponding training seeds. Restart the evaluation environment
with the same seed for each policy; different actions still consume random
cards differently. Evaluation uses greedy actions, no exploratory branch and
no updates. Record wins, losses, draws and mean reward per seed. Plot training
mean reward in blocks of **1,000 games** and distinguish training from evaluation.

Use a **100-player-action safety limit**. Games normally end by sticking or
busting. Record unexpected cut-offs separately, investigate them, and report
any excluded games. A cut-off is not a win/loss/draw or a complete Monte Carlo return.

Explain which methods learn a model, state values or action values; which
evaluate a supplied policy versus choose actions; how terminal transitions
change updates; how usable aces affect decisions; and how exploration and
limited visits affect results. Did Q-learning improve on the baseline? Use
all seed results, including failures; finite training does not guarantee improvement.

## What to hand in

- Scans/photos of original paper calculations for all four methods, saved
  before learning-code implementation, plus clearly marked corrections.
- Paper answers and the comparison table for states/actions in Blackjack,
  Durak, tic-tac-toe and Nim.
- Fixed-history replay program, saved output and paper-versus-program table.
- Your learning code, dependency versions and commands to reproduce experiments.
- `SUBMISSION.md` with comparisons, explanations and AI-use declaration.
- `outputs/` with game logs, summaries, visit counts, value/Q tables,
  hit/stick policy tables and plots.

Follow existing course guidance on AI use. Collaboration and submission
arrangements remain for the professor to specify.

## References

- [Berkeley CS188 Lecture 10 — Reinforcement Learning](https://inst.eecs.berkeley.edu/~cs188/fa26/assets/lectures/cs188-fa26-lec10.pdf).
- [Berkeley CS188 Lecture 11 — Reinforcement Learning II](https://inst.eecs.berkeley.edu/~cs188/fa26/assets/lectures/cs188-fa26-lec11.pdf).
- [Gymnasium Blackjack rules, observations and settings](https://gymnasium.farama.org/environments/toy_text/blackjack/).
- [Durak rules reference](https://www.pagat.com/beating/podkidnoy_durak.html); use the specific variant and elementary moves stated in the counting question.



