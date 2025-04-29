# tm2.zig
Binary Tsetlin machine implementation in Zig. Currently, very slow training, fast inference performance. Covered by tests. CPU only, yet.

## Tsetlin machine
In short, Tsetlin machine is a type of machine learning model. It is energy efficient, does not use floating point computations, bitwise, losslessly quantizable down to 1 bit per weight, lossy compressible up to how many important clauses you want to keep, weights are interpretable (if your inputs are interpretable), it does not overfit on small dataset, it does not require your task to be differentiable, have linear time complexity... God knows what else can it do.

## This work
This implementation uses different probabilities in Type I(a), Type I(b) and Type II feedbacks, same as in [Julia multi-class TM implementation](https://github.com/BooBSD/Tsetlin.jl).

As in Julia implementation, this implementation is multithreaded and uses lock-free Hogwild! approach to update its weights, with per-sample parallelization.

It differs from other implementations by using one array, thus being simple to understand, see i.e. [./examples/01-vectorized.py](./examples/01-vectorized.py) for a **less than 62 lines of code** reference implementation in NumPy (both training and inference).

Automata states in this binary TM represented as a 1D view of a 4D array of signed integers, where first dimension is clause polarity $C^{+}$ and $C^{-}$, second dimension is a clause index, third dimension is original or negated literal, and fourth dimension is a feature index.

So, the shape is `(2, n_clauses, 2, n_features)`, where `n_clauses` is a number of clauses per polarity, and `n_features` is a number of features. For multi-class Tsetlin machine, shape would be `(n_classes, 2, n_clauses, 2, n_features)`, but it is out of scope for this repo.

Same shapes for compiled models. Data type of compiled state is a binary mask of whether state action is include or exclude. Included states are non-negative integers (`states >= 0`), excluded states are negative integers (`states < 0`). This work uses signed automata states, so state action threshold is hardcoded at 0.

Unlike others implementation, this implementation does not use any floating point computations other than logging metrics.

## Changes

- 27 Apr 2025: (5352175c) inference now 5 times faster than the first commit.
- 27 Apr 2025: (8694edda) fixed step size bug, now inference again 5 times faster. It is 25 times faster than the first commit!
- 28 Apr 2025: New ablated resource allocation - twice as fast converging to the same accuracy. 17.5% performance improvement per epoch.
- 29 Apr 2025: 26% training time improvement.

## Learn

### Inference

Feature $x_i$ is an input bit at position $i$ in the input vector $X$. It can be either 0 or 1. We have $n_{\text{features}}$ features.

A literal $l_i$ is a feature $x_i$ itself:

$$
l_i = x_i
$$

This way, literal set $L=\{x_0, x_1, \ldots, x_{n_{\text{features}}}\}$ has $n_{\text{literals}} = n_{\text{features}}$ literals.

We can extend literal definition to also have negation of the feature:

$$
l_{i + n_{\text{features}}} = \neg x_i
$$

This way, literal set $L=\{x_0, x_1, \ldots, x_{n_{\text{features}}}, \neg x_0, \neg x_1, \ldots, \neg x_{n_{\text{features}}}\}$ has $n_{\text{literals}} = 2 \cdot n_{\text{features}}$ literals.

> [!IMPORTANT]
> Tsetlin machine does not know about existence of features - it operates on literals. So it does not care whether the feature was negated or not, nor during inference, nor during training.

Literal subset $L_j$ is a subset of literal set $L$ we defined above, i.e.:

$$
L_j = \{l_{42}, l_{69}, l_{1337}\}
$$

Literal subsets are learnable model's parameters. Each of them can be stored either as an index set of included literals, or as a binary mask of included literals. Example: the set `{0, 2}` will be the same as binary mask `[1, 0, 1, 0, 0, 0, 0, 0]` for 8 elements.

Clause $C_j$ is a logical AND of a literal subset $L_j$:

$$
C_j = \bigwedge L_j = \bigwedge_{l_i \in L_j} l_i
$$

Clause is a definition of a single neuron. It fires when all literals in the literal subset are true.

> [!TIP]
> We can omit feature negation in literal definition. This halves number of literals and model size, but clauses may not learn some patterns, i.e. when we want to teach clause to fire when our first feature $x_0$ is 0.

Binary Tsetlin machine defines positive $C^{+}$ and negative $C^{-}$ polarity clauses. Positive polarity clauses votes for target value $y$ being $1$, negative polarity clauses votes for target value $y$ being $0$:

$$
\text{votes} = \sum C^{+} - \sum C^{-}
$$

$$
\hat{y} = \text{votes} \geq 0
$$

That is about binary Tsetlin machine inference.

> [!NOTE]
> Binary Tsetlin machine inference has:
> - time complexity: $O(2 \times n_{\text{clauses per polarity}} \times n_{\text{literals}})$;
> - memory complexity: $O(1)$ extra.
> - storage requirements: `2 * n_clauses * 2 * n_features` bits when implemented using binary masks, or less in other implementations.

### Training

To train literal subset $L_j$, we define automata $A_j$ as a signed integer array of size $n_{\text{literals}}$ with values in full integer range, i.e. $[-2^{31}, 2^{31}-1]$ for 32-bit integers, and threshold $T$ being $0$.

Literal $l_i$ is included in the literal subset $L_j$ if $A_j[i] \geq 0$, and excluded if $A_j[i] < 0$.

This document describes the mechanisms of **Type I** and **Type II Feedback** used in Tsetlin Automata. These feedback mechanisms reinforce or penalize clause actions based on the outputs of the automata and the desired target label $y$.

#### Type I Feedback

Type I feedback is given stochastically to clauses with:
- Positive polarity when $y = 1$
- Negative polarity when $y = 0$

An affected clause reinforces each of its Tsetlin Automata based on:
1. The clause output $C_j(X)$,
2. The action of the targeted Tsetlin Automaton (Include or Exclude),
3. The value of the literal $l_k$ assigned to the automaton.

The two rules governing Type I feedback are as follows:
- Type 1(a) (Recognize): **Include is rewarded** and **Exclude is penalized** with probability $\frac{s-1}{s}$ when $C_j(X) = 1$ and $l_k = 1$. 
  > This creates strong reinforcement, enabling the clause to remember and refine the pattern it recognizes in $X$.
- Type 1(b) (Erase): **Include is penalized** and **Exclude is rewarded** with probability $\frac{1}{s}$ when $C_j(X) = 0$ or $l_k = 0$.
  > This results in weak reinforcement, making infrequent patterns more common.

Here, $s$ is a hyperparameter that controls the frequency of patterns produced.

##### Type I Feedback Table

| State Action    | Clause $C_j(X)$ | Literal $l_k$ | P(Reward)       | P(Penalty)      |
|-----------------|-----------------|---------------|-----------------|-----------------|
| Include Literal | $1$             | $1$           | $\frac{s-1}{s}$ | $0$             |
|                 | $1$             | $0$           | NA              | NA              |
|                 | $0$             | $1$           | $0$             | $\frac{1}{s}$   |
|                 | $0$             | $0$           | $0$             | $\frac{1}{s}$   |
| Exclude Literal | $1$             | $1$           | $0$             | $\frac{s-1}{s}$ |
|                 | $1$             | $0$           | $\frac{1}{s}$   | $0$             |
|                 | $0$             | $1$           | $\frac{1}{s}$   | $0$             |
|                 | $0$             | $0$           | $\frac{1}{s}$   | $0$             |

#### Type II Feedback

Also known as **Reject**,

Type II feedback is given stochastically to clauses with:
- Positive polarity when $y = 0$
- Negative polarity when $y = 1$

An affected clause reinforces each of its Tsetlin Automata based on:
1. The clause output $C_j(X)$,
2. The action of the targeted Tsetlin Automaton (Include or Exclude),
3. The value of the literal $l_k$ assigned to the automaton.

Type II feedback penalizes **Exclude** when $C_j(X) = 1$ and $l_k = 0$.
> This feedback is strong and produces candidate literals for discriminating between $y = 0$ and $y = 1$.

##### Type II Feedback Table

| State Action    | Clause $C_j(X)$ | Literal $l_k$ | P(Reward) | P(Penalty) |
|-----------------|-----------------|---------------|-----------|------------|
| Include Literal | $1$             | $1$           | $0$       | $0$        |
|                 | $1$             | $0$           | NA        | NA         |
|                 | $0$             | $1$           | $0$       | $0$        |
|                 | $0$             | $0$           | $0$       | $0$        |
| Exclude Literal | $1$             | $1$           | $0$       | $0$        |
|                 | $1$             | $0$           | $0$       | $1.0$      |
|                 | $0$             | $1$           | $0$       | $0$        |
|                 | $0$             | $0$           | $0$       | $0$        |

### Resource allocation

We do not want our network to overfit on very similar samples that are prevail in the dataset. This way target value $T$ for $\text{votes}$ was introduced. It means that sample that got votes larger than this threshold will not be used for training in current step.

Original resource allocation defines it as a gradual per-clause update skip:
```zig
// ...in the fit() after counting votes:
const p_clause_update: f32 = @as(f32, @floatFromInt(std.math.clamp(if (target) -votes else votes, -t, t))) / (@as(f32, @floatFromInt(2 * t))) + 0.5;
// feedback loop:
for (0..2) |i_polarity| for (0..n_clauses) |i_clause| if (random.float(f32) < p_clause_update) {
    // give feedback to this automata clause...
};
// return prediction...
```
It can be ablated this to a simple per-sample update skip:
```zig
// ...in the fit() after counting votes:
if (target == true and votes < t or target == false and votes >= -t)) {
    // feedback loop...
}
// return prediction...
```
which does not use floating point computations, division and PRNGs. It gives twice as early converging to the same accuracy, while being a bit slower or equal in an average performance per epoch. This check is an extreme case of the original resource allocation.

## Usage
### IMDB classification
I have achieved 86%+ accuracy (no parameter tuning was performed) at 2.44 MiB compiled state with the following parameters:
```
S: i8, t: 8, r: 11990383208106557440 (0.65), n_features: 40000, n_clauses: 128, state size: 20480000 (19.5MiB)
```
1. Obtain IMDBTrainingData.txt and IMDBTestData.txt using `python3 produce_dataset.py`
2. Convert them to IMDBTrainingData.bin and IMDBTestData.bin using `python3 tobin.py`
3. Run the example:
    ```terminal
    ➜  tm2.zig git:(main) ✗ zig build run -freference-trace --release=fast
    S: i8, t: 8, r: 11990383208106557440 (0.65), n_features: 40000, n_clauses: 128, state size: 20480000 (19.5MiB)
    epoch | epochs | sample | samples | train   | test    | best    | training   | epoch     | train     | compile   | test      | fit perf   | test perf
        1 |     25 |  25000 |   25000 |  68.99% |  78.95% |  78.95% |        74s |   74.438s |   74.255s | 0.001659s | 0.181313s | 336.68it/s | 137882.97it/s
        2 |     25 |  25000 |   25000 |  77.87% |  85.39% |  85.39% |       139s |   64.586s |   64.412s | 0.001735s | 0.172430s | 388.13it/s | 144986.52it/s
        3 |     25 |  25000 |   25000 |  80.66% |  85.98% |  85.98% |       200s |   60.647s |   60.478s | 0.001613s | 0.167126s | 413.37it/s | 149587.73it/s
        4 |     25 |  25000 |   25000 |  83.95% |  84.76% |  85.98% |       256s |   55.918s |   55.735s | 0.001850s | 0.180727s | 448.55it/s | 138330.23it/s
        5 |     25 |  25000 |   25000 |  85.97% |  85.27% |  85.98% |       310s |   54.755s |   54.568s | 0.001785s | 0.185771s | 458.14it/s | 134574.20it/s
    epoch | epochs | sample | samples | train   | test    | best    | training   | epoch     | train     | compile   | test      | fit perf   | test perf
        6 |     25 |  25000 |   25000 |  86.15% |  70.50% |  85.98% |       365s |   54.661s |   54.483s | 0.001654s | 0.176689s | 458.86it/s | 141491.67it/s
        7 |     25 |  25000 |   25000 |  86.66% |  86.49% |  86.49% |       418s |   52.979s |   52.793s | 0.001709s | 0.183478s | 473.54it/s | 136256.47it/s
        8 |     25 |  25000 |   25000 |  84.91% |  85.42% |  86.49% |       472s |   54.434s |   54.249s | 0.001200s | 0.183858s | 460.84it/s | 135974.40it/s
        9 |     25 |  25000 |   25000 |  85.48% |  85.96% |  86.49% |       527s |   54.167s |   53.981s | 0.001630s | 0.185002s | 463.13it/s | 135133.56it/s
       10 |     25 |  25000 |   25000 |  88.19% |  85.87% |  86.49% |       576s |   49.836s |   49.650s | 0.001574s | 0.183966s | 503.52it/s | 135894.62it/s
    epoch | epochs | sample | samples | train   | test    | best    | training   | epoch     | train     | compile   | test      | fit perf   | test perf
       11 |     25 |  25000 |   25000 |  88.26% |  82.08% |  86.49% |       628s |   51.844s |   51.656s | 0.001591s | 0.186086s | 483.97it/s | 134346.55it/s
       12 |     25 |  25000 |   25000 |  87.38% |  66.83% |  86.49% |       681s |   52.395s |   52.212s | 0.001619s | 0.180818s | 478.81it/s | 138260.95it/s
       13 |     25 |  25000 |   25000 |  86.43% |  85.16% |  86.49% |       733s |   52.486s |   52.300s | 0.001567s | 0.183764s | 478.01it/s | 136044.12it/s
       14 |     25 |  25000 |   25000 |  86.72% |  86.34% |  86.49% |       786s |   53.021s |   52.822s | 0.001498s | 0.197703s | 473.29it/s | 126452.54it/s
       15 |     25 |  25000 |   25000 |  87.18% |  86.75% |  86.75% |       840s |   54.150s |   53.958s | 0.001717s | 0.190355s | 463.32it/s | 131333.55it/s
    epoch | epochs | sample | samples | train   | test    | best    | training   | epoch     | train     | compile   | test      | fit perf   | test perf
       16 |     25 |  25000 |   25000 |  87.50% |  85.34% |  86.75% |       892s |   51.848s |   51.660s | 0.001786s | 0.186407s | 483.94it/s | 134115.40it/s
       17 |     25 |  25000 |   25000 |  86.81% |  84.94% |  86.75% |       944s |   51.369s |   51.181s | 0.001192s | 0.186753s | 488.46it/s | 133866.55it/s
       18 |     25 |  25000 |   25000 |  88.38% |  85.35% |  86.75% |       994s |   50.551s |   50.352s | 0.001758s | 0.197861s | 496.51it/s | 126351.39it/s
       19 |     25 |  25000 |   25000 |  86.47% |  87.06% |  87.06% |      1050s |   55.918s |   55.719s | 0.001745s | 0.197174s | 448.68it/s | 126791.69it/s
       20 |     25 |  25000 |   25000 |  89.39% |  86.41% |  87.06% |      1100s |   50.347s |   50.126s | 0.004063s | 0.216902s | 498.74it/s | 115259.48it/s
    ```

Repo will be active until I find a job or will lost hope in TM or figure out how to get money out of this thing. If you want to help - you are welcome, please fork and send PRs (or money). If you need help, feel free to open issue or contact me 😏.

## License
Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution
Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

