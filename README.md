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

- 27 Apr 2025: (5352175c) inference now 5 times faster than the first commit
- 27 Apr 2025: (8694edda) fixed step size bug, now inference again 5 times faster. It is 25 times faster than the first commit!

## Learn basics

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
C_j = \bigwedge L_j
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
        1 |     25 |  25000 |   25000 |  69.68% |  71.66% |  71.66% |       102s |  102.190s |  102.027s | 0.001781s | 0.161133s | 245.03it/s | 155151.42it/s
        2 |     25 |  25000 |   25000 |  76.22% |  80.52% |  80.52% |       195s |   93.205s |   93.037s | 0.001620s | 0.166739s | 268.71it/s | 149934.84it/s
        3 |     25 |  25000 |   25000 |  81.66% |  81.36% |  81.36% |       280s |   84.235s |   84.059s | 0.001489s | 0.173763s | 297.41it/s | 143874.48it/s
        4 |     25 |  25000 |   25000 |  81.09% |  64.20% |  81.36% |       365s |   85.320s |   85.149s | 0.001613s | 0.169188s | 293.60it/s | 147764.20it/s
        5 |     25 |  25000 |   25000 |  81.63% |  75.91% |  81.36% |       450s |   85.288s |   85.106s | 0.002135s | 0.179902s | 293.75it/s | 138964.88it/s
    epoch | epochs | sample | samples | train   | test    | best    | training   | epoch     | train     | compile   | test      | fit perf   | test perf
        6 |     25 |  25000 |   25000 |  83.86% |  83.84% |  83.84% |       532s |   81.768s |   81.587s | 0.001856s | 0.178881s | 306.42it/s | 139757.44it/s
        7 |     25 |  25000 |   25000 |  84.36% |  80.15% |  83.84% |       612s |   80.050s |   79.868s | 0.001709s | 0.180481s | 313.02it/s | 138518.42it/s
        8 |     25 |  25000 |   25000 |  84.12% |  84.28% |  84.28% |       698s |   85.545s |   85.350s | 0.001709s | 0.192999s | 292.91it/s | 129534.51it/s
        9 |     25 |  25000 |   25000 |  86.17% |  84.57% |  84.57% |       780s |   82.467s |   82.280s | 0.001737s | 0.185772s | 303.84it/s | 134573.88it/s
       10 |     25 |  25000 |   25000 |  85.60% |  83.45% |  84.57% |       863s |   83.081s |   82.882s | 0.001656s | 0.197698s | 301.63it/s | 126455.38it/s
    epoch | epochs | sample | samples | train   | test    | best    | training   | epoch     | train     | compile   | test      | fit perf   | test perf
       11 |     25 |  25000 |   25000 |  86.06% |  84.94% |  84.94% |       948s |   84.435s |   84.211s | 0.002026s | 0.222811s | 296.87it/s | 112202.59it/s
       12 |     25 |  25000 |   25000 |  89.16% |  84.85% |  84.94% |      1025s |   76.931s |   76.703s | 0.003631s | 0.224351s | 325.93it/s | 111432.64it/s
       13 |     25 |  25000 |   25000 |  87.25% |  85.16% |  85.16% |      1106s |   81.581s |   81.379s | 0.001598s | 0.200694s | 307.21it/s | 124567.63it/s
       14 |     25 |  25000 |   25000 |  87.41% |  85.27% |  85.27% |      1188s |   82.326s |   82.127s | 0.001592s | 0.198122s | 304.41it/s | 126184.88it/s
       15 |     25 |  25000 |   25000 |  87.42% |  84.18% |  85.27% |      1269s |   80.504s |   80.317s | 0.001646s | 0.184598s | 311.27it/s | 135429.77it/s
    epoch | epochs | sample | samples | train   | test    | best    | training   | epoch     | train     | compile   | test      | fit perf   | test perf
       16 |     25 |  25000 |   25000 |  89.57% |  85.62% |  85.62% |      1344s |   74.577s |   74.385s | 0.001814s | 0.189711s | 336.09it/s | 131779.28it/s
       17 |     25 |  25000 |   25000 |  88.56% |  85.76% |  85.76% |      1420s |   76.918s |   76.721s | 0.002013s | 0.195322s | 325.86it/s | 127994.09it/s
       18 |     25 |  25000 |   25000 |  87.92% |  85.04% |  85.76% |      1496s |   75.730s |   75.532s | 0.001638s | 0.196561s | 330.99it/s | 127187.03it/s
       19 |     25 |  25000 |   25000 |  88.08% |  84.96% |  85.76% |      1572s |   75.887s |   75.674s | 0.001893s | 0.211503s | 330.37it/s | 118201.87it/s
       20 |     25 |  25000 |   25000 |  88.36% |  85.47% |  85.76% |      1650s |   77.547s |   77.342s | 0.001579s | 0.203653s | 323.24it/s | 122757.73it/s
    ```

Repo will be active until I find a job or will lost hope in TM or figure out how to get money out of this thing. If you want to help - you are welcome, please fork and send PRs (or money). If you need help, feel free to open issue or contact me 😏.

## License
Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution
Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

