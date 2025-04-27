# tm2.zig
Binary Tsetlin machine implementation in Zig. Currently, very slow training, moderate inference. Covered by tests. CPU only, yet.

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

- 27 Apr 2025: inference now 5 times faster that first commit

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
        1 |     25 |  25000 |   25000 |  68.82% |  77.45% |  77.45% |       104s |  104.491s |  103.766s | 0.001715s | 0.723566s | 240.93it/s | 34551.11it/s
        2 |     25 |  25000 |   25000 |  78.04% |  73.89% |  77.45% |       201s |   96.649s |   95.901s | 0.001783s | 0.745952s | 260.69it/s | 33514.22it/s
        3 |     25 |  25000 |   25000 |  79.07% |  82.97% |  82.97% |       297s |   95.380s |   94.590s | 0.001535s | 0.788503s | 264.30it/s | 31705.66it/s
        4 |     25 |  25000 |   25000 |  80.88% |  73.80% |  82.97% |       389s |   92.390s |   91.588s | 0.001610s | 0.800240s | 272.96it/s | 31240.61it/s
        5 |     25 |  25000 |   25000 |  82.09% |  83.40% |  83.40% |       483s |   93.628s |   92.791s | 0.001796s | 0.835171s | 269.42it/s | 29934.01it/s
    epoch | epochs | sample | samples | train   | test    | best    | training   | epoch     | train     | compile   | test      | fit perf   | test perf
        6 |     25 |  25000 |   25000 |  84.10% |  75.24% |  83.40% |       573s |   90.667s |   89.825s | 0.001642s | 0.840211s | 278.32it/s | 29754.43it/s
        7 |     25 |  25000 |   25000 |  85.21% |  84.38% |  84.38% |       664s |   90.982s |   90.023s | 0.001615s | 0.957751s | 277.71it/s | 26102.81it/s
        8 |     25 |  25000 |   25000 |  85.21% |  83.57% |  84.38% |       758s |   94.306s |   93.400s | 0.001569s | 0.904129s | 267.66it/s | 27650.91it/s
        9 |     25 |  25000 |   25000 |  87.36% |  84.58% |  84.58% |       847s |   88.494s |   87.505s | 0.002583s | 0.986472s | 285.70it/s | 25342.85it/s
       10 |     25 |  25000 |   25000 |  86.87% |  84.77% |  84.77% |       936s |   89.350s |   88.412s | 0.001738s | 0.936849s | 282.77it/s | 26685.21it/s
    epoch | epochs | sample | samples | train   | test    | best    | training   | epoch     | train     | compile   | test      | fit perf   | test perf
       11 |     25 |  25000 |   25000 |  86.80% |  84.43% |  84.77% |      1026s |   89.872s |   88.923s | 0.001586s | 0.947215s | 281.14it/s | 26393.16it/s
       12 |     25 |  25000 |   25000 |  86.28% |  84.80% |  84.80% |      1116s |   89.979s |   89.062s | 0.001646s | 0.915271s | 280.70it/s | 27314.31it/s
       13 |     25 |  25000 |   25000 |  87.68% |  84.64% |  84.80% |      1204s |   87.994s |   87.077s | 0.001986s | 0.914821s | 287.10it/s | 27327.76it/s
       14 |     25 |  25000 |   25000 |  87.84% |  83.62% |  84.80% |      1292s |   87.493s |   86.573s | 0.003639s | 0.916811s | 288.77it/s | 27268.43it/s
       15 |     25 |  25000 |   25000 |  87.66% |  84.68% |  84.80% |      1378s |   86.710s |   85.768s | 0.003123s | 0.938839s | 291.48it/s | 26628.63it/s
    epoch | epochs | sample | samples | train   | test    | best    | training   | epoch     | train     | compile   | test      | fit perf   | test perf
       16 |     25 |  25000 |   25000 |  87.51% |  84.71% |  84.80% |      1465s |   86.913s |   85.998s | 0.001745s | 0.913779s | 290.71it/s | 27358.91it/s
       17 |     25 |  25000 |   25000 |  87.84% |  81.30% |  84.80% |      1552s |   86.621s |   85.752s | 0.001747s | 0.867740s | 291.54it/s | 28810.47it/s
       18 |     25 |  25000 |   25000 |  88.06% |  84.10% |  84.80% |      1633s |   81.310s |   80.393s | 0.001715s | 0.915876s | 310.97it/s | 27296.26it/s
       19 |     25 |  25000 |   25000 |  88.22% |  84.90% |  84.90% |      1718s |   84.460s |   83.524s | 0.002112s | 0.934024s | 299.31it/s | 26765.91it/s
       20 |     25 |  25000 |   25000 |  89.68% |  78.00% |  84.90% |      1799s |   81.042s |   80.132s | 0.001916s | 0.908177s | 311.98it/s | 27527.68it/s
    ```

Repo will be active until I find a job or will lost hope in TM or figure out how to get money out of this thing. If you want to help - you are welcome, please fork and send PRs (or money). If you need help, feel free to open issue or contact me 😏.

## License
Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution
Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

