# tm2.zig
Binary Tsetlin machine implementation in Zig. Currently slow. Covered by tests. CPU only.

## Tsetlin machine 
In short, Tsetlin machine is a type of machine learning model. It is energy efficient, does not use floating point computations, bitwise, losslessly quantizable down to 1 bit per weight, lossy compressible up to how many important clauses you want to keep, weights are interpretable (if your inputs are interpretable), it does not overfit on small dataset, it does not require your task to be differentiable... God knows what else can it do.

## This work
This implementation uses different probabilities in Type I(a), Type I(b) and Type II feedbacks, same as in [Julia multi-class TM implementation](https://github.com/BooBSD/Tsetlin.jl).

As in Julia implementation, this implementation is multithreaded and uses lock-free Hogwild! approach to update its weights, with per-sample parallelization.

It differs from other implementations by using one array, thus being simple to understand, see i.e. [./examples/01-vectorized.py](./examples/01-vectorized.py) for a **less than 62 lines of code** reference implementation in NumPy (both training and inference).

Automata states in this binary TM represented as a 1D view of a 4D array of signed integers, where first dimension is clause polarity $C^{+}$ and $C^{-}$, second dimension is a clause index, third dimension is original or negated literal, and fourth dimension is a feature index.

So, the shape is `(2, n_clauses, 2, n_features)`, where `n_clauses` is a number of clauses per polarity, and `n_features` is a number of features. For multi-class Tsetlin machine, shape would be `(n_classes, 2, n_clauses, 2, n_features)`, but it is out of scope for this repo.

Same shapes for compiled models. Data type of compiled state is a binary mask of whether state action is include or exclude. Included states are non-negative integers (`states >= 0`), excluded states are negative integers (`states < 0`). This work uses signed automata states, so state action threshold is hardcoded at 0.

Unlike others implementation, this implementation does not use any floating point computations other than logging metrics.

## Feedback Mechanisms in Tsetlin Automata

This document describes the mechanisms of **Type I** and **Type II Feedback** used in Tsetlin Automata. These feedback mechanisms reinforce or penalize clause actions based on the outputs of the automata and the desired target label $y$.

### Type I Feedback

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

#### Type I Feedback Table

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

### Type II Feedback

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

#### Type II Feedback Table

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
        1 |   1000 |  25000 |   25000 |  68.73% |  77.36% |  77.36% |       138s |  138.485s |   95.986s | 0.002100s | 42.497345s | 260.45it/s | 588.27it/s
        2 |   1000 |  25000 |   25000 |  78.10% |  78.92% |  78.92% |       265s |  126.246s |   82.208s | 0.002201s | 44.035507s | 304.11it/s | 567.72it/s
        3 |   1000 |  25000 |   25000 |  79.16% |  75.72% |  78.92% |       390s |  125.711s |   81.332s | 0.002175s | 44.377140s | 307.38it/s | 563.35it/s
        4 |   1000 |  25000 |   25000 |  79.54% |  82.13% |  82.13% |       515s |  124.065s |   80.800s | 0.002180s | 43.262543s | 309.41it/s | 577.87it/s
        5 |   1000 |  25000 |   25000 |  82.68% |  83.74% |  83.74% |       635s |  120.476s |   75.076s | 0.002690s | 45.397040s | 333.00it/s | 550.70it/s
    epoch | epochs | sample | samples | train   | test    | best    | training   | epoch     | train     | compile   | test      | fit perf   | test perf
        6 |   1000 |  25000 |   25000 |  83.30% |  84.82% |  84.82% |       755s |  119.910s |   74.186s | 0.002477s | 45.721040s | 336.99it/s | 546.79it/s
        7 |   1000 |  25000 |   25000 |  84.52% |  84.27% |  84.82% |       872s |  117.405s |   71.678s | 0.002240s | 45.723910s | 348.78it/s | 546.76it/s
        8 |   1000 |  25000 |   25000 |  84.70% |  84.58% |  84.82% |       995s |  123.049s |   74.713s | 0.002449s | 48.333706s | 334.61it/s | 517.24it/s
        9 |   1000 |  25000 |   25000 |  85.30% |  83.04% |  84.82% |      1117s |  121.315s |   73.238s | 0.002002s | 48.074383s | 341.35it/s | 520.03it/s
       10 |   1000 |  25000 |   25000 |  86.13% |  85.02% |  85.02% |      1240s |  122.918s |   73.310s | 0.001927s | 49.605343s | 341.02it/s | 503.98it/s
    epoch | epochs | sample | samples | train   | test    | best    | training   | epoch     | train     | compile   | test      | fit perf   | test perf
       11 |   1000 |  25000 |   25000 |  87.40% |  84.22% |  85.02% |      1362s |  122.150s |   72.252s | 0.002440s | 49.895798s | 346.01it/s | 501.04it/s
       12 |   1000 |  25000 |   25000 |  87.44% |  84.80% |  85.02% |      1485s |  123.380s |   72.650s | 0.002002s | 50.727870s | 344.12it/s | 492.83it/s
       13 |   1000 |  25000 |   25000 |  87.08% |  84.82% |  85.02% |      1608s |  122.710s |   74.302s | 0.002359s | 48.405530s | 336.46it/s | 516.47it/s
       14 |   1000 |  25000 |   25000 |  89.70% |  85.32% |  85.32% |      1725s |  117.308s |   68.037s | 0.002137s | 49.268864s | 367.45it/s | 507.42it/s
       15 |   1000 |  25000 |   25000 |  87.68% |  69.96% |  85.32% |      1844s |  118.590s |   69.751s | 0.002397s | 48.836530s | 358.42it/s | 511.91it/s
    epoch | epochs | sample | samples | train   | test    | best    | training   | epoch     | train     | compile   | test      | fit perf   | test perf
       16 |   1000 |  25000 |   25000 |  88.58% |  85.08% |  85.32% |      1962s |  118.118s |   67.542s | 0.002141s | 50.574356s | 370.14it/s | 494.32it/s
       17 |   1000 |  25000 |   25000 |  89.13% |  83.21% |  85.32% |      2077s |  115.034s |   67.508s | 0.002079s | 47.524090s | 370.33it/s | 526.05it/s
       18 |   1000 |  25000 |   25000 |  87.99% |  76.00% |  85.32% |      2191s |  114.280s |   67.798s | 0.002383s | 46.479305s | 368.74it/s | 537.87it/s
       19 |   1000 |  25000 |   25000 |  90.72% |  81.87% |  85.32% |      2301s |  109.385s |   63.249s | 0.002198s | 46.133587s | 395.26it/s | 541.90it/s
       20 |   1000 |  25000 |   25000 |  90.99% |  80.72% |  85.32% |      2414s |  113.685s |   63.705s | 0.002099s | 49.977290s | 392.43it/s | 500.23it/s
    ```

Repo will be active until I find a job or will lost hope in TM or figure out how to get money out of this thing. If you want to help - you are welcome, please fork and send PRs (or money). If you need help, feel free to open issue or contact me 😏.

## License
Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution
Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

