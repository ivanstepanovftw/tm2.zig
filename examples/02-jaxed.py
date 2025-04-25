"""
How to test:
```shell
pytest -v examples/02-jaxed.py -p no:warnings
```
How to run:
```shell
python3 -m examples.02-jaxed
```
"""
from functools import partial

from jax import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp


@jax.jit
def add_sat(a: jax.Array, b: jax.Array):
    """
    https://en.cppreference.com/w/cpp/numeric/add_sat
    """
    if a.dtype != b.dtype:
        raise TypeError(f"{a.dtype=} and {b.dtype=} don't match.")
    dtype = a.dtype
    max_int = jnp.array(jnp.iinfo(dtype).max, dtype)
    min_int = jnp.array(jnp.iinfo(dtype).min, dtype)
    c = a + b
    overflow = (a > 0) & (b > max_int - a)
    c = jnp.where(overflow, max_int, c)
    underflow = (a < 0) & (b < min_int - a)
    c = jnp.where(underflow, min_int, c)
    return c


def test_sat():
    a = jnp.array([-128, -128, 0, 0, 0, 127, 127], dtype=jnp.int8)
    b = jnp.array([-1, 1, -1, 0, 1, -1, 1], dtype=jnp.int8)
    c = add_sat(a, b)
    assert c.dtype == a.dtype
    assert jnp.array_equal(c, jnp.array([-128, -127, -1, 0, 1, 126, 127], dtype=jnp.int8))


def init_state(n_features: int, n_clauses: int, dtype=jnp.int8):
    assert jnp.iinfo(dtype).kind == 'i', "dtype must be a signed integer type"

    # Automata states are stored in a 4D array:
    # Shape: (2, clause_dim, 2, in_dim)
    #  - Axis 0: polarity (0: votes for y = 1, 1: votes for y = 0)
    #  - Axis 1: literal type (0: features, 1: ~features)
    #  - Axis 2: input feature (in_dim)
    #  - Axis 3: clause index per polarity
    # Negative state value is a subject to exclude literals from the clause firing.
    # Initialize near state action threshold as _empty_ clauses.
    return -jnp.ones((2, 2, n_features, n_clauses), dtype=dtype)


@jax.jit
def compile_exclude(
        state: jnp.ndarray,  # (2, 2, in_dim, clause_dim), signed integer
) -> jnp.ndarray:
    exclude = state < 0  # (2, 2, in_dim, clause_dim), bool
    return exclude


@jax.jit
def evaluate_clauses(
        exclude: jnp.ndarray,  # (2, 2, in_dim, clause_dim), bool
        features: jnp.ndarray,  # (batch..., in_dim), bool
) -> jnp.ndarray:
    """
    Forward pass and fit the Tsetlin machine. If target is None, only the forward pass is performed.
    By default, use `result["predictions"]` to get the predictions.
    """
    # As we are focusing on batched input, we cannot simply use `clauses = all(literals[include])`,
    # clauses = all(literals[include]) = all(exclude | literals)
    literals = jnp.stack((features, ~features), -2)  # (batch..., 2, in_dim), bool
    literals_ = literals[..., None, :, :, None]  # (batch..., 2, in_dim, 1), bool
    success = exclude | literals_  # (batch..., 2, 2, in_dim, clause_dim), bool
    clauses = success.all((-3, -2))  # (batch..., 2, clause_dim), bool
    return clauses


@jax.jit
def evaluate(
        exclude: jnp.ndarray,  # (2, 2, in_dim, clause_dim), bool
        features: jnp.ndarray,  # (batch..., in_dim), bool
) -> jnp.ndarray:
    """
    Forward pass and fit the Tsetlin machine. If target is None, only the forward pass is performed.
    By default, use `result["predictions"]` to get the predictions.
    """
    # As we are focusing on batched input, we cannot simply use `clauses = all(literals[include])`,
    # clauses = all(literals[include]) = all(exclude | literals) = !any(include & ~literals)
    literals = jnp.stack((features, ~features), -2)  # (batch..., 2, in_dim), bool
    literals_ = literals[..., None, :, :, None]  # (batch..., 2, in_dim, 1), bool
    success = exclude | literals_  # (batch..., 2, 2, in_dim, clause_dim), bool
    clauses = success.all((-3, -2))  # (batch..., 2, clause_dim), bool

    pos_votes = clauses[..., :1, :].sum((-2, -1))  # (batch...,), int64
    neg_votes = clauses[..., 1:, :].sum((-2, -1))  # (batch...,), int64
    votes = pos_votes - neg_votes  # (batch...,), int64
    predictions = votes >= 0  # (batch...,), bool
    return predictions


@partial(jax.jit, static_argnums=(6, 7, 8))
def fit(
        state: jnp.ndarray,
        features: jnp.ndarray,  # (batch..., in_dim), bool
        target: jnp.ndarray,  # (batch...,), bool
        t: int = 8,
        r: float = 0.75,
        rng_key: jax.Array = None,
        use_1a: bool = True,
        use_1b: bool = True,
        use_2: bool = True,
) -> (jnp.ndarray, jnp.ndarray):
    """
    Forward pass and fit the Tsetlin machine. If target is None, only the forward pass is performed.
    Returns updated state and the predictions.
    """
    # As we are focusing on batched input, we cannot simply use `clauses = all(literals[include])`,
    # clauses = all(literals[include]) = all(exclude | literals) = !any(include & ~literals)
    exclude = state < 0  # (2, 2, in_dim, clause_dim), bool
    literals = jnp.stack((features, ~features), -2)  # (batch..., 2, in_dim), bool
    literals_ = literals[..., None, :, :, None]  # (batch..., 1, 2, in_dim, 1), bool
    success = exclude | literals_  # (batch..., 2, 2, in_dim, clause_dim), bool
    clauses = success.all((-3, -2))  # (batch..., 2, clause_dim), bool

    pos_votes = clauses[..., :1, :].sum((-2, -1))  # (batch...,), int64
    neg_votes = clauses[..., 1:, :].sum((-2, -1))  # (batch...,), int64
    votes = pos_votes - neg_votes  # (batch...,), int64
    predictions = votes >= 0  # (batch...,), bool

    batch_shape = features.shape[:-1]
    batch_axes = tuple(range(len(batch_shape)))
    assert batch_shape == target.shape, "target shape must be equal to batch_shape"

    type_1_feedback = jnp.stack((target, ~target), axis=-1)[..., None, None, None]  # (batch..., 2, 1, 1, 1), bool
    clauses_ = clauses[..., None, None, :]  # (batch..., 2, 1, 1, clause_dim), bool

    rand_mask_1a = rand_mask_1b = rand_mask_2 = jnp.array(True)
    if rng_key is not None:
        # The "resource allocation" is basically:
        # p_clause_update = np.where(target, 1, -1) * votes.clip(-t, t) / (2 * t) + 0.5
        p_clause_update = (t - jnp.where(target, votes, -votes).clip(-t, t)) / (2 * t)  # (batch...,), float
        # p_clause_update = (jnp.where(target, votes, -votes).clip(-t, t) + t) * (scale // (2 * t))  # for integer random, also don't forget to scale constants like `1` in `(1 - r)` below.
        p_clause_update_ = p_clause_update[..., None, None, None, None]  # (batch..., 1, 1, 1, 1), float

        key_1a, key_1b, key_2 = jax.random.split(rng_key, 3)
        rand_mask_1a = jax.random.uniform(key_1a, (*batch_shape, 2, 1, 1, state.shape[-1])) <= p_clause_update_  # (batch..., 2, 1, 1, clause_dim), bool
        rand_mask_1b = jax.random.uniform(key_1b, batch_shape + state.shape) <= p_clause_update_ * (1 - r)  # (batch..., 2, 2, in_dim, clause_dim), bool
        rand_mask_2 = jax.random.uniform(key_2, batch_shape + state.shape) <= p_clause_update_ * r  # (batch..., 2, 2, in_dim, clause_dim), bool

    type1a = type_1_feedback & clauses_ & literals_ & rand_mask_1a
    type1b = type_1_feedback & ~(clauses_ & literals_) & rand_mask_1b
    type2 = ~type_1_feedback & clauses_ & ~literals_ & exclude & rand_mask_2

    feedback_1a = type1a.sum(batch_axes, state.dtype)  # (2, 2, in_dim, clause_dim), state.dtype
    feedback_1b = -type1b.sum(batch_axes, state.dtype)  # (2, 2, in_dim, clause_dim), state.dtype
    feedback_2 = type2.sum(batch_axes, state.dtype)  # (2, 2, in_dim, clause_dim), state.dtype

    if use_1a: state = add_sat(state, feedback_1a)
    if use_1b: state = add_sat(state, feedback_1b)
    if use_2: state = add_sat(state, feedback_2)

    return state, predictions


def test_forward():
    config = dict(n_features=1, n_clauses=4)
    features = jnp.array([[True], [False]])  # literals = [[1, 0], [0, 1]]
    state = jnp.array([  # Shape: (2, clause_dim, 2, in_dim) = (2, 4, 2, 1)
        #                # include     | exclude    | clauses (indexing)     | clauses (vectorized)
        #                # states >= 0 | states < 0 | all(literals[include]) | all(literals | exclude)
        [[[1], [1]],     # [1, 1]      | [0, 0]     | all([1, 0]) = 0        | all([1, 0]) = 0
         [[1], [-1]],    # [1, 0]      | [0, 1]     | all([1])    = 1        | all([1, 1]) = 1
         [[-1], [1]],    # [0, 1]      | [1, 0]     | all([0])    = 0        | all([1, 0]) = 0
         [[-1], [-1]]],  # [0, 0]      | [1, 1]     | all([])     = 1        | all([1, 1]) = 1
        [[[1], [1]],     # [1, 1]      | [0, 0]     | all([1, 0]) = 0        | all([1, 0]) = 0
         [[1], [-1]],    # [1, 0]      | [0, 1]     | all([1])    = 1        | all([1, 1]) = 1
         [[-1], [1]],    # [0, 1]      | [1, 0]     | all([0])    = 0        | all([1, 0]) = 0
         [[-1], [-1]]],  # [0, 0]      | [1, 1]     | all([])     = 1        | all([1, 1]) = 1
    ], dtype=jnp.int8).transpose(0, 2, 3, 1)
    assert state.shape == (2, 2, config["n_features"], config["n_clauses"])
    exclude = compile_exclude(state)  # (2, clause_dim, 2, in_dim), bool
    clauses = evaluate_clauses(exclude, features)
    target = jnp.array([
        [[False, True, False, True], [False, True, False, True]],  # For X = [1], literals = [1, 0]
        [[False, False, True, True], [False, False, True, True]],  # For X = [0], literals = [0, 1]
    ])
    assert jnp.array_equal(clauses, target)


def test_forward_xor():
    """
    Test the XOR function.
    y_pred(X) = u(x1x¯2 + ¯x1x2 − x1x2− x¯1x¯2)
    """
    config = dict(n_features=2, n_clauses=2)
    features = jnp.array([
        [False, False],  # literals = [[0, 0], [1, 1]]
        [False, True],   # literals = [[0, 1], [1, 0]]
        [True, False],   # literals = [[1, 0], [0, 1]]
        [True, True],    # literals = [[1, 1], [0, 0]]
    ])
    state = jnp.array([  # Shape: (2, clause_dim, 2, in_dim) = (2, 2, 2, 2)
        #                      # w | j | include          | exclude          | clauses (vectorized)    | prediction
        #                      #   |   | states >= 0      | states < 0       | all(literals | exclude) | u(clauses[1].sum() - clauses[0].sum())
        [[[0, -1], [-1, 0]],   # 1 | 0 | [[1, 0], [0, 1]] | [[0, 1], [1, 0]] | [[0, 0], [0, 1]]        | u(1 - 0) = u(1) = 1
         [[-1, 0], [0, -1]]],  # 1 | 1 | [[0, 1], [1, 0]] | [[1, 0], [0, 1]] | [[1, 0], [0, 0]]        | u(0 - 1) = u(-1) = 0
        [[[0, 0], [-1, -1]],   # 0 | 0 | [[1, 1], [0, 0]] | [[0, 0], [1, 1]] | [[1, 0], [0, 0]]        | u(0 - 1) = u(-1) = 0
         [[-1, -1], [0, 0]]],  # 0 | 1 | [[0, 0], [1, 1]] | [[1, 1], [0, 0]] | [[0, 0], [1, 0]]        | u(1 - 0) = u(1) = 1
    ], dtype=jnp.int8).transpose(0, 2, 3, 1)
    assert state.shape == (2, 2, config["n_features"], config["n_clauses"])
    exclude = compile_exclude(state)  # (2, clause_dim, 2, in_dim), bool
    predictions = evaluate(exclude, features)
    target = jnp.array([False, True, True, False])
    assert jnp.array_equal(predictions, target)


def test_fit_1a():
    config = dict(n_features=1, n_clauses=4)
    features = jnp.array([[True]])  # literals = [1, 0]
    target = jnp.array([True])  # y = 1
    state = jnp.array([  # shape: (2, clause_dim, 2, in_dim) = (2, 4, 2, 1)
        # Type 1a - memorize automata on clause == 1 and literal == 1
        #                # w | j | feedback (type 1) | condition (type 1a) | state_delta (type 1a)
        #                #   |   | w == y            | clauses & literals  | feedback & condition
        [[[1], [1]],     # 1 | 0 |                 1 | 0 & [1, 0] = [0, 0] | [0, 0]
         [[1], [-1]],    # 1 | 1 |                 1 | 1 & [1, 0] = [1, 0] | [1, 0]
         [[-1], [1]],    # 1 | 2 |                 1 | 0 & [1, 0] = [0, 0] | [0, 0]
         [[-1], [-1]]],  # 1 | 3 |                 1 | 1 & [1, 0] = [1, 0] | [1, 0]
        [[[1], [1]],     # 0 | 0 |                 0 | 0 & [1, 0] = [0, 0] | [0, 0]
         [[1], [-1]],    # 0 | 1 |                 0 | 1 & [1, 0] = [1, 0] | [0, 0]
         [[-1], [1]],    # 0 | 2 |                 0 | 0 & [1, 0] = [0, 0] | [0, 0]
         [[-1], [-1]]],  # 0 | 3 |                 0 | 1 & [1, 0] = [1, 0] | [0, 0]
    ], dtype=jnp.int8).transpose(0, 2, 3, 1)
    assert state.shape == (2, 2, config["n_features"], config["n_clauses"])
    state, predictions = fit(state, features, target, rng_key=None, use_1a=True, use_1b=False, use_2=False)
    target_state = jnp.array([
        [[[1], [1]],     # change: [0, 0]
         [[2], [-1]],    # change: [1, 0]
         [[-1], [1]],    # change: [0, 0]
         [[0], [-1]]],   # change: [1, 0]
        [[[1], [1]],     # change: [0, 0]
         [[1], [-1]],    # change: [0, 0]
         [[-1], [1]],    # change: [0, 0]
         [[-1], [-1]]],  # change: [0, 0]
    ]).transpose(0, 2, 3, 1)
    assert jnp.array_equal(state, target_state)


def test_fit_1b():
    config = dict(n_features=1, n_clauses=4)
    features = jnp.array([[True]])  # literals = [1, 0]
    target = jnp.array([True])  # y = 1
    state = jnp.array([
        # Type 1b - forget automata on clause == 0 or literal == 0
        #                # w | j | feedback (type 1) | condition (type 1b)   | -feedback_1b (type 1b)
        #                #   |   | w == y            | !(clauses & literals) | -(feedback & condition)
        [[[1], [1]],     # 1 | 0 |                 1 | ![0, 0] = [1, 1]      | [-1, -1]
         [[1], [-1]],    # 1 | 1 |                 1 | ![1, 0] = [0, 1]      | [ 0, -1]
         [[-1], [1]],    # 1 | 2 |                 1 | ![0, 0] = [1, 1]      | [-1, -1]
         [[-1], [-1]]],  # 1 | 3 |                 1 | ![1, 0] = [0, 1]      | [ 0, -1]
        [[[1], [1]],     # 0 | 0 |                 0 | ![0, 0] = [1, 1]      | [0, 0]
         [[1], [-1]],    # 0 | 1 |                 0 | ![1, 0] = [0, 1]      | [0, 0]
         [[-1], [1]],    # 0 | 2 |                 0 | ![0, 0] = [1, 1]      | [0, 0]
         [[-1], [-1]]],  # 0 | 3 |                 0 | ![1, 0] = [0, 1]      | [0, 0]
    ], dtype=jnp.int8).transpose(0, 2, 3, 1)
    assert state.shape == (2, 2, config["n_features"], config["n_clauses"])
    state, predictions = fit(state, features, target, rng_key=None, use_1a=False, use_1b=True, use_2=False)
    target_state = jnp.array([
        [[[0], [0]],     # change: [-1, -1]
         [[1], [-2]],    # change: [ 0, -1]
         [[-2], [0]],    # change: [-1, -1]
         [[-1], [-2]]],  # change: [ 0, -1]
        [[[1], [1]],     # change: [0, 0]
         [[1], [-1]],    # change: [0, 0]
         [[-1], [1]],    # change: [0, 0]
         [[-1], [-1]]],  # change: [0, 0]
    ]).transpose(0, 2, 3, 1)
    assert jnp.array_equal(state, target_state)


def test_fit_2():
    config = dict(n_features=1, n_clauses=4)
    features = jnp.array([[True]])  # literals = [1, 0]
    target = jnp.array([True])  # y = 1
    state = jnp.array([
        # Type 2 - penalize excluded (thus, remember) automata on clause == 1 and literal == 0
        #                # w | j | feedback (type 2) | condition (type 2)  | state_delta (type 2)
        #                #   |   | !(w == y)         | clauses & !literals | feedback & condition & exclude
        [[[1], [1]],     # 1 | 0 |                 0 | 0 & [0, 1] = [0, 0] | [0, 0]
         [[1], [-1]],    # 1 | 1 |                 0 | 1 & [0, 1] = [0, 1] | [0, 0]
         [[-1], [1]],    # 1 | 2 |                 0 | 0 & [0, 1] = [0, 0] | [0, 0]
         [[-1], [-1]]],  # 1 | 3 |                 0 | 1 & [0, 1] = [0, 1] | [0, 0]
        [[[1], [1]],     # 0 | 0 |                 1 | 0 & [0, 1] = [0, 0] | [0, 0]
         [[1], [-1]],    # 0 | 1 |                 1 | 1 & [0, 1] = [0, 1] | [0, 1]
         [[-1], [1]],    # 0 | 2 |                 1 | 0 & [0, 1] = [0, 0] | [0, 0]
         [[-1], [-1]]],  # 0 | 3 |                 1 | 1 & [0, 1] = [0, 1] | [0, 1]
    ], dtype=jnp.int8).transpose(0, 2, 3, 1)
    assert state.shape == (2, 2, config["n_features"], config["n_clauses"])
    state, predictions = fit(state, features, target, rng_key=None, use_1a=False, use_1b=False, use_2=True)
    target_state = jnp.array([
        [[[1], [1]],     # change: [0, 0]
         [[1], [-1]],    # change: [0, 0]
         [[-1], [1]],    # change: [0, 0]
         [[-1], [-1]]],  # change: [0, 0]
        [[[1], [1]],     # change: [0, 0]
         [[1], [0]],     # change: [0, 1]
         [[-1], [1]],    # change: [0, 0]
         [[-1], [0]]],   # change: [0, 1]
    ]).transpose(0, 2, 3, 1)
    assert jnp.array_equal(state, target_state)


class Stopwatch:
    import time
    time = time.perf_counter

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = self.time()

    @property
    def duration(self):
        end = getattr(self, "end", None) or self.time()
        return end - self.start

    def __str__(self):
        return f"{self.duration:.3f} s"

    __repr__ = __str__


def main_autoregressive():
    import numpy as np
    # Load data as bytes
    # data_bytes = jnp.frombuffer(b"U" * 1000, dtype=jnp.uint8)  # [0 1 0 1 0 1 0 1 0 1 ...]
    # data_bytes = jnp.frombuffer(b"3" * 1000, dtype=jnp.uint8)  # [0 0 1 1 0 0 1 1 0 0 ...]
    # data_bytes = jnp.frombuffer(b"U3" * 1000, dtype=jnp.uint8)
    # data_bytes = jnp.frombuffer(b"\x0F" * 1000, dtype=jnp.uint8)  # [0 0 0 0 1 1 1 1 0 0 ...]
    # data_bytes = jnp.frombuffer(b"U3\x0F" * 1000, dtype=jnp.uint8)
    # data_bytes = jnp.frombuffer(b"01" * 500, dtype=jnp.uint8)
    data_bytes = jnp.frombuffer(b"0123456789" * 100, dtype=jnp.uint8)
    # data_bytes = np.memmap("/home/i/d/enwik8", mode='r')
    # data_bytes = np.memmap("/home/i/d/enwik9", mode='r')

    # Unpack into bits (Big Endian)
    bits = jnp.unpackbits(data_bytes).astype(bool)
    print(f"{bits=}")
    print(f"Total bits: {len(bits)}")

    # Training parameters
    context_length = 8 * 3
    batch_size = 1
    epochs = 5
    steps_per_epoch = 1000
    rng_key = jax.random.PRNGKey(0)

    config = dict(n_features=context_length, n_clauses=128 * 8)
    state = init_state(**config)

    # Create a sliding window of size (context_length + 1)
    # so that each window has `context_length` input bits + 1 target bit
    windows = np.lib.stride_tricks.sliding_window_view(bits, context_length + 1)

    # Training
    for epoch in range(1, epochs + 1):
        with Stopwatch() as sw:
            total_error = 0
            for step in range(steps_per_epoch):
                # Sample random window indices for the batch
                subkey, rng_key = jax.random.split(rng_key)
                indices = jax.random.randint(subkey, (batch_size, ), 0, len(windows))
                batch_window = windows[indices]

                # Split into X and y
                X = batch_window[..., :-1]
                y = batch_window[..., -1]

                rng_key, subkey = jax.random.split(rng_key)
                state, y_pred = fit(state, X, y, t=1, r=0.821480149837791, rng_key=subkey)

                total_error += jnp.mean(y_pred != y)

        avg_error = total_error / steps_per_epoch
        accuracy = (1 - avg_error) * 100
        print(f"Epoch {epoch} accuracy: {accuracy:.2f}% (elapsed: {sw}, samples/sec: {batch_size * steps_per_epoch / sw.duration:.2f})")

    # Save the model state (tm.state)
    # print(f"state:\n{state}")
    # Optimize the model for inference: `compile_exclude(state)`

    # Testing and comparison
    # Here we sample 128 windows for the test set
    subkey, rng_key = jax.random.split(rng_key)
    test_indices = jax.random.randint(subkey, (128,), 0, len(windows))
    X_test_window = windows[test_indices]
    X_test = X_test_window[..., :-1]
    y_test = X_test_window[..., -1]

    # Predict with the Tsetlin Machine
    exclude = compile_exclude(state)  # (2, clause_dim, 2, in_dim), bool
    with Stopwatch() as sw:
        y_pred_tm = evaluate(exclude, X_test[:1])
    print(f"Elapsed time for compilation and 1 sample evaluation: {sw}, samples/sec: {1 / sw.duration:.2f}")
    with Stopwatch() as sw:
        y_pred_tm = evaluate(exclude, X_test)
    print(f"Elapsed time for evaluation: {sw}, samples/sec: {len(X_test) / sw.duration:.2f}")
    acc_tm = jnp.mean(y_pred_tm == y_test)

    # Compare arrays
    print(f"Test Accuracy: {acc_tm * 100:.2f}%")

    # Decoding demonstration
    n_decode_bits = 8 * 8
    context_start = 0
    prompt = bits[context_start: context_start + context_length].copy()
    context = prompt.copy()
    decoded_bits = jnp.empty(n_decode_bits, dtype=bool)

    for i in range(n_decode_bits):
        next_bit = evaluate(exclude, context)
        decoded_bits = decoded_bits.at[i].set(next_bit)
        context = jnp.roll(context, -1)
        context = context.at[-1].set(next_bit)

    print(f"Prompt bits: {X_test[0].astype(int)}")
    print(f"Prompt bytes: {jnp.packbits(prompt).tobytes()}")
    print(f"Prompt string: {jnp.packbits(prompt).tobytes().decode('utf-8', errors='replace')!r}")
    print(f"Decoded bits: {decoded_bits.astype(int)}")
    print(f"Decoded bytes: {jnp.packbits(decoded_bits).tobytes()}")
    print(f"Decoded string: {jnp.packbits(decoded_bits).tobytes().decode('utf-8', errors='replace')!r}")


if __name__ == '__main__':
    main_autoregressive()
