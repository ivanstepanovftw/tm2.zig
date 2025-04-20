"""
How to test:
```shell
pytest -v 00-naive.py -p no:warnings
```
How to run:
```shell
python3 -m examples.01-vectorized
```
"""
from typing import Literal

import numpy as np
from numba import vectorize


@vectorize([
    'int8(int8, int8)',
    # 'int16(int16, int16)',
    # ...
], identity=1)
def add_sat(a, b):
    """
    Ufunc to add two integers with saturation. Works only for the same argument types. Native implementation https://github.com/jax-ml/jax/issues/26566
    """
    info = np.iinfo(type(a))
    c = a + b
    if overflow := a > 0 and b > info.max - a:
        c = info.max
    if underflow := a < 0 and b < info.min - a:
        c = info.min
    return c


def test_sat():
    a = np.array([-128, -128, 0, 0, 0, 127, 127], dtype=np.int8)
    b = np.array([-1, 1, -1, 0, 1, -1, 1], dtype=np.int8)
    c = add_sat(a, b)
    assert c.dtype == a.dtype
    assert np.array_equal(c, np.array([-128, -127, -1, 0, 1, 126, 127], dtype=np.int8))


class VectorizedBinaryTsetlinMachineTrainer:
    def __init__(self, n_features: int, n_clauses: int, dtype=np.int8):
        assert np.iinfo(dtype).kind == 'i', "dtype must be a signed integer type"

        # Automata states are stored in a 4D array:
        # Shape: (2, clause_dim, 2, in_dim)
        #  - Axis 0: polarity (0: votes for y = 1, 1: votes for y = 0)
        #  - Axis 1: clause index per polarity
        #  - Axis 2: literal type (0: original, 1: negated)
        #  - Axis 3: input feature (in_dim)
        # Negative state value is a subject to exclude literals from the clause firing.
        # Initialize near state action threshold as _empty_ clauses.
        self.state = -np.ones((2, n_clauses, 2, n_features), dtype=dtype)

    def __call__(
            self,
            features: np.ndarray,  # (batch..., in_dim), bool
            target: np.ndarray | None = None,  # (batch...,), bool
            t: int = 8,
            r: float = 0.75,
            rng: np.random.Generator | None = np.random.default_rng(),
            use_1a: bool = True,
            use_1b: bool = True,
            use_2: bool = True,
    ) -> dict[Literal["clauses", "votes", "predictions", "feedback_1a", "feedback_1b", "feedback_2"], np.ndarray]:
        """
        Forward pass and fit steps the Tsetlin machine. If target is None, only the forward pass is performed.
        Returns dictionary. By default, use `result["predictions"]` to get the predictions.
        When `rng` is `None`, we are not using random in the feedback and skipping resource allocation, so `t` and `r` parameters are skipped.
        """
        result = {}

        # As we are focusing on batched input, we cannot simply use `clauses = all(literals[include])`,
        # clauses = all(literals[include]) = all(exclude | literals) = !any(include & ~literals)
        exclude = self.state < 0  # (2, clause_dim, 2, in_dim), bool
        literals = np.stack((features, ~features), -2)  # (batch..., 2, in_dim), bool
        literals_ = literals[..., None, None, :, :]  # (batch..., 1, 1, 2, in_dim), bool
        success = exclude | literals_  # (batch..., 2, clause_dim, 2, in_dim), bool
        result["clauses"] = clauses = success.all((-2, -1))  # (batch..., 2, clause_dim), bool

        pos_votes = clauses[..., :1, :].sum((-2, -1))  # (batch...,), int64
        neg_votes = clauses[..., 1:, :].sum((-2, -1))  # (batch...,), int64
        result["votes"] = votes = pos_votes - neg_votes  # (batch...,), int64
        result["predictions"] = predictions = votes >= 0  # (batch...,), bool

        if target is None:
            return result

        batch_shape = features.shape[:-1]
        batch_axes = tuple(range(len(batch_shape)))
        assert batch_shape == target.shape, "target shape must be equal to batch_shape"

        rand_mask_1a = rand_mask_1b = rand_mask_2 = np.array(True)
        if rng is not None:
            # The "resource allocation" is basically:
            # p_clause_update = np.where(target, -1, 1) * votes.clip(-t, t) / (2 * t) + 0.5  # naive and simple
            # p_clause_update = (t - np.where(target, votes, -votes).clip(-t, t)) / (2 * t)  # numpy optimized
            ## p_clause_update = (np.where(target, votes, -votes).clip(-t, t) + t) * (scale // (2 * t))  # for integer random, also don't forget to scale constants like `1` in `(1 - r)` below.
            # p_clause_update_ = p_clause_update[..., None, None, None, None]  # (batch..., 1, 1, 1, 1), float
            # rand_mask_1a = rng.random(batch_shape + self.state.shape[:2] + (1,) * 2) <= p_clause_update_  # (batch..., 2, clause_dim, 1, 1), bool
            # rand_mask_1b = rng.random(batch_shape + self.state.shape) <= p_clause_update_ * (1 - r)     # (batch..., 2, clause_dim, 2, in_dim), bool
            # rand_mask_2 = rng.random(batch_shape + self.state.shape) <= p_clause_update_ * r            # (batch..., 2, clause_dim, 2, in_dim), bool

            # Dumber version: per sample skip instead of gradually per clause
            too_confident = (target & (votes > t)) | (~target & (votes < -t))  # shape: (batch..., 2)
            too_confident = too_confident[..., None, None, None]  # (batch..., 2, 1, 1, 1), bool
            rand_mask_1a = ~too_confident
            rand_mask_1b = (rng.random(batch_shape + self.state.shape) > r) & ~too_confident
            rand_mask_2 = (rng.random(batch_shape + self.state.shape) <= r) & ~too_confident

        type_1_feedback = np.stack((target, ~target), axis=-1)[..., None, None, None]  # (batch..., 2, 1, 1, 1), bool
        clauses_ = clauses[..., None, None]  # (batch..., 2, clause_dim, 1, 1), bool

        type1a = type_1_feedback & clauses_ & literals_ & rand_mask_1a
        type1b = type_1_feedback & ~(clauses_ & literals_) & rand_mask_1b
        type2 = ~type_1_feedback & clauses_ & ~literals_ & exclude & rand_mask_2

        result["feedback_1a"] = feedback_1a = type1a.sum(batch_axes, self.state.dtype)  # (2, clause_dim, 2, in_dim), state.dtype
        result["feedback_1b"] = feedback_1b = -type1b.sum(batch_axes, self.state.dtype)  # (2, clause_dim, 2, in_dim), state.dtype
        result["feedback_2"] = feedback_2 = type2.sum(batch_axes, self.state.dtype)  # (2, clause_dim, 2, in_dim), state.dtype

        if use_1a: add_sat(self.state, feedback_1a, out=self.state)
        if use_1b: add_sat(self.state, feedback_1b, out=self.state)
        if use_2:  add_sat(self.state, feedback_2, out=self.state)

        return result


def test_forward():
    config = dict(n_features=1, n_clauses=4)
    tm = VectorizedBinaryTsetlinMachineTrainer(**config)
    features = np.array([[True], [False]])  # literals = [[1, 0], [0, 1]]
    tm.state = np.array([  # Shape: (2, clause_dim, 2, in_dim) = (2, 4, 2, 1)
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
    ], dtype=tm.state.dtype)
    assert tm.state.shape == (2, config["n_clauses"], 2, config["n_features"])
    clauses = tm(features)["clauses"]
    target = np.array([
        [[False, True, False, True], [False, True, False, True]],  # For X = [1], literals = [1, 0]
        [[False, False, True, True], [False, False, True, True]],  # For X = [0], literals = [0, 1]
    ])
    assert np.array_equal(clauses, target)


def test_forward_xor():
    """
    Test the XOR function.
    y_pred(X) = u(x1x¯2 + ¯x1x2 − x1x2− x¯1x¯2)
    """
    config = dict(n_features=2, n_clauses=2)
    tm = VectorizedBinaryTsetlinMachineTrainer(**config)
    features = np.array([
        [False, False],  # literals = [[0, 0], [1, 1]]
        [False, True],   # literals = [[0, 1], [1, 0]]
        [True, False],   # literals = [[1, 0], [0, 1]]
        [True, True],    # literals = [[1, 1], [0, 0]]
    ])
    tm.state = np.array([  # Shape: (2, clause_dim, 2, in_dim) = (2, 2, 2, 2)
        #                      # w | j | include          | exclude          | clauses (vectorized)    | prediction
        #                      #   |   | states >= 0      | states < 0       | all(literals | exclude) | u(clauses[1].sum() - clauses[0].sum())
        [[[0, -1], [-1, 0]],   # 1 | 0 | [[1, 0], [0, 1]] | [[0, 1], [1, 0]] | [[0, 0], [0, 1]]        | u(1 - 0) = u(1) = 1
         [[-1, 0], [0, -1]]],  # 1 | 1 | [[0, 1], [1, 0]] | [[1, 0], [0, 1]] | [[1, 0], [0, 0]]        | u(0 - 1) = u(-1) = 0
        [[[0, 0], [-1, -1]],   # 0 | 0 | [[1, 1], [0, 0]] | [[0, 0], [1, 1]] | [[1, 0], [0, 0]]        | u(0 - 1) = u(-1) = 0
         [[-1, -1], [0, 0]]],  # 0 | 1 | [[0, 0], [1, 1]] | [[1, 1], [0, 0]] | [[0, 0], [1, 0]]        | u(1 - 0) = u(1) = 1
    ], dtype=tm.state.dtype)
    assert tm.state.shape == (2, config["n_clauses"], 2, config["n_features"])
    predictions = tm(features)["predictions"]
    target = np.array([False, True, True, False])
    assert np.array_equal(predictions, target)


def test_fit_1a():
    config = dict(n_features=1, n_clauses=4)
    tm = VectorizedBinaryTsetlinMachineTrainer(**config)
    features = np.array([[True]])  # literals = [1, 0]
    target = np.array([True])  # y = 1
    tm.state = np.array([  # shape: (2, clause_dim, 2, in_dim) = (2, 4, 2, 1)
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
    ], dtype=tm.state.dtype)
    assert tm.state.shape == (2, config["n_clauses"], 2, config["n_features"])
    delta = tm(features, target, rng=None, use_1a=True, use_1b=False, use_2=False)["feedback_1a"]
    target_delta = np.array([
        [[[0], [0]],
         [[1], [0]],
         [[0], [0]],
         [[1], [0]]],
        [[[0], [0]],
         [[0], [0]],
         [[0], [0]],
         [[0], [0]]],
    ])
    assert np.array_equal(delta, target_delta)
    target_state = np.array([
        [[[1], [1]],     # change: [0, 0]
         [[2], [-1]],    # change: [1, 0]
         [[-1], [1]],    # change: [0, 0]
         [[0], [-1]]],   # change: [1, 0]
        [[[1], [1]],     # change: [0, 0]
         [[1], [-1]],    # change: [0, 0]
         [[-1], [1]],    # change: [0, 0]
         [[-1], [-1]]],  # change: [0, 0]
    ])
    assert np.array_equal(tm.state, target_state)


def test_fit_1b():
    config = dict(n_features=1, n_clauses=4)
    tm = VectorizedBinaryTsetlinMachineTrainer(**config)
    features = np.array([[True]])  # literals = [1, 0]
    target = np.array([True])  # y = 1
    tm.state = np.array([
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
    ], dtype=tm.state.dtype)
    assert tm.state.shape == (2, config["n_clauses"], 2, config["n_features"])
    delta = tm(features, target, rng=None, use_1a=False, use_1b=True, use_2=False)["feedback_1b"]
    target_delta = np.array([
        [[[-1], [-1]],
         [[0], [-1]],
         [[-1], [-1]],
         [[0], [-1]]],
        [[[0], [0]],
         [[0], [0]],
         [[0], [0]],
         [[0], [0]]],
    ])
    assert np.array_equal(delta, target_delta)
    target_state = np.array([
        [[[0], [0]],     # change: [-1, -1]
         [[1], [-2]],    # change: [ 0, -1]
         [[-2], [0]],    # change: [-1, -1]
         [[-1], [-2]]],  # change: [ 0, -1]
        [[[1], [1]],     # change: [0, 0]
         [[1], [-1]],    # change: [0, 0]
         [[-1], [1]],    # change: [0, 0]
         [[-1], [-1]]],  # change: [0, 0]
    ])
    assert np.array_equal(tm.state, target_state)


def test_fit_2():
    config = dict(n_features=1, n_clauses=4)
    tm = VectorizedBinaryTsetlinMachineTrainer(**config)
    features = np.array([[True]])  # literals = [1, 0]
    target = np.array([True])  # y = 1
    tm.state = np.array([
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
    ], dtype=tm.state.dtype)
    assert tm.state.shape == (2, config["n_clauses"], 2, config["n_features"])
    delta = tm(features, target, rng=None, use_1a=False, use_1b=False, use_2=True)["feedback_2"]
    target_delta = np.array([
        [[[0], [0]],
         [[0], [0]],
         [[0], [0]],
         [[0], [0]]],
        [[[0], [0]],
         [[0], [1]],
         [[0], [0]],
         [[0], [1]]],
    ])
    assert np.array_equal(delta, target_delta)
    target_state = np.array([
        [[[1], [1]],     # change: [0, 0]
         [[1], [-1]],    # change: [0, 0]
         [[-1], [1]],    # change: [0, 0]
         [[-1], [-1]]],  # change: [0, 0]
        [[[1], [1]],     # change: [0, 0]
         [[1], [0]],     # change: [0, 1]
         [[-1], [1]],    # change: [0, 0]
         [[-1], [0]]],   # change: [0, 1]
    ])
    assert np.array_equal(tm.state, target_state)


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
    # Load data as bytes
    # data_bytes = np.frombuffer(b"U" * 1000, dtype=np.uint8)  # [0 1 0 1 0 1 0 1 0 1 ...]
    # data_bytes = np.frombuffer(b"3" * 1000, dtype=np.uint8)  # [0 0 1 1 0 0 1 1 0 0 ...]
    # data_bytes = np.frombuffer(b"U3" * 1000, dtype=np.uint8)
    # data_bytes = np.frombuffer(b"\x0F" * 1000, dtype=np.uint8)  # [0 0 0 0 1 1 1 1 0 0 ...]
    # data_bytes = np.frombuffer(b"U3\x0F" * 1000, dtype=np.uint8)
    # data_bytes = np.frombuffer(b"01" * 500, dtype=np.uint8)
    # data_bytes = np.frombuffer(b"0123456789" * 100, dtype=np.uint8)
    data_bytes = np.memmap("/home/i/d/enwik8", mode='r')
    # data_bytes = np.memmap("/home/i/d/enwik9", mode='r')

    # Unpack into bits (Big Endian)
    bits = np.unpackbits(data_bytes).astype(bool)
    print(f"{bits=}")
    print(f"Total bits: {len(bits)}")

    # Training parameters
    context_length = 8 * 3
    batch_size = 1
    epochs = 5
    steps_per_epoch = 1000
    rng = np.random.default_rng(42)

    config = dict(n_features=context_length, n_clauses=128 * 8)
    tm = VectorizedBinaryTsetlinMachineTrainer(**config)

    # Create a sliding window of size (context_length + 1)
    # so that each window has `context_length` input bits + 1 target bit
    windows = np.lib.stride_tricks.sliding_window_view(bits, context_length + 1)

    # Training
    for epoch in range(1, epochs + 1):
        with Stopwatch() as sw:
            total_error = 0
            for step in range(steps_per_epoch):
                # Sample random window indices for the batch
                indices = rng.integers(0, len(windows), size=batch_size)
                batch_window = windows[indices]

                # Split into X and y
                X = batch_window[..., :-1]
                y = batch_window[..., -1]

                result = tm(X, y, t=1, r=0.821480149837791, rng=rng)
                y_pred = result["predictions"]

                total_error += np.mean(y_pred != y)

        avg_error = total_error / steps_per_epoch
        accuracy = (1 - avg_error) * 100
        print(f"Epoch {epoch} accuracy: {accuracy:.2f}% (elapsed: {sw}, samples/sec: {batch_size * steps_per_epoch / sw.duration:.2f})")

    # Save the model state (tm.state)
    # print(f"tm.state:\n{tm.state}")
    # Optimize the model for inference (tm.state < 0)
    # exclude = tm.state < 0

    # Testing and comparison
    # Here we sample 128 windows for the test set
    test_indices = rng.integers(0, len(windows), size=128)
    X_test_window = windows[test_indices]
    X_test = X_test_window[..., :-1]
    y_test = X_test_window[..., -1]

    # Predict with the Tsetlin Machine
    y_pred_tm = tm(X_test)["predictions"]
    acc_tm = np.mean(y_pred_tm == y_test)

    # Compare arrays
    print(f"Test Accuracy: {acc_tm * 100:.2f}%")

    # Decoding demonstration
    n_decode_bits = 8 * 8
    context_start = 0
    prompt = bits[context_start: context_start + context_length].copy()
    context = prompt.copy()
    decoded_bits = np.empty(n_decode_bits, dtype=bool)

    for i in range(n_decode_bits):
        next_bit = tm(context)["predictions"]
        decoded_bits[i] = next_bit
        context = np.roll(context, -1)
        context[-1] = next_bit

    print(f"Prompt bits: {X_test[0].astype(int)}")
    print(f"Prompt bytes: {np.packbits(prompt).tobytes()}")
    print(f"Prompt string: {np.packbits(prompt).tobytes().decode('utf-8', errors='replace')!r}")
    print(f"Decoded bits: {decoded_bits.astype(int)}")
    print(f"Decoded bytes: {np.packbits(decoded_bits).tobytes()}")
    print(f"Decoded string: {np.packbits(decoded_bits).tobytes().decode('utf-8', errors='replace')!r}")


if __name__ == '__main__':
    main_autoregressive()
