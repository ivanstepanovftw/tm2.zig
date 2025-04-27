/// Easy-to-use duplicate of `std.math.divCeil` without error handling.
pub fn divCeil(a: anytype, b: anytype) @TypeOf(a) {
    return (a + b - 1) / b;
}

/// Rounds up `value` to the next multiple of `mod`. Duplicates `std.mem.alignForward`, but with `comptime_int` support.
pub fn alignForward(value: anytype, mod: anytype) @TypeOf(value) {
    return ((value + mod - 1) / mod) * mod;
}

/// Unmanaged binary vector.
pub const BitVector = struct {
    const Self = @This();

    len: usize,
    bytes: []u8,

    pub fn init(allocator: std.mem.Allocator, n: usize) !Self {
        const n_bytes = divCeil(n, 8);
        const bytes = try allocator.alloc(u8, n_bytes);
        return Self{ .len = n, .bytes = @ptrCast(bytes) };
    }

    pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
        allocator.free(self.bytes);
    }

    pub fn from(n: usize, bytes: []u8) Self {
        return Self{ .len = n, .bytes = bytes };
    }

    pub inline fn setValue(self: *Self, index: usize, value: bool) void {
        std.debug.assert(index < self.len);
        const i_byte = index / 8;
        const i_mask = index % 8;
        const mask: u8 = @as(u8, 1) << @as(u3, @intCast(i_mask));
        if (value) {
            self.bytes[i_byte] |= mask;
        } else {
            self.bytes[i_byte] &= ~mask;
        }
    }

    pub inline fn setValueTrue(self: *Self, index: usize) void {
        std.debug.assert(index < self.len);
        const i_byte = index / 8;
        const i_mask = index % 8;
        const mask: u8 = @as(u8, 1) << @as(u3, @intCast(i_mask));
        self.bytes[i_byte] |= mask;
    }

    pub inline fn getValue(self: Self, index: usize) bool {
        std.debug.assert(index < self.len);
        const i_byte = index / 8;
        const i_mask = index % 8;
        const mask: u8 = @as(u8, 1) << @as(u3, @intCast(i_mask));
        return self.bytes[i_byte] & mask != 0;
    }
};

test BitVector {
    const allocator = std.testing.allocator;
    const n = 1<<20 + 3;
    var bv = try BitVector.init(allocator, n);
    defer bv.deinit(allocator);

    const seed = 111111111;
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    for (0..n) |i|
        bv.setValue(i, random.boolean());
    prng.seed(seed);
    for (0..n) |i|
        try testing.expectEqual(random.boolean(), bv.getValue(i));
}

test "@Vector(8, u1) layout is native-endian" {
    const v: @Vector(8, u1) = .{ 1, 0, 0, 0, 0, 0, 0, 0 };
    const u: u8 = @bitCast(v);
    try std.testing.expectEqual(u, 1);
}

pub const Config = struct {
    n_features: usize,
    n_clauses: usize,

    pub fn init(
        n_features: usize,
        n_clauses: usize,
    ) Config {
        return Config{
            .n_features = n_features,
            .n_clauses = n_clauses,
        };
    }

    pub fn stateSize(self: Config) usize {
        return 2 * self.n_clauses * 2 * self.n_features;
    }

    pub fn stateIndex(
        self: Config,
        i_polarity: usize,
        i_clause: usize,
        i_negated: usize,
        i_feature: usize,
    ) usize {
        std.debug.assert(i_polarity < 2);
        std.debug.assert(i_clause < self.n_clauses);
        std.debug.assert(i_negated < 2);
        std.debug.assert(i_feature < self.n_features);
        const stride_negated = self.n_features;
        const stride_clause = stride_negated * 2;
        const stride_polarity = stride_clause * self.n_clauses;
        return i_feature + stride_negated * i_negated + stride_clause * i_clause + stride_polarity * i_polarity;
    }
};


/// 0.042881s
pub fn getIncludesOnePass(comptime S: type, states: []S, includes: *BitVector) void {
    std.debug.assert(states.len == includes.len);
    for (0..states.len) |i_state|
        includes.setValue(i_state, states[i_state] >= 0);
}

/// 0.009739s
pub fn getIncludesTwoPass(comptime S: type, states: []S, includes: *BitVector) void {
    std.debug.assert(states.len == includes.len);
    @memset(includes.bytes, 0);
    for (0..states.len) |i_state|
        if (states[i_state] >= 0)
            includes.setValueTrue(i_state);
}

/// 0.001773s (outdated)
pub fn getIncludesVector8(comptime S: type, states: []S, includes: *BitVector) void {
    std.debug.assert(states.len == includes.len);
    var i_state: usize = 0;
    while (i_state + 8 < states.len) : (i_state += 8) {
        const threshold: @Vector(8, S) = @splat(0);
        const states_vec: @Vector(8, S) = @bitCast(states[i_state .. i_state + 8][0..8].*);
        includes.bytes[i_state / 8] = @bitCast(states_vec >= threshold);
    }
    while (i_state < states.len) : (i_state += 1) {
        includes.setValue(i_state, states[i_state] >= 0);
    }
}

/// 0.001386s (outdated)
/// Repeats logic of `getIncludesVector8` but uses suggested vector length.
pub fn getIncludesVectorA(comptime S: type, states: []S, includes: *BitVector) void {
    std.debug.assert(states.len == includes.len);
    var i_state: usize = 0;
    const l = std.simd.suggestVectorLength(S) orelse 1;
    if (l >= 8 and (l % 8) == 0) {
        while (i_state + l <= states.len) : (i_state += l) {
            const threshold: @Vector(l, S) = @splat(0);
            const states_vec: @Vector(l, S) = @bitCast(states[i_state .. i_state + l][0..l].*);
            const includes_vec: @Vector(divCeil(l, 8), u8) = @bitCast(states_vec >= threshold);
            const dest_includes_vec_ptr: *@Vector(divCeil(l, 8), u8) = @alignCast(@ptrCast(&includes.bytes[i_state / 8]));
            dest_includes_vec_ptr.* = includes_vec;
        }
    }
    while (i_state < states.len) : (i_state += 1) {
        includes.setValue(i_state, states[i_state] >= 0);
    }
}

/// Best of `getIncludesVector8` and `getIncludesVectorA`.
pub fn getIncludesVectorB(comptime S: type, states: []S, includes: *BitVector) void {
    std.debug.assert(states.len == includes.len);
    var i_state: usize = 0;
    const l = alignForward(std.simd.suggestVectorLength(S) orelse 1, 8);
    while (i_state + l <= states.len) : (i_state += l) {
        const threshold: @Vector(l, S) = @splat(0);
        const states_vec: @Vector(l, S) = @bitCast(states[i_state .. i_state + l][0..l].*);
        const includes_vec: @Vector(l / 8, u8) = @bitCast(states_vec >= threshold);
        const dest_includes_vec_ptr: *@Vector(l / 8, u8) = @alignCast(@ptrCast(&includes.bytes[i_state / 8]));
        dest_includes_vec_ptr.* = includes_vec;
    }
    while (i_state < states.len) : (i_state += 1) {
        includes.setValue(i_state, states[i_state] >= 0);
    }
}

pub const getIncludes = getIncludesVectorB;

pub fn getClauseU1(
    config: Config,
    comptime S: type,
    states: *[]S,
    features: BitVector,
    i_polarity: usize,
    i_clause: usize,
) u1 {
    for (0..2) |i_negated| for (0..config.n_features) |i_feature| {
        const i_state = config.stateIndex(i_polarity, i_clause, i_negated, i_feature);
        const exclude: u1 = @intFromBool(states.*[i_state] < 0);
        const feature: u1 = @intFromBool(features.getValue(i_feature));
        const literal: u1 = if (i_negated == 0) feature else ~feature;
        if (exclude | literal == 0) {
            return 0;
        }
    };
    return 1;
}

pub fn getClauseIncludesU1(
    config: Config,
    includes: BitVector,
    features: BitVector,
    i_polarity: usize,
    i_clause: usize,
) u1 {
    for (0..2) |i_negated| for (0..config.n_features) |i_feature| {
        const i_state = config.stateIndex(i_polarity, i_clause, i_negated, i_feature);
        const include: u1 = @intFromBool(includes.getValue(i_state));
        const exclude: u1 = ~include;
        const feature: u1 = @intFromBool(features.getValue(i_feature));
        const literal: u1 = if (i_negated == 0) feature else ~feature;
        if (exclude | literal == 0) {
            return 0;
        }
    };
    return 1;
}

pub fn evaluateIncludesU1(
    config: Config,
    comptime V: type,
    includes: BitVector,
    features: BitVector,
) V {
    var votes: V = 0;
    for (0..2) |i_polarity| for (0..config.n_clauses) |i_clause| {
        const clause: u1 = getClauseIncludesU1(config, includes, features, i_polarity, i_clause);
        if (i_polarity == 0) votes += clause else votes -= clause;
    };
    return votes;
}

test evaluateIncludesU1 {
    const allocator = std.testing.allocator;
    const n_features = 2;
    const n_clauses  = 2;
    const config = Config.init(n_features, n_clauses);
    const V = i32;  // vote‐count return type

    var includes = try BitVector.init(allocator, config.stateSize());
    defer includes.deinit(allocator);
    @memset(includes.bytes, 0);

    const xorIncludes = [_]u1{
        1, 0, 0, 1,
        0, 1, 1, 0,
        1, 1, 0, 0,
        0, 0, 1, 1,
    };
    for (0..xorIncludes.len, xorIncludes) |i, value|
        includes.setValue(i, value != 0);

    var features = try BitVector.init(allocator, n_features);
    defer features.deinit(allocator);
    @memset(features.bytes, 0);

    const xorFeaturesBatch = [4][2]u1{
        .{0, 0},
        .{0, 1},
        .{1, 0},
        .{1, 1},
    };
    const xorTargetVotesBatch = [4]i32{
        -1,
        1,
        1,
        -1,
    };
    inline for (xorFeaturesBatch, xorTargetVotesBatch) |xorFeatures, xorTargetVotes| {
        for (0..xorFeatures.len, xorFeatures) |i, value|
            features.setValue(i, value != 0);
        const votes = evaluateIncludesU1(config, V, includes, features);
        try testing.expectEqual(xorTargetVotes, votes);

        const votes2 = evaluateIncludesVector(config, V, includes, features);
        try testing.expectEqual(xorTargetVotes, votes2);
    }
}

pub fn evaluateIncludesVector(
    config: Config,
    comptime V: type,
    includes: BitVector,
    features: BitVector,
) V {
    if (@typeInfo(V).int.signedness != .signed) unreachable;
    var votes: V = 0;
    const l = std.simd.suggestVectorLength(u8) orelse 1;
    const Vec = @Vector(l, u8);

    for (0..2) |i_polarity| {
        for (0..config.n_clauses) |i_clause| {
            var c: u8 = 0xFF;
            var i_feature: usize = 0;
            while (c == 0xFF and i_feature + l * 8 < config.n_features) : (i_feature += l * 8) {
                const literal_pos: Vec = @bitCast(features.bytes[i_feature/8 .. i_feature/8 + l][0..l].*);
                const literal_neg: Vec = ~literal_pos;
                const i_pos = config.stateIndex(i_polarity, i_clause, 0, i_feature);
                const i_neg = config.stateIndex(i_polarity, i_clause, 1, i_feature);
                const includes_pos: Vec = @bitCast(includes.bytes[i_pos/8 .. i_pos/8 + l][0..l].*);
                const includes_neg: Vec = @bitCast(includes.bytes[i_neg/8 .. i_neg/8 + l][0..l].*);
                c &= @reduce(.And, (~includes_pos | literal_pos) & (~includes_neg | literal_neg));
            }
            while (c == 0xFF and i_feature < config.n_features) : (i_feature += 1) {
                const feature_bit: u8 = if (features.getValue(i_feature)) 0xFF else 0;
                const i_pos = config.stateIndex(i_polarity, i_clause, 0, i_feature);
                const i_neg = config.stateIndex(i_polarity, i_clause, 1, i_feature);
                const ip: u8 = if (includes.getValue(i_pos)) 0xFF else 0;
                const in: u8 = if (includes.getValue(i_neg)) 0xFF else 0;
                c &= ((~ip | feature_bit) & (~in | ~feature_bit));
            }

            const p: V = @intFromBool(c == 0xFF);
            if (i_polarity == 0) votes += p else votes -= p;
        }
    }

    return votes;
}

// pub const evaluateIncludes = evaluateIncludesU1;
pub const evaluateIncludes = evaluateIncludesVector;

test "fuzz getIncludes() and evaluate()" {
    const allocator = std.testing.allocator;
    const n_features = 256+4;
    const n_clauses = 256+4;
    const config = Config.init(n_features, n_clauses);
    const S = i8;   // automata type
    const V = i32;  // vote‐count return type

    var prng = std.Random.DefaultPrng.init(111111111);
    const random = prng.random();

    const states_buf = try allocator.alloc(S, config.stateSize());
    defer allocator.free(states_buf);
    @memset(states_buf, -1);

    var includesOnePass = try BitVector.init(allocator, config.stateSize());
    defer includesOnePass.deinit(allocator);
    @memset(includesOnePass.bytes, 0);

    var includesTwoPass = try BitVector.init(allocator, config.stateSize());
    defer includesTwoPass.deinit(allocator);
    @memset(includesTwoPass.bytes, 0);

    var includesVector8 = try BitVector.init(allocator, config.stateSize());
    defer includesVector8.deinit(allocator);
    @memset(includesVector8.bytes, 0);

    var includesVectorA = try BitVector.init(allocator, config.stateSize());
    defer includesVectorA.deinit(allocator);
    @memset(includesVectorA.bytes, 0);

    var includesVectorB = try BitVector.init(allocator, config.stateSize());
    defer includesVectorB.deinit(allocator);
    @memset(includesVectorB.bytes, 0);

    var features = try BitVector.init(allocator, n_features);
    defer features.deinit(allocator);
    @memset(features.bytes, 0);

    for (0..1_000) |i| {
        random.bytes(std.mem.sliceAsBytes(states_buf[0..(i % config.stateSize() + 1)]));

        // getIncludesOnePass(S, states_buf, &includesOnePass);
        // getIncludesTwoPass(S, states_buf, &includesTwoPass);
        // getIncludesVector8(S, states_buf, &includesVector8);
        // getIncludesVectorA(S, states_buf, &includesVectorA);
        getIncludesVectorB(S, states_buf, &includesVectorB);

        // try testing.expectEqualSlices(u8, includesOnePass.bytes, includesTwoPass.bytes);
        // try testing.expectEqualSlices(u8, includesOnePass.bytes, includesVector8.bytes);
        // try testing.expectEqualSlices(u8, includesOnePass.bytes, includesVectorA.bytes);
        // try testing.expectEqualSlices(u8, includesOnePass.bytes, includesVectorB.bytes);

        const includes = includesVectorB;
        const v1 = evaluateIncludesU1(config, V, includes, features);
        const v2 = evaluateIncludesVector(config, V, includes, features);
        // if (v1 != v2) {
        //     std.debug.print(
        //         \\includes: {d}
        //         \\features: {d}
        //         \\evaluateIncludesU1: {d}
        //         \\evaluateIncludesVector: {d}
        //     , .{
        //         @as(@Vector(alignForward(n_features, 8), u1), @bitCast(includes.bytes[0..divCeil(n_features, 8)].*)),
        //         @as(@Vector(alignForward(n_features, 8), u1), @bitCast(features.bytes[0..divCeil(n_features, 8)].*)),
        //         v1,
        //         v2,
        //     });
        // }
        try testing.expectEqual(v1, v2);
    }
}

inline fn randomUniform(
    random: std.Random,
    comptime T: type,
) T {
    return switch (@typeInfo(T)) {
        .float => random.float(T),
        else => random.int(T),
    };
}


pub fn fit(
    config: Config,
    comptime S: type,
    states: *[]S,
    features: BitVector,
    target: bool,
    comptime V: type,
    t: V, // non-negative
    comptime F: type,
    comptime R: type,
    r: R,
    maybe_random: ?std.Random,
    comptime use_type_1a_feedback: bool,
    comptime use_type_1b_feedback: bool,
    comptime use_type_2_feedback: bool,
) i64 {
    _ = .{use_type_1a_feedback, use_type_1b_feedback, use_type_2_feedback};
    var votes: V = 0;
    for (0..2) |i_polarity| for (0..config.n_clauses) |i_clause| {
        const clause: u1 = getClauseU1(config, S, states, features, i_polarity, i_clause);
        if (i_polarity == 0) votes += clause else votes -= clause;
    };

    // Resource allocation...
    // if (target == true and votes >= t or target == false and votes < -t) {
    //     return votes;
    // }
    const p_clause_update: F = @as(F, @floatFromInt(@min(@max(if (target) -votes else votes, -t), t))) / (@as(F, @floatFromInt(2 * t))) + 0.5;

    // if (maybe_random) |random| if (random.float(F) >= p_clause_update) {
    //     return votes;
    // };

    for (0..2) |i_polarity| for (0..config.n_clauses) |i_clause| {
        // Use resource allocation:
        if (maybe_random) |random| if (random.float(F) >= p_clause_update) {
            continue;
        };

        const clause: u1 = getClauseU1(config, S, states, features, i_polarity, i_clause);

        for (0..2) |i_negated| for (0..config.n_features) |i_feature| {
            var literal: u1 = @intFromBool(features.getValue(i_feature));
            literal = if (i_negated == 0) literal else ~literal;

            const i_state = config.stateIndex(i_polarity, i_clause, i_negated, i_feature);
            if (@intFromBool(target) != i_polarity) {
                states.*[i_state] +|= @intFromBool(use_type_1a_feedback and (clause & literal) == 1);
                states.*[i_state] -|= @intFromBool(use_type_1b_feedback and (clause & literal) == 0 and (if (maybe_random) |random| randomUniform(random, R) >= r else true));
            } else {
                const excluded = states.*[i_state] < 0;
                states.*[i_state] +|= @intFromBool(use_type_2_feedback and (clause == 1) and (literal == 0) and excluded and (if (maybe_random) |random| randomUniform(random, R) < r else true));
            }
        };
    };

    return votes;
}

test fit {
    const allocator = std.testing.allocator;
    const config = Config.init(1, 4);
    const S = i8;
    const test_states = [_]S{
        1,  1,
        1, -1,
       -1,  1,
       -1, -1,
        1,  1,
        1, -1,
       -1,  1,
       -1, -1,
    };
    // var includes = try BitVector.init(allocator, states.len);
    // defer includes.deinit(allocator);
    var features = try BitVector.init(allocator, config.n_features);
    defer features.deinit(allocator);
    features.setValue(0, true);

    // get_includes(S, &states, &includes);
    // const votes = evaluate(config, u8, features, includes);
    // std.debug.print("votes: {}\n", .{votes});

    // feedback 1a
    // if (false)
    {
        var states = try allocator.alloc(S, test_states.len);
        defer allocator.free(states);
        for (0..test_states.len) |i| {
            states[i] = test_states[i];
        }
        _ = fit(config, S, &states, features, true, i32, 8, f32, f32, 0.75, null, true, false, false);
        const expected = [_]S{
            1,  1,
            2, -1,
           -1,  1,
            0, -1,
            1,  1,
            1, -1,
           -1,  1,
           -1, -1,
        };
        std.debug.print(
            \\type 1a feedback
            \\original: {d}
            \\     got: {d}
            \\expected: {d}
            \\
        , .{test_states, states, expected});
        for (0..states.len) |i| {
            try testing.expectEqual(expected[i], states[i]);
        }
    }
    // feedback 1b
    // if (false)
    {
        var states = try allocator.alloc(S, test_states.len);
        defer allocator.free(states);
        for (0..test_states.len) |i| {
            states[i] = test_states[i];
        }
        _ = fit(config, S, &states, features, true, i32, 8, f32, f32, 0.75, null, false, true, false);
        const expected = [_]S{
            0,  0,
            1, -2,
           -2,  0,
           -1, -2,
            1,  1,
            1, -1,
           -1,  1,
           -1, -1,
        };
        // std.debug.print("expected: {d}\ngot: {d}\n", .{expected, states});
        std.debug.print(
            \\type 1b feedback
            \\original: {d}
            \\     got: {d}
            \\expected: {d}
            \\
        , .{test_states, states, expected});
        for (0..states.len) |i| {
            try testing.expectEqual(expected[i], states[i]);
        }
    }
    // feedback 2
    // if (false)
    {
        var states = try allocator.alloc(S, test_states.len);
        defer allocator.free(states);
        for (0..test_states.len) |i| {
            states[i] = test_states[i];
        }
        _ = fit(config, S, &states, features, true, i32, 8, f32, f32, 0.75, null, false, false, true);
        const expected = [_]S{
            1,  1,
            1, -1,
           -1,  1,
           -1, -1,
            1,  1,
            1,  0,
           -1,  1,
           -1,  0,
        };
        std.debug.print(
            \\type 2 feedback
            \\original: {d}
            \\     got: {d}
            \\expected: {d}
            \\
        , .{test_states, states, expected});
        for (0..states.len) |i| {
            try testing.expectEqual(expected[i], states[i]);
        }
    }
}

const std = @import("std");
const testing = std.testing;
