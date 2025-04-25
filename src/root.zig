pub fn divCeil(a: anytype, b: anytype) @TypeOf(a) {
    return (a + b - 1) / b;
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
        return 2 * 2 * self.n_features * self.n_clauses;
    }

    pub fn stateIndex(
        self: Config,
        i_polarity: usize,
        i_negated: usize,
        i_feature: usize,
        i_clause: usize,
    ) usize {
        std.debug.assert(i_polarity < 2);
        std.debug.assert(i_negated < 2);
        std.debug.assert(i_feature < self.n_features);
        std.debug.assert(i_clause < self.n_clauses);
        const stride_feature = self.n_clauses;
        const stride_negated = stride_feature * self.n_features;
        const stride_polarity = stride_negated * 2;
        return stride_polarity * i_polarity + stride_negated * i_negated + stride_feature * i_feature + i_clause;
    }
};


/// 0.042881s
pub fn getIncludesOnePass(comptime S: type, states: []S, includes: *BitVector) void {
    std.debug.assert(@typeInfo(S).int.signedness == .signed);
    std.debug.assert(states.len == includes.len);
    for (0..states.len) |i_state|
        includes.setValue(i_state, states[i_state] >= 0);
}

/// 0.009739s
pub fn getIncludesTwoPass(comptime S: type, states: []S, includes: *BitVector) void {
    std.debug.assert(@typeInfo(S).int.signedness == .signed);
    std.debug.assert(states.len == includes.len);
    @memset(includes.bytes, 0);
    for (0..states.len) |i_state|
        if (states[i_state] >= 0)
            includes.setValueTrue(i_state);
}

/// 0.001773s
pub fn getIncludesVector8(comptime S: type, states: []S, includes: *BitVector) void {
    std.debug.assert(@typeInfo(S).int.signedness == .signed);
    std.debug.assert(states.len == includes.len);
    var i: usize = 0;
    while (i < states.len) : (i += 8) {
        const threshold: @Vector(8, S) = @splat(@as(S, 0));
        const states_vec: @Vector(8, S) = @bitCast(states[i .. i + 8][0..8].*);
        const includes_vec: @Vector(8, u1) = @bitCast(states_vec >= threshold);
        includes.bytes[i / 8] = @bitCast(includes_vec);
    }
}

/// 0.001386s
pub fn getIncludesVector(comptime S: type, states: []S, includes: *BitVector) void {
    std.debug.assert(@typeInfo(S).int.signedness == .signed);
    std.debug.assert(states.len == includes.len);
    var i: usize = 0;
    const l = std.simd.suggestVectorLength(S) orelse 1;
    while (i < states.len) : (i += l) {
        const threshold: @Vector(l, S) = @splat(@as(S, 0));
        const states_vec: @Vector(l, S) = @bitCast(states[i .. i + l][0..l].*);
        const includes_vec: @Vector(l, u1) = @bitCast(states_vec >= threshold);
        // If l is byte-aligned, pack 8 lanes into one u8 and write L/8 bytes at once
        if (l >= 8 and (l % 8) == 0) {
            const bytes_vec: @Vector(l/8, u8) = @bitCast(includes_vec);
            const dest_vec_ptr: *@Vector(l/8, u8) = @alignCast(@ptrCast(&includes.bytes[i / 8]));
            dest_vec_ptr.* = bytes_vec;
        } else {
            // Otherwise fall back lane‐by‐lane
            var j: usize = 0;
            while (j < l) : (j += 1) {
                // lanes of u1 are 0 or 1, so != 0 gives true/false
                includes.setValue(i + j, includes_vec[j] != 0);
            }
        }
    }
}

pub const getIncludes = getIncludesVector;

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
        const exclude: u1 = ~@intFromBool(includes.getValue(i_state));
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
    }
}

pub fn evaluateIncludesVector(
    config: Config,
    comptime V: type,
    includes: BitVector,
    features: BitVector,
) V {
    // only signed V supported
    if (@typeInfo(V).int.signedness != .signed) unreachable;
    var votes: V = 0;

    // number of clauses we can do in parallel
    const C = std.simd.suggestVectorLength(u1) orelse 1;
    std.debug.assert(config.n_clauses % C == 0);
    const C_bytes = C / 8;  // how many bytes that is

    inline for (0..2) |i_polarity| {
        var clause_off: usize = 0;
        // step through clauses C at a time
        while (clause_off < config.n_clauses) : (clause_off += C) {
            // mask of “alive” clauses in this block
            var mask: @Vector(C, u1) = @splat(1);

            // for each feature, AND in the survive‐bits
            for (0..config.n_features) |f| {
                // broadcast feature bit
                const fb: u1 = @intFromBool(features.getValue(f));
                const fv: @Vector(C, u1) = @splat(fb);
                const fnv: @Vector(C, u1) = ~fv;

                // compute bit‐offsets of this feature’s clause‐block
                const base_pos = config.stateIndex(i_polarity, 0, f, 0);
                const base_neg = config.stateIndex(i_polarity, 1, f, 0);
                const bitp = base_pos + clause_off;
                const bitn = base_neg + clause_off;
                const bytep = bitp / 8;
                const byten = bitn / 8;

                // slice out exactly C_bytes from the packed .bytes
                const slice_pos = includes.bytes[bytep .. bytep + C_bytes];
                const slice_neg = includes.bytes[byten .. byten + C_bytes];

                // reinterpret as C lanes of u1
                const ip: @Vector(C, u1) = @bitCast(slice_pos[0..C_bytes].*);
                const in_: @Vector(C, u1) = @bitCast(slice_neg[0..C_bytes].*);

                // (~ip | fv) & (~in | !fv)
                mask &= (~ip | fv) & (~in_ | fnv);

                // early‐exit if none survive
                // if (@popCount(mask) == 0) break;
                if (@reduce(.And, mask) == 0) break;
            }

            // how many survivors?
            // const d: usize = @popCount(mask);
            const d: usize = @reduce(.Add, mask);
            const dv: V = @intCast(d);
            if (i_polarity == 0) votes += dv else votes -= dv;
        }
    }

    return votes;
}

// pub const evaluateIncludes = evaluateIncludesU1;
pub const evaluateIncludes = evaluateIncludesVector;

test "fuzz getIncludes() and evaluate()" {
    const allocator = std.testing.allocator;
    const n_features = 16;
    const n_clauses  = 32;
    const config = Config.init(n_features, n_clauses);
    const S = i8;   // automata type
    const V = i32;  // vote‐count return type

    var prng = std.Random.DefaultPrng.init(111111111);
    const random = prng.random();

    const states_buf = try allocator.alloc(S, config.stateSize());
    defer allocator.free(states_buf);

    var includesOnePass = try BitVector.init(allocator, config.stateSize());
    defer includesOnePass.deinit(allocator);
    @memset(includesOnePass.bytes, 0);

    var includesTwoPass = try BitVector.init(allocator, config.stateSize());
    defer includesTwoPass.deinit(allocator);
    @memset(includesTwoPass.bytes, 0);

    var includesVector8 = try BitVector.init(allocator, config.stateSize());
    defer includesVector8.deinit(allocator);
    @memset(includesVector8.bytes, 0);

    var includesVector = try BitVector.init(allocator, config.stateSize());
    defer includesVector.deinit(allocator);
    @memset(includesVector.bytes, 0);

    var features = try BitVector.init(allocator, n_features);
    defer features.deinit(allocator);
    @memset(features.bytes, 0);

    for (0..10_000) |_| {
        random.bytes(std.mem.sliceAsBytes(states_buf));

        getIncludesOnePass(S, states_buf, &includesOnePass);

        getIncludesTwoPass(S, states_buf, &includesTwoPass);
        try testing.expectEqualSlices(u8, includesOnePass.bytes, includesTwoPass.bytes);

        getIncludesVector8(S, states_buf, &includesVector8);
        try testing.expectEqualSlices(u8, includesOnePass.bytes, includesVector8.bytes);

        getIncludesVector(S, states_buf, &includesVector);
        try testing.expectEqualSlices(u8, includesOnePass.bytes, includesVector.bytes);

        _ = .{config, V, features, includesOnePass, includesTwoPass, includesVector8};
        // const v1 = evaluateCompiledU1(config, V, includesOnePass, features);
        // const v2 = evaluateCompiledU8(config, V, includesOnePass, features);
        // try testing.expectEqual(v1, v2);
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
    _ = .{config, states, features, target, t, F, r, maybe_random};
    return 0;
    // var votes: V = 0;
    // for (0..2) |i_polarity| for (0..config.n_clauses) |i_clause| {
    //     const clause: u1 = getClauseU1(config, S, states, features, i_polarity, i_clause);
    //     if (i_polarity == 0) votes += clause else votes -= clause;
    // };
    //
    // // Resource allocation...
    // // if (target == true and votes >= t or target == false and votes < -t) {
    // //     return votes;
    // // }
    // const p_clause_update: F = @as(F, @floatFromInt(@min(@max(if (target) -votes else votes, -t), t))) / (@as(F, @floatFromInt(2 * t))) + 0.5;
    //
    // // if (maybe_random) |random| if (random.float(F) >= p_clause_update) {
    // //     return votes;
    // // };
    //
    // for (0..2) |i_polarity| for (0..config.n_clauses) |i_clause| {
    //     // Use resource allocation:
    //     if (maybe_random) |random| if (random.float(F) >= p_clause_update) {
    //         continue;
    //     };
    //
    //     const clause: u1 = getClauseU1(config, S, states, features, i_polarity, i_clause);
    //
    //     for (0..2) |i_negated| for (0..config.n_features) |i_feature| {
    //         var literal: u1 = @intFromBool(features.getValue(i_feature));
    //         literal = if (i_negated == 0) literal else ~literal;
    //
    //         const i_state = config.stateIndex(i_polarity, i_clause, i_negated, i_feature);
    //         if (@intFromBool(target) != i_polarity) {
    //             states.*[i_state] +|= @intFromBool(use_type_1a_feedback and (clause & literal) == 1);
    //             states.*[i_state] -|= @intFromBool(use_type_1b_feedback and (clause & literal) == 0 and (if (maybe_random) |random| randomUniform(random, R) >= r else true));
    //         } else {
    //             const excluded = states.*[i_state] < 0;
    //             states.*[i_state] +|= @intFromBool(use_type_2_feedback and (clause == 1) and (literal == 0) and excluded and (if (maybe_random) |random| randomUniform(random, R) < r else true));
    //         }
    //     };
    // };
    //
    // return votes;
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
