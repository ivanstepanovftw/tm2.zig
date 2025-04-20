pub fn divCeil(a: anytype, b: anytype) @TypeOf(a) {
    return (a + b - 1) / b;
}

/// Unsafe unmanaged binary vector.
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
        if (index >= self.len) unreachable;
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
        if (index >= self.len) unreachable;
        const i_byte = index / 8;
        const i_mask = index % 8;
        const mask: u8 = @as(u8, 1) << @as(u3, @intCast(i_mask));
        self.bytes[i_byte] |= mask;
    }

    pub inline fn getValue(self: Self, index: usize) bool {
        if (index >= self.len) unreachable;
        const i_byte = index / 8;
        const i_mask = index % 8;
        const mask: u8 = @as(u8, 1) << @as(u3, @intCast(i_mask));
        return self.bytes[i_byte] & mask != 0;
    }
};

test BitVector {
    const allocator = std.testing.allocator;
    const n = 3;
    var bv = try BitVector.init(allocator, n);
    defer bv.deinit(allocator);

    for (0..n) |i| {
        bv.setValue(i, i % 2 == 0);
    }

    for (0..n) |i| {
        const expected = i % 2 == 0;
        const actual = bv.getValue(i);
        try testing.expectEqual(expected, actual);
    }
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
        if (i_polarity >= 2) unreachable;
        if (i_clause >= self.n_clauses) unreachable;
        if (i_negated >= 2) unreachable;
        if (i_feature >= self.n_features) unreachable;
        const stride_negated = self.n_features;
        const stride_clause = stride_negated * 2;
        const stride_polarity = stride_clause * self.n_clauses;
        return i_feature + stride_negated * i_negated + stride_clause * i_clause + stride_polarity * i_polarity;
    }
};


/// 0.0420s
// pub fn get_includes(
//     comptime S: type,
//     states: []S,
//     includes: *BitVector,
// ) void {
//     // if (states.len != includes.len) unreachable;
//     for (0..includes.len) |i_state| {
//         includes.setValue(i_state, states[i_state] >= 0);
//     }
// }

/// 0.0060s
/// 0.0020s
pub fn get_includes(
    comptime S: type,
    states: []S,
    includes: *BitVector,
) void {
    if (states.len != includes.len) unreachable;

    // @memset(includes.bytes, 0);
    // for (0..includes.len) |i_state| {
    //     if (states[i_state] >= 0)
    //         includes.setValueTrue(i_state);
    // }

    var i: usize = 0;
    const l = 8;
    while (i < includes.len) : (i += l) {
        const threshold: @Vector(l, S) = @splat(@as(S, 0));
        const states_vec: @Vector(l, S) = @bitCast(states[i .. i + l][0..l].*);
        const includes_vec: @Vector(l, u1) = @bitCast(states_vec >= threshold);
        includes.bytes[i / 8] = @bitCast(includes_vec);
    }
}


pub fn get_clause_1(
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


pub fn evaluate(
    config: Config,
    comptime V: type,
    features: BitVector,
    includes: BitVector,
) V {
    var votes: V = 0;
    for (0..2) |i_polarity| for (0..divCeil(config.n_clauses, 8)) |i_clause_byte| {
        var clause: u8 = ~@as(u8, 0);
        for (0..2) |i_negated| for (0..divCeil(config.n_features, 8)) |i_feature_byte| {
            const i_state = config.stateIndex(i_polarity, i_clause_byte * 8, i_negated, i_feature_byte * 8);
            const include: u8 = includes.bytes[i_state / 8];
            const feature: u8 = features.bytes[i_feature_byte];
            const literal: u8 = if (i_negated == 0) feature else ~feature;
            clause &= ~include | literal;
        };
        const diff = @popCount(clause);
        if (i_polarity == 0) votes += diff else votes -= diff;
    };

    return votes;
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
    includes: *BitVector,
    features: BitVector,
    target: bool,
    comptime V: type,
    t: V, // non-negative
    comptime F: type,
    comptime R: type,
    r: R,
    random: ?std.Random,
    comptime use_type_1a_feedback: bool,
    comptime use_type_1b_feedback: bool,
    comptime use_type_2_feedback: bool,
) i64 {
    _ = .{t, r, random, includes, use_type_1a_feedback, use_type_1b_feedback, use_type_2_feedback};

    var votes: V = 0;
    for (0..2) |i_polarity| for (0..divCeil(config.n_clauses, 8)) |i_clause_byte| {
        var clause: u8 = ~@as(u8, 0);
        for (0..2) |i_negated| for (0..divCeil(config.n_features, 8)) |i_feature_byte| {
            const i_state = config.stateIndex(i_polarity, i_clause_byte * 8, i_negated, i_feature_byte * 8);
            const include: u8 = includes.bytes[i_state / 8];
            const feature: u8 = features.bytes[i_feature_byte];
            const literal: u8 = if (i_negated == 0) feature else ~feature;
            clause &= ~include | literal;
        };
        const diff = @popCount(clause);
        if (i_polarity == 0) votes += diff else votes -= diff;
    };

    // Resource allocation...
    if (target == true and votes >= t or target == false and votes < -t) {
        return votes;
    }
    const p_clause_update: F = @as(F, @floatFromInt(@min(@max(if (target) -votes else votes, -t), t))) / (@as(F, @floatFromInt(2 * t))) + 0.5;

    for (0..2) |i_polarity| for (0..config.n_clauses) |i_clause| {
        // Use resource allocation:
        if (random) |rng| if (rng.float(F) >= p_clause_update) {
            continue;
        };

        const clause: u1 = get_clause_1(config, S, states, features, i_polarity, i_clause);

        for (0..1) |i_negated| for (0..config.n_features) |i_feature| {
            var literal: u1 = @intFromBool(features.getValue(i_feature));
            literal = if (i_negated == 0) literal else ~literal;

            const i_state = config.stateIndex(i_polarity, i_clause, i_negated, i_feature);
            if (@intFromBool(target) != i_polarity) {
                states.*[i_state] +|= @intFromBool(use_type_1a_feedback and (clause & literal) == 1);
                states.*[i_state] -|= @intFromBool(use_type_1b_feedback and (clause & literal) == 0 and (if (random) |rng| randomUniform(rng, R) >= r else true));
            } else {
                const excluded = states.*[i_state] < 0;
                states.*[i_state] +|= @intFromBool(use_type_2_feedback and (clause == 1) and (literal == 0) and excluded and (if (random) |rng| randomUniform(rng, R) < r else true));
            }
        };
    };

    return votes;
}


const std = @import("std");
const testing = std.testing;
