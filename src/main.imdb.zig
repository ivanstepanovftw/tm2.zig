const std = @import("std");
const root = @import("root.zig");

pub const SimpleLibCMappedFile = struct {
    file: std.fs.File,
    data: []u8,

    pub fn init(path: []const u8, write: bool, with_size: ?usize) !SimpleLibCMappedFile {
        const file = try std.fs.cwd().openFile(path, .{ .mode = if (write) .read_write else .read_only });
        errdefer file.close();
        const prot: c_uint = if (write) std.os.linux.PROT.READ | std.os.linux.PROT.WRITE else std.os.linux.PROT.READ;
        const flags = std.os.linux.MAP{ .TYPE = std.os.linux.MAP_TYPE.PRIVATE };
        const size: usize = with_size orelse (try file.metadata()).size();
        const ptr = std.c.mmap(null, size, prot, flags, file.handle, 0);
        if (ptr == std.c.MAP_FAILED) return error.MappedFileFailed;
        const data = @as([*]u8, @ptrCast(ptr))[0..size];
        return SimpleLibCMappedFile{ .file = file, .data = data };
    }

    pub fn deinit(self: SimpleLibCMappedFile) void {
        const ptr: *align(4096) const anyopaque = @alignCast(@ptrCast(self.data.ptr));
        _ = std.c.munmap(ptr, self.data.len);
        self.file.close();
    }
};

pub fn convertTxtToBin(comptime n_cols: comptime_int, input_path: []const u8, output_path: []const u8) !void {
    var input_mapped = try SimpleLibCMappedFile.init(input_path, false, null);
    defer input_mapped.deinit();

    const output_file = try std.fs.cwd().createFile(output_path, .{ .truncate = true });
    defer output_file.close();
    var writer = output_file.writer();

    const n_bytes = root.divCeil(n_cols, 8);
    var bytes: [n_bytes]u8 = undefined;

    var line_it = std.mem.splitScalar(u8, input_mapped.data, '\n');
    while (line_it.next()) |line| {
        var tokens = std.mem.tokenizeAny(u8, line, " ");
        var i: usize = 0;
        @memset(&bytes, 0);
        while (i < n_cols) : (i += 1) {
            // Expect each token to be "0" or "1"; default to false if missing.
            if (tokens.next()) |token| {
                const value = token[0] == '1';
                const i_byte = i / 8;
                const i_mask = i % 8;
                const mask = @as(u8, 1) << @as(u3, @intCast(i_mask));
                if (value) {
                    bytes[i_byte] |= mask;
                }
            }
        }
        try writer.writeAll(&bytes);
    }
}

pub fn main() !void {
    // var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    // defer arena.deinit();
    // const allocator = arena.allocator();

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer if (gpa.deinit() != .ok) @panic("leak");

    // const train_filepath = "/home/i/Downloads/Telegram Desktop/IMDBTrainingData.txt";
    // const test_filepath = "/home/i/Downloads/Telegram Desktop/IMDBTestData.txt";
    // const train_bin_filepath = "/home/i/Downloads/Telegram Desktop/IMDBTrainingData.bin";
    // const test_bin_filepath = "/home/i/Downloads/Telegram Desktop/IMDBTestData.bin";
    // const train_bin_filepath = "/home/i/p/tm2.zig/test3.3.bin";
    // const test_bin_filepath = "/home/i/p/tm2.zig/test3.3.bin";
    const train_bin_filepath = "IMDBTrainingData.bin";
    const test_bin_filepath = "IMDBTestData.bin";
    // const test_bin_filepath = "IMDBTrainingData.bin";

    // const n_clauses = 2048;
    const n_clauses = 128;
    // const n_clauses = 16;
    const n_features = 40_000;
    // const n_features = 3;
    // const convert = true;
    // const convert = false;
    // if (convert) {
    //     std.debug.print("convert.\n", .{});
    //     try convertTxtToBin(n_features + 1, train_filepath, train_bin_filepath);
    //     std.debug.print("train done.\n", .{});
    //     try convertTxtToBin(n_features + 1, test_filepath, test_bin_filepath);
    //     std.debug.print("test done.\n", .{});
    //     // try convertTxtToBin(4, "/home/i/p/tm2.zig/test3.3.txt", "/home/i/p/tm2.zig/test3.3.bin");
    //     // try convertTxtToBin(5, "/home/i/p/tm2.zig/test4.4.txt", "/home/i/p/tm2.zig/test4.4.bin");
    //     // try convertTxtToBin(6, "/home/i/p/tm2.zig/test5.4.txt", "/home/i/p/tm2.zig/test5.4.bin");
    //     std.debug.print("test done.\n", .{});
    //     return;
    // }
    // try convertTxtToBin(4, "/home/i/p/tm2.zig/test3.3.txt", "/home/i/p/tm2.zig/test3.3.bin");

    const example_size_bits = n_features + 1;
    const example_size_bytes = root.divCeil(example_size_bits, 8);

    // const train_map = try SimpleLibCMappedFile.init(train_bin_filepath, false, null);
    // defer train_map.deinit();
    // const test_map = try SimpleLibCMappedFile.init(test_bin_filepath, false, null);
    // defer test_map.deinit();
    // const train_bytes = train_map.data;
    // const test_bytes = test_map.data;

    const train_file = try std.fs.cwd().openFile(train_bin_filepath, .{ .mode = .read_only });
    defer train_file.close();
    const test_file = try std.fs.cwd().openFile(test_bin_filepath, .{ .mode = .read_only });
    defer test_file.close();
    const train_bytes = try train_file.readToEndAlloc(allocator, 125025000);
    defer allocator.free(train_bytes);
    const test_bytes = try test_file.readToEndAlloc(allocator, 125025000);
    defer allocator.free(test_bytes);

    const config = root.Config.init(n_features, n_clauses);

    const S = i8;
    // const S = i16;
    var states = try allocator.alloc(S, config.stateSize());
    defer allocator.free(states);
    @memset(states, -1);

    var includes = try root.BitVector.init(allocator, config.stateSize());
    defer includes.deinit(allocator);
    @memset(includes.bytes, 0);

    const V = i32;
    const F = f32;
    const t = 8;
    const r_float: f32 = 0.65;
    const RInt = u64;  // r value scale for integer randomness
    const r_scale = 1 << @typeInfo(RInt).int.bits;
    const r_int: RInt = @intFromFloat(r_float * r_scale);
    // const r = r_float;
    const r = r_int;
    const n_epochs: usize = 1000;
    std.debug.print("S: {}, t: {d}, r: {d} ({d}), n_features: {d}, n_clauses: {d}, state size: {d} ({d:.1})\n", .{S, t, r, r_float, n_features, n_clauses, config.stateSize(), std.fmt.fmtIntSizeBin(config.stateSize() * @sizeOf(S))});

    var prng = std.Random.DefaultPrng.init(std.crypto.random.int(u64));
    const random = prng.random();
    // const random = std.crypto.random;
    _ = .{random};

    const thread_count = @max(1, (std.Thread.getCpuCount() catch 1) - 1);
    // const thread_count = 8;
    var pool: std.Thread.Pool = undefined;
    try pool.init(.{
        .allocator = allocator,
        .n_jobs = thread_count,
    });
    defer pool.deinit();
    var wg: std.Thread.WaitGroup = .{};

    var best_test_accuracy: F = 0.0;
    var training_timer = try std.time.Timer.start();
    const test_every_n_samples = 50000;

    for (0..n_epochs) |i_epoch| {
        var epoch_timer = try std.time.Timer.start();
        var train_delta_timer = try std.time.Timer.start();
        var train_duration_ns: u64 = 0;
        var train_errors = std.atomic.Value(usize).init(0);

        const n_train_samples = train_bytes.len / example_size_bytes;
        for (0..n_train_samples) |i_train_sample| {
            //
            // Test
            //
            if ((i_train_sample > 0 and i_train_sample % test_every_n_samples == 0) or i_train_sample == n_train_samples - 1) {
                wg.wait();
                train_duration_ns += train_delta_timer.read();
                defer train_delta_timer.reset();
                wg.reset();

                const train_accuracy = 100.0 * (1.0 - (@as(F, @floatFromInt(train_errors.load(.monotonic))) / @as(F, @floatFromInt(i_train_sample + 1))));
                const train_duration = @as(F, @floatFromInt(train_duration_ns)) / @as(F, @floatFromInt(std.time.ns_per_s));
                const train_perf = @as(F, @floatFromInt(i_train_sample + 1)) / train_duration;

                var compile_timer = try std.time.Timer.start();
                root.getIncludes(S, states, &includes);
                const compile_duration = @as(F, @floatFromInt(compile_timer.read())) / @as(F, @floatFromInt(std.time.ns_per_s));

                // var test_errors: usize = 0;
                var test_errors = std.atomic.Value(usize).init(0);
                var test_timer = try std.time.Timer.start();

                const n_test_samples = test_bytes.len / example_size_bytes;
                // const n_test_samples = test_bytes.len / example_size_bytes / 10000;
                for (0..n_test_samples) |i_test_sample| {
                    const test_offset = i_test_sample * example_size_bytes;
                    const test_example_data = test_bytes[test_offset .. test_offset + example_size_bytes];
                    const test_features_slice = test_example_data[0 .. example_size_bytes - 1];
                    const test_label = if (test_example_data[example_size_bytes - 1] == 0) false else true;

                    // const test_features = root.BitVector.from(n_features, @constCast(@ptrCast(test_features_slice)));
                    // const test_pred = root.evaluateIncludes(config, V, includes, test_features) >= 0;
                    // const test_error = test_pred != test_label;
                    // test_errors += @intFromBool(test_error);

                    pool.spawnWg(&wg,
                        struct {
                            fn run(data: []u8, lbl: bool, cfg: root.Config, inc: *root.BitVector, err_cnt: *std.atomic.Value(usize)) void {
                                const features = root.BitVector.from(n_features, @constCast(@ptrCast(data)));
                                const pred = root.evaluateIncludes(cfg, V, inc.*, features) >= 0;
                                if (pred != lbl) {
                                    _ = err_cnt.fetchAdd(1, .monotonic);
                                }
                            }
                        }.run,
                        .{ test_features_slice, test_label, config, &includes, &test_errors }
                    );
                }
                wg.wait();
                const test_duration_ns = test_timer.read();
                wg.reset();

                const i_test_sample = n_test_samples - 1;
                // const test_accuracy = 100.0 * (1.0 - (@as(F, @floatFromInt(test_errors)) / @as(F, @floatFromInt(i_test_sample + 1))));
                const test_accuracy = 100.0 * (1.0 - (@as(F, @floatFromInt(test_errors.load(.monotonic))) / @as(F, @floatFromInt(i_test_sample + 1))));
                if (test_accuracy > best_test_accuracy) {
                    best_test_accuracy = test_accuracy;
                }
                const test_duration = @as(F, @floatFromInt(test_duration_ns)) / @as(F, @floatFromInt(std.time.ns_per_s));
                const test_perf = @as(F, @floatFromInt(i_test_sample + 1)) / test_duration;
                const epoch_duration = @as(F, @floatFromInt(epoch_timer.read())) / @as(F, @floatFromInt(std.time.ns_per_s));
                const training_duration = @as(F, @floatFromInt(training_timer.read())) / @as(F, @floatFromInt(std.time.ns_per_s));
                if (i_epoch % 5 == 0) {
                    std.debug.print("{s:<5} | {s:<6} | {s:<6} | {s:<7} | {s:<7} | {s:<7} | {s:<7} | {s:<10} | {s:<9} | {s:<9} | {s:<9} | {s:<9} | {s:<10} | {s}\n",
                    .{"epoch", "epochs", "sample", "samples", "train", "test", "best", "training", "epoch", "train", "compile", "test", "fit perf", "test perf"});
                }
                std.debug.print("{d:>5} | {d:>6} | {d:>6} | {d:>7} | {d:>6.2}% | {d:>6.2}% | {d:>6.2}% | {d:>9.0}s | {d:>8.3}s | {d:>8.3}s | {d:>8.6}s | {d:>8.6}s | {d:>6.2}it/s | {d:.2}it/s\n",
                    .{ i_epoch + 1, n_epochs, i_train_sample + 1, n_train_samples, train_accuracy, test_accuracy, best_test_accuracy, training_duration, epoch_duration, train_duration, compile_duration, test_duration, train_perf, test_perf },
                );
                // if (i_train_sample == n_train_samples - 1) {
                //     std.debug.print("\n", .{});
                // } else {
                //     std.debug.print("\r", .{});
                // }
            }

            //
            // Train
            //
            const train_offset = i_train_sample * example_size_bytes;
            const train_example_data = train_bytes[train_offset .. train_offset + example_size_bytes];
            const train_features_slice = train_example_data[0 .. example_size_bytes - 1];
            // const train_features = root.BitVector.from(n_features, @constCast(@ptrCast(train_features_slice)));
            const train_label = if (train_example_data[example_size_bytes - 1] == 0) false else true;

            // const train_vote = root.fit(config, S, &states, &includes, train_features, train_label, V, t, F, @TypeOf(r), r, random, true, true, true);
            // const train_pred = train_vote >= 0;
            // const train_error = train_pred != train_label;
            // epoch_error += @intFromBool(train_error);
            pool.spawnWg(&wg,
                struct {
                    fn run(data: []u8, lbl: bool, config_: root.Config, states_: *[]S, err_cnt: *std.atomic.Value(usize), prng_: std.Random.DefaultPrng) void {
                        var prng__ = prng_;
                        const random_ = prng__.random();
                        const features = root.BitVector.from(n_features, @constCast(@ptrCast(data)));
                        const vote = root.fit(config_, S, states_, features, lbl, V, t, F, @TypeOf(r), r, random_, true, true, true);
                        const pred = vote >= 0;
                        if (pred != lbl) {
                            _ = err_cnt.fetchAdd(1, .monotonic);
                        }
                    }
                }.run,
                .{ train_features_slice, train_label, config, &states, &train_errors, blk: {
                    var prng_state: std.Random.DefaultPrng = undefined;
                    random.bytes(std.mem.asBytes(&prng_state)[0..@sizeOf(@TypeOf(prng_state))]);
                    break :blk prng_state;
                }}
            );
            // if (i_train_sample % 5000 == 0) {
            //     std.debug.print("\n", .{});
            // }
        }
    }
}
