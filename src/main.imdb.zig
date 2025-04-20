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
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) {
            std.debug.print("leak\n", .{});
        }
    }

    // const train_filepath = "/home/i/Downloads/Telegram Desktop/IMDBTrainingData.txt";
    // const test_filepath = "/home/i/Downloads/Telegram Desktop/IMDBTestData.txt";
    // const train_bin_filepath = "/home/i/Downloads/Telegram Desktop/IMDBTrainingData.bin";
    // const test_bin_filepath = "/home/i/Downloads/Telegram Desktop/IMDBTestData.bin";
    // const train_bin_filepath = "/home/i/p/tm2.zig/test3.3.bin";
    // const test_bin_filepath = "/home/i/p/tm2.zig/test3.3.bin";
    const train_bin_filepath = "IMDBTrainingData.bin";
    const test_bin_filepath = "IMDBTestData.bin";

    // const n_clauses = 1024;
    // const n_clauses = 128;
    const n_clauses = 16;
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
    const t = 4;
    // const t = 8;
    // const t = 11;
    const r_float = 0.85;
    const R = u64;
    // const r_scale: comptime_int = 1 << 64;
    const r_scale: comptime_int = 1 << @typeInfo(R).int.bits;
    const r_int: u64 = @intFromFloat(r_float * r_scale);
    // const r: f64 = r_float;
    const r = r_int;
    const epochs: usize = 1000;
    const state_size_bits = 2 * n_clauses * 2 * n_features * @bitSizeOf(S);
    std.debug.print("S: {}, t: {d}, r_float: {d}, r_scale: {d}, r_int: {d}, r: {d}, n_features: {d}, n_clauses: {d}, total size: 2*n_clauses*2*n_features*bitSizeOf(S) = {d} bits, {d} bytes, {d} KiB, {d} MiB\n", .{S, t, r_float, r_scale, r_int, r, n_features, n_clauses, state_size_bits, state_size_bits / 8, state_size_bits / 8 / 1024, state_size_bits / 8 / 1024 / 1024});

    var prng = std.Random.DefaultPrng.init(std.crypto.random.int(u64));
    const random = prng.random();
    // const random = std.crypto.random;

    var pool: std.Thread.Pool = undefined;
    try pool.init(.{
        .allocator = allocator,
        .n_jobs = 8,
    });
    defer pool.deinit();

    var best_test_accuracy: f64 = 0.0;

    for (0..epochs) |epoch| {
        const epoch_start_time = std.time.milliTimestamp();
        var epoch_error: usize = 0;
        const n_train_samples = train_bytes.len / example_size_bytes;
        var epoch_cum_test_time: i64 = 0;
        for (0..n_train_samples) |idx| {

            //
            // Test
            //
            if ((idx > 0 and idx % 64 == 0) or idx == n_train_samples - 1) {
                const train_accuracy = 100.0 * (1.0 - (@as(f64, @floatFromInt(epoch_error)) / @as(f64, @floatFromInt(idx + 1))));
                const train_end_time = std.time.milliTimestamp();
                const train_elapsed_time = train_end_time - epoch_start_time - epoch_cum_test_time;
                const train_s_elapsed = @as(f64, @floatFromInt(train_elapsed_time)) / 1000.0;
                const train_it_s = @as(f64, @floatFromInt(idx + 1)) / (@as(f64, @floatFromInt(train_elapsed_time)) / 1000.0);

                const compile_start_time = std.time.milliTimestamp();
                root.get_includes(S, states, &includes);
                const compile_end_time = std.time.milliTimestamp();
                const compile_elapsed_time = compile_end_time - compile_start_time;
                const compile_s_elapsed = @as(f64, @floatFromInt(compile_elapsed_time)) / 1000.0;

                var test_errors: usize = 0;
                const test_start_time = std.time.milliTimestamp();

                const n_test_samples = test_bytes.len / example_size_bytes;
                for (0..n_test_samples) |test_idx| {
                    const test_offset = test_idx * example_size_bytes;
                    const test_example_data = test_bytes[test_offset .. test_offset + example_size_bytes];
                    const test_features_slice = test_example_data[0 .. example_size_bytes - 1];
                    const test_features = root.BitVector.from(n_features, @constCast(@ptrCast(test_features_slice)));
                    const test_label = if (test_example_data[example_size_bytes - 1] == 0) false else true;
                    const test_pred = root.evaluate(config, V, test_features, includes) >= 0;
                    const test_error = test_pred != test_label;
                    test_errors += @intFromBool(test_error);
                }
                const test_idx = n_test_samples - 1;
                const test_accuracy = 100.0 * (1.0 - (@as(f64, @floatFromInt(test_errors)) / @as(f64, @floatFromInt(test_idx + 1))));
                if (test_accuracy > best_test_accuracy) {
                    best_test_accuracy = test_accuracy;
                }
                const end_time = std.time.milliTimestamp();
                const elapsed_time = end_time - test_start_time;
                epoch_cum_test_time += elapsed_time + compile_elapsed_time;
                const s_elapsed = @as(f64, @floatFromInt(elapsed_time)) / 1000.0;
                const it_s = @as(f64, @floatFromInt(test_idx + 1)) / (@as(f64, @floatFromInt(elapsed_time)) / 1000.0);
                std.debug.print(
                    "epoch: {d}/{d}, train sample: {d}/{d}, accuracy: {d:.2}% and {d:.2}% (best: {d:.2}%), elapsed: {d:.2}s, {d:.4}s, {d:.3}s, perf: {d:.2}it/s and {d:.2}it/s",
                    .{ epoch + 1, epochs, idx + 1, n_train_samples, train_accuracy, test_accuracy, best_test_accuracy, train_s_elapsed, compile_s_elapsed, s_elapsed, train_it_s, it_s },
                );
                if (idx == n_train_samples - 1) {
                    std.debug.print("\n", .{});
                } else {
                    std.debug.print("\r", .{});
                }
            }

            //
            // Train
            //
            const train_offset = idx * example_size_bytes;
            const train_example_data = train_bytes[train_offset .. train_offset + example_size_bytes];
            const train_features_slice = train_example_data[0 .. example_size_bytes - 1];
            const train_features = root.BitVector.from(n_features, @constCast(@ptrCast(train_features_slice)));
            const train_label = if (train_example_data[example_size_bytes - 1] == 0) false else true;

            const train_vote = root.fit(config, S, &states, &includes, train_features, train_label, V, t, F, @TypeOf(r), r, random, true, true, true);
            // const train_vote = root.fit_thread_polarity(config, S, &states, &includes, train_features, train_label, t, @TypeOf(r), r, random, true, true, true);
            // const train_vote = root.fit_thread_clause(config, S, &states, &includes, train_features, train_label, t, @TypeOf(r), r, random, true, true, true);
            // const train_vote = root.fit_poolwg_polarity(config, &pool, S, &states, &includes, train_features, train_label, t, @TypeOf(r), r, random, true, true, true);
            // const train_vote = root.fit_poolwg_clause(config, &pool, S, &states, &includes, train_features, train_label, t, @TypeOf(r), r, random, true, true, true);
            // const train_vote = root.fit_pool_polarity(config, &pool, S, &states, &includes, train_features, train_label, t, @TypeOf(r), r, random, true, true, true);
            // const train_vote = root.fit_pool_clause(config, &pool, S, &states, &includes, train_features, train_label, t, @TypeOf(r), r, random, true, true, true);

            if (idx % 5000 == 0) {
                std.debug.print("\n", .{});
            }

            const train_pred = train_vote >= 0;
            const train_error = train_pred != train_label;
            epoch_error += @intFromBool(train_error);
        }
    }
}
