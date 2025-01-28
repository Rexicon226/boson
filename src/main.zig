const std = @import("std");
const builtin = @import("builtin");
const fe = @import("fe.zig");
const assert = std.debug.assert;

const EPSILON = 1e-9;

const Flat = struct {
    inputs: []const Variable,
    instructions: []const Instruction,

    const Instruction = struct {
        dest: Variable,
        lhs: Variable,
        rhs: Variable,
        op: Op,

        const Op = enum {
            mul,
            add,
            set,
        };
    };

    fn setVar(
        array: []i32,
        variable: Variable,
        vars: std.AutoArrayHashMap(Variable, void),
    ) void {
        if (variable.isConstant()) {
            array[0] += @bitCast(variable.getConstant());
        } else {
            array[vars.getIndex(variable).?] += 1;
        }
    }

    fn getVar(
        variable: Variable,
        r: []const i32,
        vars: std.AutoArrayHashMap(Variable, void),
    ) i32 {
        if (variable.isConstant()) {
            return variable.getConstant();
        } else {
            return r[vars.getIndex(variable).?];
        }
    }

    fn solve(
        flat: *const Flat,
        variables: std.AutoArrayHashMap(Variable, void),
        allocator: std.mem.Allocator,
    ) ![]const i32 {
        var r = try allocator.alloc(i32, variables.count());
        @memset(r, 0);
        r[0] = 1;
        for (0..flat.inputs.len) |i| {
            r[i + 1] = 3; // set the input to 3, somewhat arbitrary value
        }
        for (flat.instructions) |inst| {
            switch (inst.op) {
                .set => r[variables.getIndex(inst.dest).?] = getVar(inst.lhs, r, variables),
                .add,
                .mul,
                => {
                    const rhs = getVar(inst.rhs, r, variables);
                    const lhs = getVar(inst.lhs, r, variables);
                    r[variables.getIndex(inst.dest).?] = switch (inst.op) {
                        .add => rhs + lhs,
                        .mul => rhs * lhs,
                        else => unreachable,
                    };
                },
            }
        }
        return r;
    }

    pub fn format(
        flat: Flat,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        try writer.writeAll("def foo(");
        for (flat.inputs, 0..) |input, i| {
            try writer.print("{}", .{input});
            if (i != flat.inputs.len - 1) try writer.writeAll(", ");
        }
        try writer.writeAll("):\n");

        for (flat.instructions) |inst| {
            try writer.writeByteNTimes(' ', 2);

            switch (inst.op) {
                .set => try writer.print("{} = {}", .{ inst.dest, inst.lhs }),
                .add,
                .mul,
                => try writer.print("{} = {} {s} {}", .{
                    inst.dest,
                    inst.lhs,
                    switch (inst.op) {
                        .add => "+",
                        .mul => "*",
                        else => unreachable,
                    },
                    inst.rhs,
                }),
            }

            try writer.writeByte('\n');
        }
    }
};

const Variable = enum(u64) {
    none,
    one,
    out,

    _,

    fn makeNew(counter: *u32) Variable {
        defer counter.* += 1;
        return @enumFromInt(counter.* + @typeInfo(Variable).Enum.fields.len);
    }

    fn newConstant(constant: i32) Variable {
        const unsigned = @as(u32, @bitCast(constant));
        return @enumFromInt(@as(u64, unsigned) << 32);
    }

    fn isConstant(variable: Variable) bool {
        return @intFromEnum(variable) >> 32 != 0;
    }

    fn getConstant(variable: Variable) i32 {
        assert(variable.isConstant());
        const unsigned: u32 = @truncate(@intFromEnum(variable) >> 32);
        return @bitCast(unsigned);
    }

    pub fn format(
        variable: Variable,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        if (variable.isConstant()) {
            const constant = variable.getConstant();
            try writer.print("{d}", .{constant});
            return;
        }
        switch (variable) {
            .none, .one, .out => |t| try writer.print("{s}", .{@tagName(t)}),
            else => try writer.print("%{d}", .{@intFromEnum(variable)}),
        }
    }
};

fn Qap(Field: type) type {
    const Fe = Finite(Field);
    return struct {
        rows: usize,
        columns: usize,

        a: []const Field,
        b: []const Field,
        c: []const Field,

        /// `Z = Π{i = N}(x - i)`
        // z: []const Fe,

        const Q = @This();

        fn transpose(
            allocator: std.mem.Allocator,
            src: []const i32,
            rows: usize,
            cols: usize,
        ) ![]const i32 {
            const result = try allocator.alloc(i32, rows * cols);
            for (0..rows) |i| {
                for (0..cols) |j| {
                    result[j * rows + i] = src[i * cols + j];
                }
            }
            return result;
        }

        fn fromFlat(flat: Flat, allocator: std.mem.Allocator) !Q {
            var used: std.AutoHashMapUnmanaged(Variable, void) = .{};
            defer used.deinit(allocator);
            for (flat.inputs) |input| {
                try used.putNoClobber(allocator, input, {});
            }

            var variables = vars: {
                var list = std.AutoArrayHashMap(Variable, void).init(allocator);
                try list.putNoClobber(.one, {});
                for (flat.inputs) |input| try list.putNoClobber(input, {});
                try list.putNoClobber(.out, {});
                for (flat.instructions) |inst| {
                    if (std.mem.indexOfScalar(Variable, flat.inputs, inst.dest) != null) continue;
                    if (inst.dest == .out) continue;
                    try list.putNoClobber(inst.dest, {});
                }
                break :vars list;
            };
            defer variables.deinit();

            const num_variables = variables.count();
            const total_size = flat.instructions.len * num_variables;
            const A = try allocator.alloc(i32, total_size);
            const B = try allocator.alloc(i32, total_size);
            const C = try allocator.alloc(i32, total_size);
            @memset(A, 0);
            @memset(B, 0);
            @memset(C, 0);

            for (flat.instructions, 0..) |inst, i| {
                const offset = i * num_variables;
                const a = A[offset..][0..num_variables];
                const b = B[offset..][0..num_variables];
                const c = C[offset..][0..num_variables];

                const gop = try used.getOrPut(allocator, inst.dest);
                if (gop.found_existing) {
                    std.debug.panic("variable already used: {}", .{inst.dest});
                }

                switch (inst.op) {
                    .set => {
                        a[variables.getIndex(inst.dest).?] += 1;
                        a[variables.getIndex(inst.lhs).?] -= 1;
                        b[0] += 1;
                    },
                    .add => {
                        c[variables.getIndex(inst.dest).?] = 1;
                        Flat.setVar(a, inst.lhs, variables);
                        Flat.setVar(a, inst.rhs, variables);
                        b[0] = 1;
                    },
                    .mul => {
                        c[variables.getIndex(inst.dest).?] = 1;
                        Flat.setVar(a, inst.lhs, variables);
                        Flat.setVar(b, inst.rhs, variables);
                    },
                }
            }

            const cols = flat.instructions.len;
            const rows = num_variables;

            const At = try transpose(allocator, A, cols, rows);
            const Bt = try transpose(allocator, B, cols, rows);
            const Ct = try transpose(allocator, C, cols, rows);

            const Ai = try interpolateMatrix(allocator, At, rows, cols);
            const Bi = try interpolateMatrix(allocator, Bt, rows, cols);
            const Ci = try interpolateMatrix(allocator, Ct, rows, cols);

            // var Z: []const f64 = &.{1};
            // for (1..cols + 1) |i| Z = try mul(allocator, Z, &.{
            //     @floatFromInt(-@as(i32, @intCast(i))),
            //     1,
            // });

            return .{
                .a = Ai,
                .b = Bi,
                .c = Ci,
                // .z = Z,
                .rows = rows,
                .columns = cols,
            };
        }

        fn interpolateMatrix(
            allocator: std.mem.Allocator,
            matrix: []const i32,
            rows: usize,
            cols: usize,
        ) ![]const Field {
            const result = try allocator.alloc(Field, matrix.len);
            for (0..rows) |i| {
                const slice = matrix[i * cols ..][0..cols];
                const interpolated = try Fe.interpolate(allocator, slice);
                @memcpy(result[i * cols ..][0..cols], interpolated);
            }
            return result;
        }

        pub fn format(
            q: Q,
            comptime _: []const u8,
            _: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            try writer.writeAll("A(poly):\n");
            try dumpMatrix(writer, q.rows, q.columns, q.a);
            try writer.writeAll("\nB(poly):\n");
            try dumpMatrix(writer, q.rows, q.columns, q.b);
            try writer.writeAll("\nC(poly):\n");
            try dumpMatrix(writer, q.rows, q.columns, q.c);
            // try writer.print("\nZ:\n{d}\n", .{q.z});
        }

        // fn check(q: Qap, allocator: std.mem.Allocator, r: []const i32) !void {
        //     // As = A . s
        //     var As: []const f64 = &.{};
        //     for (0..q.rows, r) |i, s| {
        //         const slice = q.a[i * q.columns ..][0..q.columns];
        //         As = try R1CS.add(allocator, As, try R1CS.mul(allocator, &.{@floatFromInt(s)}, slice));
        //     }

        //     // Bs = B . s
        //     var Bs: []const f64 = &.{};
        //     for (0..q.rows, r) |i, s| {
        //         const slice = q.b[i * q.columns ..][0..q.columns];
        //         Bs = try R1CS.add(allocator, Bs, try R1CS.mul(allocator, &.{@floatFromInt(s)}, slice));
        //     }

        //     // Cs = C . s
        //     var Cs: []const f64 = &.{};
        //     for (0..q.rows, r) |i, s| {
        //         const slice = q.c[i * q.columns ..][0..q.columns];
        //         Cs = try R1CS.add(allocator, Cs, try R1CS.mul(allocator, &.{@floatFromInt(s)}, slice));
        //     }

        //     // t = As * Bs - Cs
        //     const o = try R1CS.sub(allocator, try R1CS.mul(allocator, As, Bs), Cs);
        //     const Z = q.z;

        //     // if (sum(@rem(t, Z)) != 0) invalid
        //     var n_deg = degree(o).?;
        //     const d_deg = degree(Z).?;

        //     var remainder = try allocator.dupe(f64, o);
        //     while (n_deg >= d_deg) {
        //         const coeff = remainder[n_deg] / Z[d_deg];
        //         for (0..d_deg + 1) |i| {
        //             remainder[n_deg - d_deg + i] -= coeff * Z[i];
        //         }
        //         n_deg = degree(remainder) orelse break;
        //     }

        //     // check if there are any non-zero elements in the remainder
        //     if (degree(remainder) != null) return error.HasRemainder;
        // }

        // fn degree(poly: []const f64) ?usize {
        //     var i = poly.len;
        //     while (i > 0) : (i -= 1) {
        //         if (@abs(poly[i - 1]) > EPSILON) {
        //             return i - 1;
        //         }
        //     }
        //     return null; // there is no degree, all zeros.
        // }

        fn deinit(q: Q, allocator: std.mem.Allocator) void {
            allocator.free(q.a);
            allocator.free(q.b);
            allocator.free(q.c);
            // allocator.free(q.z);
        }
    };
}

fn Finite(Field: type) type {
    return struct {
        fn interpolate(
            allocator: std.mem.Allocator,
            points: []const i32,
        ) ![]const Field {
            const N = points.len;

            const F: []const Field = dd: {
                const F = try allocator.alloc(Field, sum(N));
                @memset(F, Field.zero);
                for (points, 0..) |point, i| {
                    F[sum(i)] = try Field.coerce(@intCast(point));
                }

                for (1..N) |i| {
                    for (0..i) |j| {
                        const slice = F[sum(i)..][0 .. i + 1];
                        const numerator = slice[j].sub(F[sum(i - 1)..][j]);
                        const denominator = try Field.fromInt(@intCast((i + 1) - (i - j)));
                        const result = numerator.mul(denominator.invert());
                        slice[j + 1] = result;
                    }
                }

                const result = try allocator.alloc(Field, N);
                for (0..N) |i| {
                    result[i] = F[sum(i) + i];
                }
                break :dd result;
            };

            var P: []const Field = &.{F[N - 1]};
            for (1..N) |i| {
                const single = &.{
                    try Field.coerce(-@as(i11, @intCast(N - i))),
                    try Field.fromInt(1),
                };
                const multiplied = try mul(allocator, P, single);
                P = try add(allocator, multiplied, &.{F[N - i - 1]});
            }
            return P;
        }

        inline fn sum(k: usize) usize {
            return (k * (k + 1)) / 2;
        }

        fn add(
            allocator: std.mem.Allocator,
            a: []const Field,
            b: []const Field,
        ) ![]const Field {
            var o = try allocator.alloc(Field, @max(a.len, b.len));
            @memset(o, Field.zero);
            for (a, 0..) |x, i| {
                o[i] = o[i].add(x);
            }
            for (b, 0..) |x, i| {
                o[i] = o[i].add(x);
            }
            return o;
        }

        fn mul(
            allocator: std.mem.Allocator,
            a: []const Field,
            b: []const Field,
        ) ![]const Field {
            const o = try allocator.alloc(Field, a.len + b.len - 1);
            @memset(o, Field.zero);
            for (a, 0..) |x, i| {
                for (b, 0..) |y, j| {
                    o[i + j] = o[i + j].add(x.mul(y));
                }
            }
            return o;
        }

        fn dumpPoly(poly: []const Field) void {
            for (0..poly.len) |i| {
                const d = poly.len - i - 1;
                std.debug.print("{}*x^{}", .{ poly[d], d });
                if (i != poly.len - 1) std.debug.print(" + ", .{});
            }
            std.debug.print("\n", .{});
        }
    };
}

fn dumpMatrix(
    stream: anytype,
    rows: usize,
    cols: usize,
    /// can be either []const f64 or []const i32
    matrix: anytype,
) !void {
    for (0..rows) |i| {
        try stream.writeAll("[");
        for (0..cols) |j| {
            try stream.print("{d}", .{matrix[i * cols + j]});
            if (j != cols - 1) try stream.writeAll(", ");
        }
        try stream.writeAll("]");
        if (i != rows - 1) try stream.writeByte('\n');
    }
}

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa.deinit();
    const gpa_allocator = switch (builtin.mode) {
        .Debug => gpa.allocator(),
        else => std.heap.c_allocator,
    };
    var arena = std.heap.ArenaAllocator.init(gpa_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var counter: u32 = 0;
    const x = Variable.makeNew(&counter);
    const y = Variable.makeNew(&counter);
    const tmp1 = Variable.makeNew(&counter);
    const tmp2 = Variable.makeNew(&counter);
    const five = Variable.newConstant(5);

    const flat: Flat = .{
        .inputs = &.{x},
        .instructions = &.{
            // y = x
            .{ .op = .mul, .dest = tmp1, .lhs = x, .rhs = x },
            .{ .op = .mul, .dest = y, .lhs = tmp1, .rhs = x },
            // tmp2 = y + x
            .{ .op = .add, .dest = tmp2, .lhs = y, .rhs = x },
            // out = tmp2 + 5
            .{ .op = .add, .dest = .out, .lhs = tmp2, .rhs = five },
        },
    };

    std.debug.print("{}\n", .{flat});

    const qap = try Qap(fe.F641).fromFlat(flat, allocator);
    defer qap.deinit(allocator);

    std.debug.print("{}\n", .{qap});
}

fn expectEqualFe(Field: type, expected: []const Field.IntRepr, actual: []const Field) !void {
    const allocator = std.testing.allocator;
    const actual_int = try allocator.alloc(Field.IntRepr, expected.len);
    defer allocator.free(actual_int);
    for (actual, actual_int) |a, *i| i.* = Field.toInt(a);
    try std.testing.expectEqualSlices(Field.IntRepr, expected, actual_int);
}

test "basic qap" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var counter: u32 = 0;
    const x = Variable.makeNew(&counter);
    const y = Variable.makeNew(&counter);
    const tmp1 = Variable.makeNew(&counter);
    const tmp2 = Variable.makeNew(&counter);
    const five = Variable.newConstant(5);

    const flat: Flat = .{
        .inputs = &.{x},
        .instructions = &.{
            // y = x
            .{ .op = .mul, .dest = tmp1, .lhs = x, .rhs = x },
            .{ .op = .mul, .dest = y, .lhs = tmp1, .rhs = x },
            // tmp2 = y + x
            .{ .op = .add, .dest = tmp2, .lhs = y, .rhs = x },
            // out = tmp2 + 5
            .{ .op = .add, .dest = .out, .lhs = tmp2, .rhs = five },
        },
    };

    const qap = try Qap(fe.F641).fromFlat(flat, allocator);
    defer qap.deinit(allocator);

    try expectEqualFe(fe.F641, &.{
        636, 116, 636, 535,
        8,   416, 5,   213,
        0,   0,   0,   0,
        635, 330, 637, 321,
        4,   634, 324, 320,
        640, 536, 640, 107,
    }, qap.a);

    try expectEqualFe(fe.F641, &.{
        3,   529, 323, 427,
        639, 112, 318, 214,
        0,   0,   0,   0,
        0,   0,   0,   0,
        0,   0,   0,   0,
        0,   0,   0,   0,
    }, qap.b);

    try expectEqualFe(fe.F641, &.{
        0,   0,   0,   0,
        0,   0,   0,   0,
        640, 536, 640, 107,
        4,   423, 322, 534,
        635, 330, 637, 321,
        4,   634, 324, 320,
    }, qap.c);
}

test "langrage interpolate over finite field" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const Field = Finite(fe.F641);
    const result = try Field.interpolate(arena.allocator(), &.{ 1, 0, 1, 0 });
    try std.testing.expect(result.len == 4);
    try std.testing.expectEqual(8, result[0].toInt());
    try std.testing.expectEqual(416, result[1].toInt());
    try std.testing.expectEqual(5, result[2].toInt());
    try std.testing.expectEqual(213, result[3].toInt());
}
