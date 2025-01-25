const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;

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

    fn convert(flat: *const Flat, allocator: std.mem.Allocator) !R1CS {
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
                    setVar(a, inst.lhs, variables);
                    setVar(a, inst.rhs, variables);
                    b[0] = 1;
                },
                .mul => {
                    c[variables.getIndex(inst.dest).?] = 1;
                    setVar(a, inst.lhs, variables);
                    setVar(b, inst.rhs, variables);
                },
            }
        }

        return .{
            .rows = flat.instructions.len,
            .columns = num_variables,
            .a = A,
            .b = B,
            .c = C,
            .r = try flat.computeInput(variables, allocator),
        };
    }

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

    fn computeInput(
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
        switch (variable) {
            .none, .one, .out => |t| try writer.print("{s}", .{@tagName(t)}),
            else => try writer.print("%{d}", .{@intFromEnum(variable)}),
        }
    }
};

fn printMatrix(
    stream: anytype,
    rows: usize,
    cols: usize,
    /// can be either []const f32 or []const i32
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

const R1CS = struct {
    a: []const i32,
    b: []const i32,
    c: []const i32,
    r: []const i32,
    rows: usize,
    columns: usize,

    fn interpolate(
        allocator: std.mem.Allocator,
        points: []const i32,
    ) ![]const f32 {
        var o: []const f32 = &.{};
        for (points, 0..) |point, i| {
            const result = try singleton(
                allocator,
                @intCast(i + 1),
                point,
                @intCast(points.len),
            );
            o = try add(allocator, o, result);
        }
        return o;
    }

    fn singleton(
        allocator: std.mem.Allocator,
        point: i32,
        height: i32,
        total_points: u32,
    ) ![]const f32 {
        var fac: i32 = 1;
        for (1..total_points + 1) |i| {
            if (i != point) fac *= point - @as(i32, @intCast(i));
        }
        var result: []const f32 = &.{@as(f32, @floatFromInt(height)) * (1.0 / @as(f32, @floatFromInt(fac)))};
        var c: u32 = 1;
        for (1..total_points + 1) |i| {
            if (i != point) {
                result = try mul(allocator, result, &.{ @floatFromInt(-@as(i32, @intCast(i))), 1 });
                c += 1;
            }
        }
        return result;
    }

    fn add(
        allocator: std.mem.Allocator,
        a: []const f32,
        b: []const f32,
    ) ![]const f32 {
        var o = try allocator.alloc(f32, @max(a.len, b.len));
        @memset(o, 0.0);
        for (a, 0..) |x, i| {
            o[i] += x;
        }
        for (b, 0..) |x, i| {
            o[i] += x;
        }
        return o;
    }

    fn mul(
        allocator: std.mem.Allocator,
        a: []const f32,
        b: []const f32,
    ) ![]const f32 {
        const o = try allocator.alloc(f32, a.len + b.len - 1);
        @memset(o, 0.0);
        for (a, 0..) |x, i| {
            for (b, 0..) |y, j| {
                o[i + j] += x * y;
            }
        }
        return o;
    }

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

    fn toQAP(r: R1CS, allocator: std.mem.Allocator) !Qap {
        // swapped since we're transposing the matrices
        const cols = r.rows;
        const rows = r.columns;

        const A = try transpose(allocator, r.a, cols, rows);
        const B = try transpose(allocator, r.b, cols, rows);
        const C = try transpose(allocator, r.c, cols, rows);

        const iA = try interpolateMatrix(allocator, A, rows, cols);
        const iB = try interpolateMatrix(allocator, B, rows, cols);
        const iC = try interpolateMatrix(allocator, C, rows, cols);

        return .{
            .a = iA,
            .b = iB,
            .c = iC,
            .rows = rows,
            .columns = cols,
        };
    }

    fn interpolateMatrix(
        allocator: std.mem.Allocator,
        matrix: []const i32,
        rows: usize,
        cols: usize,
    ) ![]const f32 {
        const result = try allocator.alloc(f32, matrix.len);
        for (0..rows) |i| {
            const slice = matrix[i * cols ..][0..cols];
            const interpolated = try interpolate(allocator, slice);
            @memcpy(result[i * cols ..][0..cols], interpolated);
        }
        return result;
    }

    pub fn format(
        r: R1CS,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        try writer.print("r: {d}\n", .{r.r});
        try writer.writeAll("A:\n");
        try printMatrix(writer, r.rows, r.columns, r.a);
        try writer.writeAll("\nB:\n");
        try printMatrix(writer, r.rows, r.columns, r.b);
        try writer.writeAll("\nC:\n");
        try printMatrix(writer, r.rows, r.columns, r.c);
        try writer.writeByte('\n');
    }

    fn deinit(r: R1CS, allocator: std.mem.Allocator) void {
        allocator.free(r.a);
        allocator.free(r.b);
        allocator.free(r.c);
        allocator.free(r.r);
    }
};

const Qap = struct {
    rows: usize,
    columns: usize,
    a: []const f32,
    b: []const f32,
    c: []const f32,

    pub fn format(
        q: Qap,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        try writer.writeAll("A(poly):\n");
        try printMatrix(writer, q.rows, q.columns, q.a);
        try writer.writeAll("\nB(poly):\n");
        try printMatrix(writer, q.rows, q.columns, q.b);
        try writer.writeAll("\nC(poly):\n");
        try printMatrix(writer, q.rows, q.columns, q.c);
        try writer.writeByte('\n');
    }

    fn deinit(q: Qap, allocator: std.mem.Allocator) void {
        allocator.free(q.a);
        allocator.free(q.b);
        allocator.free(q.c);
    }
};

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
            // y = x ^ 3
            .{ .op = .mul, .dest = tmp1, .lhs = x, .rhs = x },
            .{ .op = .mul, .dest = y, .lhs = tmp1, .rhs = x },
            // tmp2 = y + x
            .{ .op = .add, .dest = tmp2, .lhs = y, .rhs = x },
            .{ .op = .add, .dest = .out, .lhs = tmp2, .rhs = five },
        },
    };

    std.debug.print("{}\n", .{flat});

    const r1cs = try flat.convert(allocator);
    defer r1cs.deinit(allocator);

    std.debug.print("{}\n", .{r1cs});

    const qap = try r1cs.toQAP(allocator);
    defer qap.deinit(allocator);

    std.debug.print("{}\n", .{qap});
}

test "basic qap" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var counter: u32 = 0;
    const x = Variable.makeNew(&counter);
    const y = Variable.makeNew(&counter);
    const tmp1 = Variable.makeNew(&counter);

    const flat: Flat = .{
        .inputs = &.{x},
        .instructions = &.{
            // y = x ^ 3
            .{ .op = .mul, .dest = tmp1, .lhs = x, .rhs = x },
            .{ .op = .mul, .dest = y, .lhs = tmp1, .rhs = x },
            // out = y + x
            .{ .op = .add, .dest = .out, .lhs = y, .rhs = x },
        },
    };

    const r1cs = try flat.convert(allocator);
    defer r1cs.deinit(allocator);

    try std.testing.expectEqualSlices(i32, &.{
        0, 1, 0, 0, 0,
        0, 0, 0, 1, 0,
        0, 1, 0, 0, 1,
    }, r1cs.a);

    try std.testing.expectEqualSlices(i32, &.{
        0, 1, 0, 0, 0,
        0, 1, 0, 0, 0,
        1, 0, 0, 0, 0,
    }, r1cs.b);

    try std.testing.expectEqualSlices(i32, &.{
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1,
        0, 0, 1, 0, 0,
    }, r1cs.c);

    const qap = try r1cs.toQAP(allocator);
    defer qap.deinit(allocator);

    try std.testing.expectEqualSlices(f32, &.{
        0.0,  0.0,  0.0,
        4.0,  -4.0, 1.0,
        0.0,  0.0,  0.0,
        -3.0, 4.0,  -1.0,
        1.0,  -1.5, 0.5,
    }, qap.a);

    try std.testing.expectEqualSlices(f32, &.{
        1.0, -1.5, 0.5,
        0.0, 1.5,  -0.5,
        0.0, 0.0,  0.0,
        0.0, 0.0,  0.0,
        0.0, 0.0,  0.0,
    }, qap.b);

    try std.testing.expectEqualSlices(f32, &.{
        0.0,  0.0,  0.0,
        0.0,  0.0,  0.0,
        1.0,  -1.5, 0.5,
        3.0,  -2.5, 0.5,
        -3.0, 4.0,  -1.0,
    }, qap.c);
}

test "lagrange interpolate" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const result = try R1CS.interpolate(arena.allocator(), &.{ 1, 0, 1 });
    try std.testing.expectEqualSlices(f32, &.{ 4, -4, 1 }, result);
}
