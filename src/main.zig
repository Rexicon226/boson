const std = @import("std");
const builtin = @import("builtin");

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
                    a[variables.getIndex(inst.lhs).?] += 1;
                    a[variables.getIndex(inst.rhs).?] += 1;
                    b[0] = 1;
                },
                .mul => {
                    c[variables.getIndex(inst.dest).?] = 1;
                    a[variables.getIndex(inst.lhs).?] += 1;
                    b[variables.getIndex(inst.rhs).?] += 1;
                },
            }
        }

        return .{
            .rows = flat.instructions.len,
            .columns = num_variables,
            .a = A,
            .b = B,
            .c = C,
            .r = try flat.computeInput(&variables, allocator),
        };
    }

    fn computeInput(
        flat: *const Flat,
        variables: *const std.AutoArrayHashMap(Variable, void),
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
                .set => r[variables.getIndex(inst.dest).?] = r[variables.getIndex(inst.lhs).?],
                .add,
                .mul,
                => {
                    const rhs = r[variables.getIndex(inst.rhs).?];
                    const lhs = r[variables.getIndex(inst.lhs).?];
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

const Variable = enum(u32) {
    none,
    one,
    out,
    _,

    fn makeNew(counter: *u32) Variable {
        defer counter.* += 1;
        return @enumFromInt(counter.* + @typeInfo(Variable).Enum.fields.len);
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

const R1CS = struct {
    /// The number of rows in each matrix.
    rows: usize,
    /// The number of columns in each matrix.
    columns: usize,

    a: []const i32,
    b: []const i32,
    c: []const i32,
    r: []const i32,

    pub fn format(
        r: R1CS,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        try writer.writeAll("A:\n");
        try printMatrix(writer, r.rows, r.columns, r.a);
        try writer.writeAll("\nB:\n");
        try printMatrix(writer, r.rows, r.columns, r.b);
        try writer.writeAll("\nC:\n");
        try printMatrix(writer, r.rows, r.columns, r.c);
    }

    fn printMatrix(
        stream: anytype,
        row: usize,
        col: usize,
        matrix: []const i32,
    ) !void {
        for (0..row) |i| {
            try stream.writeAll("[");
            for (0..col) |j| {
                try stream.print("{d}", .{matrix[i * col + j]});
                if (j != col - 1) try stream.writeAll(", ");
            }
            try stream.writeAll("]");
            if (i != row - 1) try stream.writeByte('\n');
        }
    }

    fn deinit(r: R1CS, allocator: std.mem.Allocator) void {
        allocator.free(r.a);
        allocator.free(r.b);
        allocator.free(r.c);
        allocator.free(r.r);
    }
};

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa.deinit();
    const allocator = switch (builtin.mode) {
        .Debug => gpa.allocator(),
        else => std.heap.c_allocator,
    };

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

    std.debug.print("{}\n", .{flat});

    const r1cs = try flat.convert(allocator);
    defer r1cs.deinit(allocator);

    std.debug.print("{}\n", .{r1cs});

    std.debug.print("array: {d}\n", .{r1cs.a});
}

test "basic vars only" {
    const allocator = std.testing.allocator;
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

    try std.testing.expectEqualDeep(r1cs.a, &[_]i32{
        0, 1, 0, 0, 0,
        0, 0, 0, 1, 0,
        0, 1, 0, 0, 1,
    });
    try std.testing.expectEqualDeep(r1cs.b, &[_]i32{
        0, 1, 0, 0, 0,
        0, 1, 0, 0, 0,
        1, 0, 0, 0, 0,
    });
    try std.testing.expectEqualDeep(r1cs.c, &[_]i32{
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1,
        0, 0, 1, 0, 0,
    });
}
