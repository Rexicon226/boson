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
        const A = try allocator.alloc([]const i32, flat.instructions.len);
        const B = try allocator.alloc([]const i32, flat.instructions.len);
        const C = try allocator.alloc([]const i32, flat.instructions.len);

        var used: std.AutoHashMapUnmanaged(Variable, void) = .{};
        defer used.deinit(allocator);
        for (flat.inputs) |input| {
            try used.putNoClobber(allocator, input, {});
        }

        var variables = vars: {
            var list = std.AutoArrayHashMap(Variable, void).init(allocator);
            try list.putNoClobber(.one, {});
            for (flat.inputs) |input| {
                try list.putNoClobber(input, {});
            }
            try list.put(.out, {});
            for (flat.instructions) |inst| {
                if (std.mem.indexOfScalar(Variable, flat.inputs, inst.dest) != null) continue;
                if (inst.dest == .out) continue;
                try list.put(inst.dest, {});
            }
            break :vars list;
        };
        defer variables.deinit();

        for (flat.instructions, 0..) |inst, i| {
            var a = try allocator.alloc(i32, variables.count());
            var b = try allocator.alloc(i32, variables.count());
            var c = try allocator.alloc(i32, variables.count());
            @memset(a, 0);
            @memset(b, 0);
            @memset(c, 0);

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

            A[i] = a;
            B[i] = b;
            C[i] = c;
        }

        return .{
            .A = A,
            .B = B,
            .C = C,
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
    const Matrix = []const []const i32;

    A: Matrix,
    B: Matrix,
    C: Matrix,
    r: []const i32,

    pub fn format(
        r: R1CS,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        inline for (@typeInfo(R1CS).Struct.fields[0..3]) |field| {
            try writer.print("{s}:\n", .{field.name});
            for (@field(r, field.name)) |row| {
                try writer.writeAll("[");
                for (row, 0..) |item, i| {
                    try writer.print("{d}", .{item});
                    if (i != row.len - 1) try writer.writeAll(", ");
                }
                try writer.writeAll("]\n");
            }
        }
    }

    fn deinit(r: R1CS, allocator: std.mem.Allocator) void {
        inline for (@typeInfo(R1CS).Struct.fields) |field| {
            const value = @field(r, field.name);
            const T = @TypeOf(value);
            if (T != []const i32) {
                for (value) |row| {
                    allocator.free(row);
                }
            }
            allocator.free(value);
        }
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

    var flat: Flat = .{
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

    try std.testing.expectEqualDeep(r1cs.A, &[_][]const i32{
        &.{ 0, 1, 0, 0, 0 },
        &.{ 0, 0, 0, 1, 0 },
        &.{ 0, 1, 0, 0, 1 },
    });
    try std.testing.expectEqualDeep(r1cs.B, &[_][]const i32{
        &.{ 0, 1, 0, 0, 0 },
        &.{ 0, 1, 0, 0, 0 },
        &.{ 1, 0, 0, 0, 0 },
    });
    try std.testing.expectEqualDeep(r1cs.C, &[_][]const i32{
        &.{ 0, 0, 0, 1, 0 },
        &.{ 0, 0, 0, 0, 1 },
        &.{ 0, 0, 1, 0, 0 },
    });
}
