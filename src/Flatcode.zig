inputs: []const Variable,
instructions: []const Instruction,

pub const Variable = enum(u64) {
    none,
    one,
    out,
    _,

    pub fn new(counter: *u32) Variable {
        defer counter.* += 1;
        return @enumFromInt(counter.* + @typeInfo(Variable).@"enum".fields.len);
    }

    pub fn newConstant(constant: i32) Variable {
        const unsigned: u32 = @bitCast(constant);
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

pub fn setVar(
    array: []i32,
    variable: Variable,
    vars: std.AutoArrayHashMap(Variable, void),
) void {
    if (variable.isConstant()) {
        array[0] += variable.getConstant();
    } else {
        array[vars.getIndex(variable).?] += 1;
    }
}

pub fn getVar(
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

pub fn solve(
    flat: *const Flatcode,
    allocator: std.mem.Allocator,
) ![]const i32 {
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
    flat: Flatcode,
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

const Flatcode = @This();
const std = @import("std");

const assert = std.debug.assert;
