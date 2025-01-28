const std = @import("std");
const builtin = @import("builtin");
const fe = @import("fe.zig");
const Flat = @import("Flat.zig");
const boson = @import("boson.zig");

const Variable = Flat.Variable;
const Qap = boson.Qap;
const Polynomial = boson.Polynomial;
const Matrix = boson.Matrix;
const assert = std.debug.assert;

const EPSILON = 1e-9;

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

    const r = try flat.solve(allocator);
    defer allocator.free(r);

    const qap = try Qap(fe.F641).fromFlat(flat, allocator);
    defer qap.deinit(allocator);

    var S = try Matrix(fe.F641).initCoerce(allocator, r, 1);
    defer S.deinit(allocator);

    std.debug.print("{}\n\n", .{qap});
}
