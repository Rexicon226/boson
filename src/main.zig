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

    var L, var R, var O = try qap.solutionPolynomials(allocator, r);
    defer {
        L.deinit(allocator);
        R.deinit(allocator);
        O.deinit(allocator);
    }

    std.debug.print("L(x) = {}\n", .{L});
    std.debug.print("R(x) = {}\n", .{R});
    std.debug.print("O(x) = {}\n", .{O});

    var T = T: {
        var T = try L.clone(allocator);
        try T.mul(allocator, R);
        try T.sub(allocator, O);
        break :T T;
    };
    defer T.deinit(allocator);

    var Z = try qap.zeroPolynomial(allocator);
    defer Z.deinit(allocator);

    var H, var rem = try T.quorem(allocator, Z);
    defer {
        H.deinit(allocator);
        rem.deinit(allocator);
    }

    var Qr = Q: {
        var Q = try H.clone(allocator);
        try Q.mul(allocator, Z);
        try Q.add(allocator, O);
        break :Q Q;
    };
    defer Qr.deinit(allocator);

    var Ql = Q: {
        var Q = try L.clone(allocator);
        try Q.mul(allocator, R);
        break :Q Q;
    };
    defer Ql.deinit(allocator);

    std.debug.print("Qr(X) = {}\n", .{Qr});
    std.debug.print("Ql(X) = {}\n", .{Ql});
}
