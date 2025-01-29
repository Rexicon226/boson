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

    var lxp, var rxp, var oxp = try qap.generateSolutionPolynomials(allocator, r);
    defer {
        lxp.deinit(allocator);
        rxp.deinit(allocator);
        oxp.deinit(allocator);
    }

    try lxp.mul(allocator, rxp);
    try lxp.sub(allocator, oxp);

    var Z = try qap.generateZeroPolynomial(allocator);
    defer Z.deinit(allocator);

    std.debug.print("before T: {}\n", .{lxp});
    std.debug.print("Z: {}\n", .{Z});

    const T = lxp;

    var n_deg = T.degree().?;
    const d_deg = Z.degree().?;

    const t = T.coeffs.items;
    const z = Z.coeffs.items;

    while (n_deg >= d_deg) {
        const coeff = t[n_deg].mul(z[d_deg].invert());
        for (0..d_deg + 1) |i| {
            t[n_deg - d_deg + i] = t[n_deg - d_deg + i].sub(coeff.mul(z[i]));
        }
        n_deg = T.degree() orelse break;
    }

    std.debug.print("t: {}\n", .{T});

    if (T.degree() != null) @panic("has remainder");
}
