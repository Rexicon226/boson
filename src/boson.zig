const std = @import("std");
const Flat = @import("Flat.zig");
const fe = @import("fe.zig");
const Variable = Flat.Variable;
const assert = std.debug.assert;

pub fn Qap(Field: type) type {
    return struct {
        rows: usize,
        columns: usize,

        a: M,
        b: M,
        c: M,

        const Q = @This();
        const Fe = Finite(Field);
        const M = Matrix(Field);
        const Poly = Polynomial(Field);

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

        pub fn fromFlat(flat: Flat, allocator: std.mem.Allocator) !Q {
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
            defer allocator.free(A);
            const B = try allocator.alloc(i32, total_size);
            defer allocator.free(B);
            const C = try allocator.alloc(i32, total_size);
            defer allocator.free(C);

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
            defer allocator.free(At);
            const Bt = try transpose(allocator, B, cols, rows);
            defer allocator.free(Bt);
            const Ct = try transpose(allocator, C, cols, rows);
            defer allocator.free(Ct);

            const Ai = try interpolateMatrix(allocator, At, rows, cols);
            const Bi = try interpolateMatrix(allocator, Bt, rows, cols);
            const Ci = try interpolateMatrix(allocator, Ct, rows, cols);

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
        ) !M {
            const result = try allocator.alloc(Field, matrix.len);
            for (0..rows) |i| {
                const slice = matrix[i * cols ..][0..cols];
                const interpolated = try Fe.interpolate(allocator, slice);
                defer allocator.free(interpolated);
                @memcpy(result[i * cols ..][0..cols], interpolated);
            }
            return M.init(result, cols);
        }

        pub fn solutionPolynomials(
            qap: Q,
            allocator: std.mem.Allocator,
            r: []const i32,
        ) !struct { Poly, Poly, Poly } {
            var S = try Matrix(fe.F641).initCoerce(allocator, r, r.len);
            defer S.deinit(allocator);

            const lx = try S.dot(allocator, qap.a);
            defer lx.deinit(allocator);

            const rx = try S.dot(allocator, qap.b);
            defer rx.deinit(allocator);

            const ox = try S.dot(allocator, qap.c);
            defer ox.deinit(allocator);

            const lxp = try lx.toPolynomial(allocator);
            const rxp = try rx.toPolynomial(allocator);
            const oxp = try ox.toPolynomial(allocator);

            return .{
                lxp,
                rxp,
                oxp,
            };
        }

        pub fn zeroPolynomial(qap: Q, allocator: std.mem.Allocator) !Poly {
            var Z = try Poly.fromInts(allocator, &.{1});
            for (1..qap.columns + 1) |i| {
                var singleton = try Poly.fromInts(allocator, &.{
                    -@as(i32, @intCast(i)),
                    1,
                });
                defer singleton.deinit(allocator);
                try Z.mul(allocator, singleton);
            }
            return Z;
        }

        pub fn format(
            q: Q,
            comptime _: []const u8,
            _: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            try writer.writeAll("A(poly):\n");
            try writer.print("{}", .{q.a});
            try writer.writeAll("\nB(poly):\n");
            try writer.print("{}", .{q.b});
            try writer.writeAll("\nC(poly):\n");
            try writer.print("{}", .{q.c});
            // try writer.print("\nZ:\n{d}\n", .{q.z});
        }

        fn dumpMatrix(q: Q, stream: anytype, matrix: []const Field) !void {
            for (0..q.rows) |i| {
                try stream.writeAll("[");
                for (0..q.columns) |j| {
                    try stream.print("{d}", .{matrix[i * q.columns + j]});
                    if (j != q.columns - 1) try stream.writeAll(", ");
                }
                try stream.writeAll("]");
                if (i != q.rows - 1) try stream.writeByte('\n');
            }
        }

        pub fn deinit(q: Q, allocator: std.mem.Allocator) void {
            q.a.deinit(allocator);
            q.b.deinit(allocator);
            q.c.deinit(allocator);
            // allocator.free(q.z);
        }
    };
}

pub fn Polynomial(Field: type) type {
    return struct {
        const Poly = @This();
        coeffs: std.ArrayListUnmanaged(Field) = .{},

        pub fn fromCoeffs(allocator: std.mem.Allocator, coeffs: []const Field) !Poly {
            var list = try std.ArrayListUnmanaged(Field).initCapacity(allocator, coeffs.len);
            list.appendSliceAssumeCapacity(coeffs);
            return .{ .coeffs = list };
        }

        pub fn fromInts(allocator: std.mem.Allocator, coeffs: []const i32) !Poly {
            var list = try std.ArrayList(Field).initCapacity(allocator, coeffs.len);
            for (coeffs) |x| {
                list.appendAssumeCapacity(try Field.coerce(@intCast(x)));
            }
            return .{ .coeffs = list.moveToUnmanaged() };
        }

        pub fn add(poly: *Poly, allocator: std.mem.Allocator, other: Poly) !void {
            const o = other.coeffs.items;
            const p = poly.coeffs.items;

            const result = try allocator.alloc(Field, @max(o.len, p.len));
            defer allocator.free(result);
            @memset(result, Field.zero);

            for (p, 0..) |x, i| {
                result[i] = x;
            }

            for (o, 0..) |x, i| {
                result[i] = result[i].add(x);
            }

            poly.deinit(allocator);
            poly.* = try fromCoeffs(allocator, result);
        }

        pub fn sub(poly: *Poly, allocator: std.mem.Allocator, other: Poly) !void {
            const o = other.coeffs.items;
            const p = poly.coeffs.items;

            const result = try allocator.alloc(Field, @max(o.len, p.len));
            defer allocator.free(result);
            @memset(result, Field.zero);

            for (p, 0..) |x, i| {
                result[i] = x;
            }

            for (o, 0..) |x, i| {
                result[i] = result[i].sub(x);
            }

            poly.deinit(allocator);
            poly.* = try fromCoeffs(allocator, result);
        }

        pub fn mul(poly: *Poly, allocator: std.mem.Allocator, other: Poly) !void {
            const o = other.coeffs.items;
            const p = poly.coeffs.items;

            const result = try allocator.alloc(Field, o.len + p.len - 1);
            defer allocator.free(result);
            @memset(result, Field.zero);

            for (p, 0..) |x, i| {
                for (o, 0..) |y, j| {
                    result[i + j] = result[i + j].add(x.mul(y));
                }
            }

            poly.deinit(allocator);
            poly.* = try fromCoeffs(allocator, result);
        }

        pub fn quorem(poly: Poly, allocator: std.mem.Allocator, other: Poly) !struct { Poly, Poly } {
            var remainder = try poly.clone(allocator);

            var n_deg = remainder.degree().?;
            const d_deg = other.degree().?;

            const t = remainder.coeffs.items;
            const z = other.coeffs.items;

            var coeffs: std.ArrayListUnmanaged(Field) = .{};
            while (n_deg >= d_deg) {
                const coeff = t[n_deg].mul(z[d_deg].invert());
                try coeffs.append(allocator, coeff);
                for (0..d_deg + 1) |i| {
                    t[n_deg - d_deg + i] = t[n_deg - d_deg + i].sub(coeff.mul(z[i]));
                }
                n_deg = remainder.degree() orelse break;
            }
            std.mem.reverse(Field, coeffs.items);

            const H: Poly = .{ .coeffs = coeffs };
            return .{ H, remainder };
        }

        pub fn degree(poly: Poly) ?usize {
            const p = poly.coeffs.items;
            var i = p.len;
            while (i > 0) : (i -= 1) {
                if (!p[i - 1].isZero()) {
                    return i - 1;
                }
            }
            return null;
        }

        pub fn clone(poly: Poly, allocator: std.mem.Allocator) !Poly {
            return .{ .coeffs = try poly.coeffs.clone(allocator) };
        }

        pub fn deinit(poly: *Poly, allocator: std.mem.Allocator) void {
            poly.coeffs.deinit(allocator);
        }

        pub fn format(
            poly: Poly,
            comptime _: []const u8,
            _: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            const length = poly.coeffs.items.len;
            for (0..length) |i| {
                const d = length - i - 1;
                try writer.print("{}*x^{}", .{ poly.coeffs.items[d], d });
                if (i != length - 1) try writer.writeAll(" + ");
            }
        }
    };
}

pub fn Matrix(Field: type) type {
    return struct {
        items: []const Field,
        rows: usize,
        columns: usize,

        const M = @This();

        pub fn init(items: []const Field, columns: usize) !M {
            return .{
                .items = items,
                .rows = items.len / columns,
                .columns = columns,
            };
        }

        pub fn initOwned(allocator: std.mem.Allocator, items: []const Field, columns: usize) !M {
            return .{
                .items = try allocator.dupe(Field, items),
                .rows = items.len / columns,
                .columns = columns,
            };
        }

        pub fn initCoerce(allocator: std.mem.Allocator, items: anytype, columns: usize) !M {
            const fields = try allocator.alloc(Field, items.len);
            for (items, fields) |item, *field| field.* = try Field.coerce(@intCast(item));
            return .{
                .items = fields,
                .rows = fields.len / columns,
                .columns = columns,
            };
        }

        pub fn dot(m: M, allocator: std.mem.Allocator, b: M) !M {
            if (m.columns != b.rows) return error.InvalidSize;

            const result = try allocator.alloc(Field, m.rows * b.columns);
            @memset(result, Field.zero);

            for (0..m.rows) |i| {
                for (0..b.columns) |j| {
                    for (0..m.columns) |k| {
                        const multiply = m.items[i * m.columns + k].mul(b.items[k * b.columns + j]);
                        result[i * b.columns + j] = result[i * b.columns + j].add(multiply);
                    }
                }
            }

            return .{
                .rows = m.rows,
                .columns = b.columns,
                .items = result,
            };
        }

        pub fn toPolynomial(m: M, allocator: std.mem.Allocator) !Polynomial(Field) {
            if (m.rows != 1) return error.NotFlatMatrix;
            return Polynomial(Field).fromCoeffs(allocator, m.items);
        }

        pub fn deinit(m: M, allocator: std.mem.Allocator) void {
            allocator.free(m.items);
        }

        pub fn format(
            m: M,
            comptime _: []const u8,
            _: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            var max_width: usize = 0;
            for (m.items) |item| {
                const len = std.fmt.count("{d}", .{item});
                max_width = @max(len, max_width);
            }
            for (0..m.rows) |i| {
                try writer.print(
                    "{d: >[1]}",
                    .{ m.items[i * m.columns ..][0..m.columns], max_width },
                );
                if (i != m.rows - 1) try writer.writeAll("\n");
            }
        }
    };
}

fn Finite(Field: type) type {
    return struct {
        const Poly = Polynomial(Field);

        fn interpolate(
            allocator: std.mem.Allocator,
            points: []const i32,
        ) ![]const Field {
            const N = points.len;

            const F: []const Field = dd: {
                const F = try allocator.alloc(Field, sum(N));
                defer allocator.free(F);
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
            defer allocator.free(F);

            var P = try Poly.fromCoeffs(allocator, &.{F[N - 1]});
            for (1..N) |i| {
                var single = try Poly.fromCoeffs(allocator, &.{
                    try Field.coerce(-@as(i11, @intCast(N - i))),
                    try Field.fromInt(1),
                });
                defer single.deinit(allocator);
                try P.mul(allocator, single);
                var offset = try Poly.fromCoeffs(allocator, &.{F[N - i - 1]});
                defer offset.deinit(allocator);
                try P.add(allocator, offset);
            }
            return P.coeffs.toOwnedSlice(allocator);
        }

        inline fn sum(k: usize) usize {
            return (k * (k + 1)) / 2;
        }
    };
}

fn expectEqualFe(Field: type, expected: []const Field.IntRepr, actual: []const Field) !void {
    const allocator = std.testing.allocator;
    const actual_int = try allocator.alloc(Field.IntRepr, expected.len);
    defer allocator.free(actual_int);
    for (actual, actual_int) |a, *i| i.* = Field.toInt(a);
    try std.testing.expectEqualSlices(Field.IntRepr, expected, actual_int);
}

test "basic qap" {
    const allocator = std.testing.allocator;

    var counter: u32 = 0;
    const x = Variable.new(&counter);
    const y = Variable.new(&counter);
    const tmp1 = Variable.new(&counter);
    const tmp2 = Variable.new(&counter);
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
    }, qap.a.items);

    try expectEqualFe(fe.F641, &.{
        3,   529, 323, 427,
        639, 112, 318, 214,
        0,   0,   0,   0,
        0,   0,   0,   0,
        0,   0,   0,   0,
        0,   0,   0,   0,
    }, qap.b.items);

    try expectEqualFe(fe.F641, &.{
        0,   0,   0,   0,
        0,   0,   0,   0,
        640, 536, 640, 107,
        4,   423, 322, 534,
        635, 330, 637, 321,
        4,   634, 324, 320,
    }, qap.c.items);
}

test "langrage interpolate over finite field" {
    const allocator = std.testing.allocator;

    const Field = Finite(fe.F641);
    const result = try Field.interpolate(allocator, &.{ 1, 0, 1, 0 });
    defer allocator.free(result);
    try std.testing.expect(result.len == 4);
    try std.testing.expectEqual(8, result[0].toInt());
    try std.testing.expectEqual(416, result[1].toInt());
    try std.testing.expectEqual(5, result[2].toInt());
    try std.testing.expectEqual(213, result[3].toInt());
}

test "polynomial" {
    const allocator = std.testing.allocator;
    const Field = fe.F641;
    const Poly = Polynomial(Field);

    {
        var x = try Poly.fromCoeffs(
            allocator,
            &.{
                try Field.fromInt(0),
                try Field.fromInt(1),
                try Field.fromInt(2),
                try Field.fromInt(3),
            },
        );
        defer x.deinit(allocator);
    }
}

test "polynomial add" {
    const allocator = std.testing.allocator;
    const Field = fe.F641;
    const Poly = Polynomial(Field);

    {
        var x = try Poly.fromCoeffs(
            allocator,
            &.{
                try Field.fromInt(1),
                try Field.fromInt(2),
            },
        );
        defer x.deinit(allocator);
        var y = try Poly.fromCoeffs(
            allocator,
            &.{
                try Field.fromInt(2),
                try Field.fromInt(3),
                try Field.fromInt(4),
            },
        );
        defer y.deinit(allocator);

        try x.add(allocator, y);
        try expectEqualFe(Field, &.{ 3, 5, 4 }, x.coeffs.items);
    }

    {
        var x = try Poly.fromCoeffs(
            allocator,
            &.{
                try Field.fromInt(1),
                try Field.fromInt(2),
                try Field.fromInt(4),
            },
        );
        defer x.deinit(allocator);
        var y = try Poly.fromCoeffs(
            allocator,
            &.{
                try Field.fromInt(2),
                try Field.fromInt(3),
            },
        );
        defer y.deinit(allocator);

        try x.add(allocator, y);
        try expectEqualFe(Field, &.{ 3, 5, 4 }, x.coeffs.items);
    }
}

test "polynomial mul" {
    const allocator = std.testing.allocator;
    const Field = fe.F641;
    const Poly = Polynomial(Field);

    {
        var x = try Poly.fromCoeffs(
            allocator,
            &.{
                try Field.fromInt(1),
                try Field.fromInt(2),
            },
        );
        defer x.deinit(allocator);
        var y = try Poly.fromCoeffs(
            allocator,
            &.{
                try Field.fromInt(2),
                try Field.fromInt(3),
                try Field.fromInt(4),
            },
        );
        defer y.deinit(allocator);

        try x.mul(allocator, y);

        try expectEqualFe(Field, &.{ 2, 7, 10, 8 }, x.coeffs.items);
    }

    {
        var x = try Poly.fromCoeffs(
            allocator,
            &.{
                try Field.fromInt(1),
                try Field.fromInt(2),
                try Field.fromInt(4),
            },
        );
        defer x.deinit(allocator);
        var y = try Poly.fromCoeffs(
            allocator,
            &.{
                try Field.fromInt(2),
                try Field.fromInt(3),
            },
        );
        defer y.deinit(allocator);

        try x.mul(allocator, y);

        try expectEqualFe(Field, &.{ 2, 7, 14, 12 }, x.coeffs.items);
    }
}

test "polynomial degree" {
    const allocator = std.testing.allocator;
    const Field = fe.F641;
    const Poly = Polynomial(Field);

    var x = try Poly.fromCoeffs(
        allocator,
        &.{
            try Field.fromInt(1),
            try Field.fromInt(2),
            try Field.fromInt(4),
            Field.zero,
            try Field.fromInt(5),
            Field.zero,
            Field.zero,
        },
    );
    defer x.deinit(allocator);

    try std.testing.expectEqual(4, x.degree());
}
