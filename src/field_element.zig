const std = @import("std");
const crypto = std.crypto;
const debug = std.debug;
const mem = std.mem;
const meta = std.meta;

const NonCanonicalError = crypto.errors.NonCanonicalError;
const NotSquareError = crypto.errors.NotSquareError;

/// Parameters to create a finite field type.
pub const FieldParams = struct {
    fiat: type,
    field_order: comptime_int,
    field_bits: comptime_int,
    encoded_length: comptime_int,
};

/// A field element, internally stored in Montgomery domain.
pub fn Field(comptime params: FieldParams) type {
    const fiat = params.fiat;
    const MontgomeryDomainFieldElement = fiat.MontgomeryDomainFieldElement;
    const NonMontgomeryDomainFieldElement = fiat.NonMontgomeryDomainFieldElement;

    return struct {
        const Fe = @This();

        limbs: MontgomeryDomainFieldElement,

        /// Field size.
        pub const field_order = params.field_order;

        /// Number of bits to represent the set of all elements.
        pub const field_bits = params.field_bits;

        /// Number of bytes required to encode an element.
        pub const encoded_length = params.encoded_length;

        /// Zero.
        pub const zero: Fe = Fe{ .limbs = mem.zeroes(MontgomeryDomainFieldElement) };

        /// One.
        pub const one = one: {
            var fe: Fe = undefined;
            fiat.setOne(&fe.limbs);
            break :one fe;
        };

        /// Reject non-canonical encodings of an element.
        pub fn rejectNonCanonical(s_: [encoded_length]u8, endian: std.builtin.Endian) NonCanonicalError!void {
            var s = if (endian == .little) s_ else orderSwap(s_);
            const field_order_s = comptime fos: {
                var fos: [encoded_length]u8 = undefined;
                mem.writeInt(std.meta.Int(.unsigned, encoded_length * 8), &fos, field_order, .little);
                break :fos fos;
            };
            if (crypto.utils.timingSafeCompare(u8, &s, &field_order_s, .little) != .lt) {
                return error.NonCanonical;
            }
        }

        /// Swap the endianness of an encoded element.
        pub fn orderSwap(s: [encoded_length]u8) [encoded_length]u8 {
            var t = s;
            for (s, 0..) |x, i| t[t.len - 1 - i] = x;
            return t;
        }

        /// Unpack a field element.
        pub fn fromBytes(s_: [encoded_length]u8, endian: std.builtin.Endian) NonCanonicalError!Fe {
            const s = if (endian == .little) s_ else orderSwap(s_);
            try rejectNonCanonical(s, .little);
            var limbs_z: NonMontgomeryDomainFieldElement = undefined;
            fiat.fromBytes(&limbs_z, s);
            var limbs: MontgomeryDomainFieldElement = undefined;
            fiat.toMontgomery(&limbs, limbs_z);
            return Fe{ .limbs = limbs };
        }

        /// Pack a field element.
        pub fn toBytes(fe: Fe, endian: std.builtin.Endian) [encoded_length]u8 {
            var limbs_z: NonMontgomeryDomainFieldElement = undefined;
            fiat.fromMontgomery(&limbs_z, fe.limbs);
            var s: [encoded_length]u8 = undefined;
            fiat.toBytes(&s, limbs_z);
            return if (endian == .little) s else orderSwap(s);
        }

        /// Element as an integer.
        pub const IntRepr = meta.Int(.unsigned, encoded_length * 8);
        pub const SignedRepr = meta.Int(.signed, params.field_bits + 1);

        /// Create a field element from an integer.
        pub fn fromInt(x: IntRepr) NonCanonicalError!Fe {
            var s: [encoded_length]u8 = undefined;
            mem.writeInt(IntRepr, &s, x, .little);
            return fromBytes(s, .little);
        }

        /// Reduces an input into the field, wrapping around negatives as well.
        pub fn coerce(x: SignedRepr) NonCanonicalError!Fe {
            if (x < 0) {
                // TODO: this intCast isn't valid, @abs(x) can hold one more
                // than maxInt(IntRepr)
                return zero.sub(try fromInt(@intCast(@abs(x))));
            } else {
                return try fromInt(@intCast(x));
            }
        }

        /// Return the field element as an integer.
        pub fn toInt(fe: Fe) IntRepr {
            const s = fe.toBytes(.little);
            return mem.readInt(IntRepr, &s, .little);
        }

        /// Return true if the field element is zero.
        pub fn isZero(fe: Fe) bool {
            var z: @TypeOf(fe.limbs[0]) = undefined;
            fiat.nonzero(&z, fe.limbs);
            return z == 0;
        }

        /// Return true if both field elements are equivalent.
        pub fn equivalent(a: Fe, b: Fe) bool {
            return a.sub(b).isZero();
        }

        /// Return true if the element is odd.
        pub fn isOdd(fe: Fe) bool {
            const s = fe.toBytes(.little);
            return @as(u1, @truncate(s[0])) != 0;
        }

        /// Conditonally replace a field element with `a` if `c` is positive.
        pub fn cMov(fe: *Fe, a: Fe, c: u1) void {
            fiat.selectznz(&fe.limbs, c, fe.limbs, a.limbs);
        }

        /// Add field elements.
        pub fn add(a: Fe, b: Fe) Fe {
            var fe: Fe = undefined;
            fiat.add(&fe.limbs, a.limbs, b.limbs);
            return fe;
        }

        /// Subtract field elements.
        pub fn sub(a: Fe, b: Fe) Fe {
            var fe: Fe = undefined;
            fiat.sub(&fe.limbs, a.limbs, b.limbs);
            return fe;
        }

        /// Double a field element.
        pub fn dbl(a: Fe) Fe {
            var fe: Fe = undefined;
            fiat.add(&fe.limbs, a.limbs, a.limbs);
            return fe;
        }

        /// Multiply field elements.
        pub fn mul(a: Fe, b: Fe) Fe {
            var fe: Fe = undefined;
            fiat.mul(&fe.limbs, a.limbs, b.limbs);
            return fe;
        }

        /// Square a field element.
        pub fn sq(a: Fe) Fe {
            var fe: Fe = undefined;
            fiat.square(&fe.limbs, a.limbs);
            return fe;
        }

        /// Square a field element n times.
        fn sqn(a: Fe, comptime n: comptime_int) Fe {
            var i: usize = 0;
            var fe = a;
            while (i < n) : (i += 1) {
                fe = fe.sq();
            }
            return fe;
        }

        /// Compute a^n.
        pub fn pow(a: Fe, comptime T: type, comptime n: T) Fe {
            var fe = one;
            var x: T = n;
            var t = a;
            while (true) {
                if (@as(u1, @truncate(x)) != 0) fe = fe.mul(t);
                x >>= 1;
                if (x == 0) break;
                t = t.sq();
            }
            return fe;
        }

        /// Negate a field element.
        pub fn neg(a: Fe) Fe {
            var fe: Fe = undefined;
            fiat.opp(&fe.limbs, a.limbs);
            return fe;
        }

        /// Return the inverse of a field element, or 0 if a=0.
        // Field inversion from https://eprint.iacr.org/2021/549.pdf
        pub fn invert(a: Fe) Fe {
            const iterations = if (field_bits < 46)
                (49 * field_bits + 80) / 17
            else
                (49 * field_bits + 57) / 17;
            const Limbs = @TypeOf(a.limbs);
            const Word = @TypeOf(a.limbs[0]);
            const XLimbs = [a.limbs.len + 1]Word;

            var d: Word = 1;
            var f = comptime blk: {
                var f: XLimbs = undefined;
                fiat.msat(&f);
                break :blk f;
            };
            var g: XLimbs = undefined;
            fiat.fromMontgomery(g[0..a.limbs.len], a.limbs);
            g[g.len - 1] = 0;

            var r = Fe.one.limbs;
            var v = Fe.zero.limbs;

            var out1: Word = undefined;
            var out2: XLimbs = undefined;
            var out3: XLimbs = undefined;
            var out4: Limbs = undefined;
            var out5: Limbs = undefined;

            var i: usize = 0;
            while (i < iterations - iterations % 2) : (i += 2) {
                fiat.divstep(&out1, &out2, &out3, &out4, &out5, d, f, g, v, r);
                fiat.divstep(&d, &f, &g, &v, &r, out1, out2, out3, out4, out5);
            }
            if (iterations % 2 != 0) {
                fiat.divstep(&out1, &out2, &out3, &out4, &out5, d, f, g, v, r);
                v = out4;
                f = out2;
            }
            var v_opp: Limbs = undefined;
            fiat.opp(&v_opp, v);
            fiat.selectznz(&v, @as(u1, @truncate(f[f.len - 1] >> (@bitSizeOf(Word) - 1))), v, v_opp);

            const precomp = blk: {
                var precomp: Limbs = undefined;
                fiat.divstepPrecomp(&precomp);
                break :blk precomp;
            };
            var fe: Fe = undefined;
            fiat.mul(&fe.limbs, v, precomp);
            return fe;
        }

        /// Returns an random point in the finite field, which is evenly sampled from `0 <= X < order`.
        pub fn sample(random: std.Random) Fe {
            const value = random.uintLessThan(IntRepr, field_order);
            // `uintLessThan` guarantees the value will be representable.
            return fromInt(value) catch unreachable;
        }

        pub fn format(
            fe: Fe,
            comptime _: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            const int = fe.toInt();
            try std.fmt.formatInt(int, 10, .lower, options, writer);
        }
    };
}
