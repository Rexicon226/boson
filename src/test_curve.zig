// Autogenerated: 'src/ExtractionOCaml/fiat_crypto' word-by-word-montgomery --lang Zig --internal-static --public-function-case camelCase --private-function-case camelCase --public-type-case UpperCamelCase --private-type-case UpperCamelCase --no-prefix-fiat -o vesta.zig '' 64 641
// curve description:
// machine_wordsize = 64 (from "64")
// requested operations: (all)
// m = 0x281 (from "641")
//
// NOTE: In addition to the bounds specified above each function, all
//   functions synthesized for this Montgomery arithmetic require the
//   input to be strictly less than the prime modulus (m), and also
//   require the input to be in the unique saturated representation.
//   All functions also ensure that these two properties are true of
//   return values.
//
// Computed values:
//   eval z = z[0]
//   bytes_eval z = z[0] + (z[1] << 8)
//   twos_complement_eval z = let x1 := z[0] in
//                            if x1 & (2^64-1) < 2^63 then x1 & (2^64-1) else (x1 & (2^64-1)) - 2^64

const std = @import("std");
const mode = @import("builtin").mode; // Checked arithmetic is disabled in non-debug modes to avoid side channels

inline fn cast(comptime DestType: type, target: anytype) DestType {
    @setEvalBranchQuota(10000);
    if (@typeInfo(@TypeOf(target)) == .Int) {
        const dest = @typeInfo(DestType).Int;
        const source = @typeInfo(@TypeOf(target)).Int;
        if (dest.bits < source.bits) {
            const T = std.meta.Int(source.signedness, dest.bits);
            return @bitCast(@as(T, @truncate(target)));
        }
    }
    return target;
}

// The type MontgomeryDomainFieldElement is a field element in the Montgomery domain.
// Bounds: [[0x0 ~> 0xffffffffffffffff]]
pub const MontgomeryDomainFieldElement = [1]u64;

// The type NonMontgomeryDomainFieldElement is a field element NOT in the Montgomery domain.
// Bounds: [[0x0 ~> 0xffffffffffffffff]]
pub const NonMontgomeryDomainFieldElement = [1]u64;

/// The function addcarryxU64 is an addition with carry.
///
/// Postconditions:
///   out1 = (arg1 + arg2 + arg3) mod 2^64
///   out2 = ⌊(arg1 + arg2 + arg3) / 2^64⌋
///
/// Input Bounds:
///   arg1: [0x0 ~> 0x1]
///   arg2: [0x0 ~> 0xffffffffffffffff]
///   arg3: [0x0 ~> 0xffffffffffffffff]
/// Output Bounds:
///   out1: [0x0 ~> 0xffffffffffffffff]
///   out2: [0x0 ~> 0x1]
inline fn addcarryxU64(out1: *u64, out2: *u1, arg1: u1, arg2: u64, arg3: u64) void {
    @setRuntimeSafety(mode == .Debug);

    const x1 = ((cast(u128, arg1) + cast(u128, arg2)) + cast(u128, arg3));
    const x2 = cast(u64, (x1 & cast(u128, 0xffffffffffffffff)));
    const x3 = cast(u1, (x1 >> 64));
    out1.* = x2;
    out2.* = x3;
}

/// The function subborrowxU64 is a subtraction with borrow.
///
/// Postconditions:
///   out1 = (-arg1 + arg2 + -arg3) mod 2^64
///   out2 = -⌊(-arg1 + arg2 + -arg3) / 2^64⌋
///
/// Input Bounds:
///   arg1: [0x0 ~> 0x1]
///   arg2: [0x0 ~> 0xffffffffffffffff]
///   arg3: [0x0 ~> 0xffffffffffffffff]
/// Output Bounds:
///   out1: [0x0 ~> 0xffffffffffffffff]
///   out2: [0x0 ~> 0x1]
inline fn subborrowxU64(out1: *u64, out2: *u1, arg1: u1, arg2: u64, arg3: u64) void {
    @setRuntimeSafety(mode == .Debug);

    const x1 = ((cast(i128, arg2) - cast(i128, arg1)) - cast(i128, arg3));
    const x2 = cast(i1, (x1 >> 64));
    const x3 = cast(u64, (x1 & cast(i128, 0xffffffffffffffff)));
    out1.* = x3;
    out2.* = cast(u1, (cast(i2, 0x0) - cast(i2, x2)));
}

/// The function mulxU64 is a multiplication, returning the full double-width result.
///
/// Postconditions:
///   out1 = (arg1 * arg2) mod 2^64
///   out2 = ⌊arg1 * arg2 / 2^64⌋
///
/// Input Bounds:
///   arg1: [0x0 ~> 0xffffffffffffffff]
///   arg2: [0x0 ~> 0xffffffffffffffff]
/// Output Bounds:
///   out1: [0x0 ~> 0xffffffffffffffff]
///   out2: [0x0 ~> 0xffffffffffffffff]
inline fn mulxU64(out1: *u64, out2: *u64, arg1: u64, arg2: u64) void {
    @setRuntimeSafety(mode == .Debug);

    const x1 = (cast(u128, arg1) * cast(u128, arg2));
    const x2 = cast(u64, (x1 & cast(u128, 0xffffffffffffffff)));
    const x3 = cast(u64, (x1 >> 64));
    out1.* = x2;
    out2.* = x3;
}

/// The function cmovznzU64 is a single-word conditional move.
///
/// Postconditions:
///   out1 = (if arg1 = 0 then arg2 else arg3)
///
/// Input Bounds:
///   arg1: [0x0 ~> 0x1]
///   arg2: [0x0 ~> 0xffffffffffffffff]
///   arg3: [0x0 ~> 0xffffffffffffffff]
/// Output Bounds:
///   out1: [0x0 ~> 0xffffffffffffffff]
inline fn cmovznzU64(out1: *u64, arg1: u1, arg2: u64, arg3: u64) void {
    @setRuntimeSafety(mode == .Debug);

    const x1 = (~(~arg1));
    const x2 = cast(u64, (cast(i128, cast(i1, (cast(i2, 0x0) - cast(i2, x1)))) & cast(i128, 0xffffffffffffffff)));
    const x3 = ((x2 & arg3) | ((~x2) & arg2));
    out1.* = x3;
}

/// The function mul multiplies two field elements in the Montgomery domain.
///
/// Preconditions:
///   0 ≤ eval arg1 < m
///   0 ≤ eval arg2 < m
/// Postconditions:
///   eval (from_montgomery out1) mod m = (eval (from_montgomery arg1) * eval (from_montgomery arg2)) mod m
///   0 ≤ eval out1 < m
///
pub fn mul(out1: *MontgomeryDomainFieldElement, arg1: MontgomeryDomainFieldElement, arg2: MontgomeryDomainFieldElement) void {
    @setRuntimeSafety(mode == .Debug);

    const x1 = (arg1[0]);
    var x2: u64 = undefined;
    var x3: u64 = undefined;
    mulxU64(&x2, &x3, x1, (arg2[0]));
    var x4: u64 = undefined;
    var x5: u64 = undefined;
    mulxU64(&x4, &x5, x2, 0x663d80ff99c27f);
    var x6: u64 = undefined;
    var x7: u64 = undefined;
    mulxU64(&x6, &x7, x4, 0x281);
    var x8: u64 = undefined;
    var x9: u1 = undefined;
    addcarryxU64(&x8, &x9, 0x0, x2, x6);
    var x10: u64 = undefined;
    var x11: u1 = undefined;
    addcarryxU64(&x10, &x11, x9, x3, x7);
    var x12: u64 = undefined;
    var x13: u1 = undefined;
    subborrowxU64(&x12, &x13, 0x0, x10, 0x281);
    var x14: u64 = undefined;
    var x15: u1 = undefined;
    subborrowxU64(&x14, &x15, x13, cast(u64, x11), cast(u64, 0x0));
    var x16: u64 = undefined;
    cmovznzU64(&x16, x15, x12, x10);
    out1[0] = x16;
}

/// The function square squares a field element in the Montgomery domain.
///
/// Preconditions:
///   0 ≤ eval arg1 < m
/// Postconditions:
///   eval (from_montgomery out1) mod m = (eval (from_montgomery arg1) * eval (from_montgomery arg1)) mod m
///   0 ≤ eval out1 < m
///
pub fn square(out1: *MontgomeryDomainFieldElement, arg1: MontgomeryDomainFieldElement) void {
    @setRuntimeSafety(mode == .Debug);

    const x1 = (arg1[0]);
    var x2: u64 = undefined;
    var x3: u64 = undefined;
    mulxU64(&x2, &x3, x1, (arg1[0]));
    var x4: u64 = undefined;
    var x5: u64 = undefined;
    mulxU64(&x4, &x5, x2, 0x663d80ff99c27f);
    var x6: u64 = undefined;
    var x7: u64 = undefined;
    mulxU64(&x6, &x7, x4, 0x281);
    var x8: u64 = undefined;
    var x9: u1 = undefined;
    addcarryxU64(&x8, &x9, 0x0, x2, x6);
    var x10: u64 = undefined;
    var x11: u1 = undefined;
    addcarryxU64(&x10, &x11, x9, x3, x7);
    var x12: u64 = undefined;
    var x13: u1 = undefined;
    subborrowxU64(&x12, &x13, 0x0, x10, 0x281);
    var x14: u64 = undefined;
    var x15: u1 = undefined;
    subborrowxU64(&x14, &x15, x13, cast(u64, x11), cast(u64, 0x0));
    var x16: u64 = undefined;
    cmovznzU64(&x16, x15, x12, x10);
    out1[0] = x16;
}

/// The function add adds two field elements in the Montgomery domain.
///
/// Preconditions:
///   0 ≤ eval arg1 < m
///   0 ≤ eval arg2 < m
/// Postconditions:
///   eval (from_montgomery out1) mod m = (eval (from_montgomery arg1) + eval (from_montgomery arg2)) mod m
///   0 ≤ eval out1 < m
///
pub fn add(out1: *MontgomeryDomainFieldElement, arg1: MontgomeryDomainFieldElement, arg2: MontgomeryDomainFieldElement) void {
    @setRuntimeSafety(mode == .Debug);

    var x1: u64 = undefined;
    var x2: u1 = undefined;
    addcarryxU64(&x1, &x2, 0x0, (arg1[0]), (arg2[0]));
    var x3: u64 = undefined;
    var x4: u1 = undefined;
    subborrowxU64(&x3, &x4, 0x0, x1, 0x281);
    var x5: u64 = undefined;
    var x6: u1 = undefined;
    subborrowxU64(&x5, &x6, x4, cast(u64, x2), cast(u64, 0x0));
    var x7: u64 = undefined;
    cmovznzU64(&x7, x6, x3, x1);
    out1[0] = x7;
}

/// The function sub subtracts two field elements in the Montgomery domain.
///
/// Preconditions:
///   0 ≤ eval arg1 < m
///   0 ≤ eval arg2 < m
/// Postconditions:
///   eval (from_montgomery out1) mod m = (eval (from_montgomery arg1) - eval (from_montgomery arg2)) mod m
///   0 ≤ eval out1 < m
///
pub fn sub(out1: *MontgomeryDomainFieldElement, arg1: MontgomeryDomainFieldElement, arg2: MontgomeryDomainFieldElement) void {
    @setRuntimeSafety(mode == .Debug);

    const x1 = (arg2[0]);
    var x2: u64 = undefined;
    var x3: u1 = undefined;
    subborrowxU64(&x2, &x3, 0x0, (arg1[0]), x1);
    var x4: u64 = undefined;
    cmovznzU64(&x4, x3, cast(u64, 0x0), 0xffffffffffffffff);
    var x5: u64 = undefined;
    var x6: u1 = undefined;
    addcarryxU64(&x5, &x6, 0x0, x2, (x4 & 0x281));
    out1[0] = x5;
}

/// The function opp negates a field element in the Montgomery domain.
///
/// Preconditions:
///   0 ≤ eval arg1 < m
/// Postconditions:
///   eval (from_montgomery out1) mod m = -eval (from_montgomery arg1) mod m
///   0 ≤ eval out1 < m
///
pub fn opp(out1: *MontgomeryDomainFieldElement, arg1: MontgomeryDomainFieldElement) void {
    @setRuntimeSafety(mode == .Debug);

    const x1 = (arg1[0]);
    var x2: u64 = undefined;
    var x3: u1 = undefined;
    subborrowxU64(&x2, &x3, 0x0, cast(u64, 0x0), x1);
    var x4: u64 = undefined;
    cmovznzU64(&x4, x3, cast(u64, 0x0), 0xffffffffffffffff);
    var x5: u64 = undefined;
    var x6: u1 = undefined;
    addcarryxU64(&x5, &x6, 0x0, x2, (x4 & 0x281));
    out1[0] = x5;
}

/// The function fromMontgomery translates a field element out of the Montgomery domain.
///
/// Preconditions:
///   0 ≤ eval arg1 < m
/// Postconditions:
///   eval out1 mod m = (eval arg1 * ((2^64)⁻¹ mod m)^1) mod m
///   0 ≤ eval out1 < m
///
pub fn fromMontgomery(out1: *NonMontgomeryDomainFieldElement, arg1: MontgomeryDomainFieldElement) void {
    @setRuntimeSafety(mode == .Debug);

    const x1 = (arg1[0]);
    var x2: u64 = undefined;
    var x3: u64 = undefined;
    mulxU64(&x2, &x3, x1, 0x663d80ff99c27f);
    var x4: u64 = undefined;
    var x5: u64 = undefined;
    mulxU64(&x4, &x5, x2, 0x281);
    var x6: u64 = undefined;
    var x7: u1 = undefined;
    addcarryxU64(&x6, &x7, 0x0, x1, x4);
    const x8 = (cast(u64, x7) + x5);
    var x9: u64 = undefined;
    var x10: u1 = undefined;
    subborrowxU64(&x9, &x10, 0x0, x8, 0x281);
    var x11: u64 = undefined;
    var x12: u1 = undefined;
    subborrowxU64(&x11, &x12, x10, cast(u64, 0x0), cast(u64, 0x0));
    var x13: u64 = undefined;
    cmovznzU64(&x13, x12, x9, x8);
    out1[0] = x13;
}

/// The function toMontgomery translates a field element into the Montgomery domain.
///
/// Preconditions:
///   0 ≤ eval arg1 < m
/// Postconditions:
///   eval (from_montgomery out1) mod m = eval arg1 mod m
///   0 ≤ eval out1 < m
///
pub fn toMontgomery(out1: *MontgomeryDomainFieldElement, arg1: NonMontgomeryDomainFieldElement) void {
    @setRuntimeSafety(mode == .Debug);

    const x1 = (arg1[0]);
    var x2: u64 = undefined;
    var x3: u64 = undefined;
    mulxU64(&x2, &x3, x1, 0x663d80ff99c27f);
    var x4: u64 = undefined;
    var x5: u64 = undefined;
    mulxU64(&x4, &x5, x2, 0x281);
    var x6: u64 = undefined;
    var x7: u1 = undefined;
    addcarryxU64(&x6, &x7, 0x0, x1, x4);
    const x8 = (cast(u64, x7) + x5);
    var x9: u64 = undefined;
    var x10: u1 = undefined;
    subborrowxU64(&x9, &x10, 0x0, x8, 0x281);
    var x11: u64 = undefined;
    var x12: u1 = undefined;
    subborrowxU64(&x11, &x12, x10, cast(u64, 0x0), cast(u64, 0x0));
    var x13: u64 = undefined;
    cmovznzU64(&x13, x12, x9, x8);
    out1[0] = x13;
}

/// The function nonzero outputs a single non-zero word if the input is non-zero and zero otherwise.
///
/// Preconditions:
///   0 ≤ eval arg1 < m
/// Postconditions:
///   out1 = 0 ↔ eval (from_montgomery arg1) mod m = 0
///
/// Input Bounds:
///   arg1: [[0x0 ~> 0xffffffffffffffff]]
/// Output Bounds:
///   out1: [0x0 ~> 0xffffffffffffffff]
pub fn nonzero(out1: *u64, arg1: [1]u64) void {
    @setRuntimeSafety(mode == .Debug);

    const x1 = (arg1[0]);
    out1.* = x1;
}

/// The function selectznz is a multi-limb conditional select.
///
/// Postconditions:
///   out1 = (if arg1 = 0 then arg2 else arg3)
///
/// Input Bounds:
///   arg1: [0x0 ~> 0x1]
///   arg2: [[0x0 ~> 0xffffffffffffffff]]
///   arg3: [[0x0 ~> 0xffffffffffffffff]]
/// Output Bounds:
///   out1: [[0x0 ~> 0xffffffffffffffff]]
pub fn selectznz(out1: *[1]u64, arg1: u1, arg2: [1]u64, arg3: [1]u64) void {
    @setRuntimeSafety(mode == .Debug);

    var x1: u64 = undefined;
    cmovznzU64(&x1, arg1, (arg2[0]), (arg3[0]));
    out1[0] = x1;
}

/// The function toBytes serializes a field element NOT in the Montgomery domain to bytes in little-endian order.
///
/// Preconditions:
///   0 ≤ eval arg1 < m
/// Postconditions:
///   out1 = map (λ x, ⌊((eval arg1 mod m) mod 2^(8 * (x + 1))) / 2^(8 * x)⌋) [0..1]
///
/// Input Bounds:
///   arg1: [[0x0 ~> 0x3ff]]
/// Output Bounds:
///   out1: [[0x0 ~> 0xff], [0x0 ~> 0x3]]
pub fn toBytes(out1: *[2]u8, arg1: [1]u64) void {
    @setRuntimeSafety(mode == .Debug);

    const x1 = (arg1[0]);
    const x2 = cast(u8, (x1 & cast(u64, 0xff)));
    const x3 = cast(u8, (x1 >> 8));
    out1[0] = x2;
    out1[1] = x3;
}

/// The function fromBytes deserializes a field element NOT in the Montgomery domain from bytes in little-endian order.
///
/// Preconditions:
///   0 ≤ bytes_eval arg1 < m
/// Postconditions:
///   eval out1 mod m = bytes_eval arg1 mod m
///   0 ≤ eval out1 < m
///
/// Input Bounds:
///   arg1: [[0x0 ~> 0xff], [0x0 ~> 0x3]]
/// Output Bounds:
///   out1: [[0x0 ~> 0x3ff]]
pub fn fromBytes(out1: *[1]u64, arg1: [2]u8) void {
    @setRuntimeSafety(mode == .Debug);

    const x1 = (cast(u64, (arg1[1])) << 8);
    const x2 = (arg1[0]);
    const x3 = (x1 + cast(u64, x2));
    out1[0] = x3;
}

/// The function setOne returns the field element one in the Montgomery domain.
///
/// Postconditions:
///   eval (from_montgomery out1) mod m = 1 mod m
///   0 ≤ eval out1 < m
///
pub fn setOne(out1: *MontgomeryDomainFieldElement) void {
    @setRuntimeSafety(mode == .Debug);

    out1[0] = 0x1;
}

/// The function msat returns the saturated representation of the prime modulus.
///
/// Postconditions:
///   twos_complement_eval out1 = m
///   0 ≤ eval out1 < m
///
/// Output Bounds:
///   out1: [[0x0 ~> 0xffffffffffffffff], [0x0 ~> 0xffffffffffffffff]]
pub fn msat(out1: *[2]u64) void {
    @setRuntimeSafety(mode == .Debug);

    out1[0] = 0x281;
    out1[1] = cast(u64, 0x0);
}

/// The function divstepPrecomp returns the precomputed value for Bernstein-Yang-inversion (in montgomery form).
///
/// Postconditions:
///   eval (from_montgomery out1) = ⌊(m - 1) / 2⌋^(if ⌊log2 m⌋ + 1 < 46 then ⌊(49 * (⌊log2 m⌋ + 1) + 80) / 17⌋ else ⌊(49 * (⌊log2 m⌋ + 1) + 57) / 17⌋)
///   0 ≤ eval out1 < m
///
/// Output Bounds:
///   out1: [[0x0 ~> 0xffffffffffffffff]]
pub fn divstepPrecomp(out1: *[1]u64) void {
    @setRuntimeSafety(mode == .Debug);

    out1[0] = 0x140;
}

/// The function divstep computes a divstep.
///
/// Preconditions:
///   0 ≤ eval arg4 < m
///   0 ≤ eval arg5 < m
/// Postconditions:
///   out1 = (if 0 < arg1 ∧ (twos_complement_eval arg3) is odd then 1 - arg1 else 1 + arg1)
///   twos_complement_eval out2 = (if 0 < arg1 ∧ (twos_complement_eval arg3) is odd then twos_complement_eval arg3 else twos_complement_eval arg2)
///   twos_complement_eval out3 = (if 0 < arg1 ∧ (twos_complement_eval arg3) is odd then ⌊(twos_complement_eval arg3 - twos_complement_eval arg2) / 2⌋ else ⌊(twos_complement_eval arg3 + (twos_complement_eval arg3 mod 2) * twos_complement_eval arg2) / 2⌋)
///   eval (from_montgomery out4) mod m = (if 0 < arg1 ∧ (twos_complement_eval arg3) is odd then (2 * eval (from_montgomery arg5)) mod m else (2 * eval (from_montgomery arg4)) mod m)
///   eval (from_montgomery out5) mod m = (if 0 < arg1 ∧ (twos_complement_eval arg3) is odd then (eval (from_montgomery arg4) - eval (from_montgomery arg4)) mod m else (eval (from_montgomery arg5) + (twos_complement_eval arg3 mod 2) * eval (from_montgomery arg4)) mod m)
///   0 ≤ eval out5 < m
///   0 ≤ eval out5 < m
///   0 ≤ eval out2 < m
///   0 ≤ eval out3 < m
///
/// Input Bounds:
///   arg1: [0x0 ~> 0xffffffffffffffff]
///   arg2: [[0x0 ~> 0xffffffffffffffff], [0x0 ~> 0xffffffffffffffff]]
///   arg3: [[0x0 ~> 0xffffffffffffffff], [0x0 ~> 0xffffffffffffffff]]
///   arg4: [[0x0 ~> 0xffffffffffffffff]]
///   arg5: [[0x0 ~> 0xffffffffffffffff]]
/// Output Bounds:
///   out1: [0x0 ~> 0xffffffffffffffff]
///   out2: [[0x0 ~> 0xffffffffffffffff], [0x0 ~> 0xffffffffffffffff]]
///   out3: [[0x0 ~> 0xffffffffffffffff], [0x0 ~> 0xffffffffffffffff]]
///   out4: [[0x0 ~> 0xffffffffffffffff]]
///   out5: [[0x0 ~> 0xffffffffffffffff]]
pub fn divstep(out1: *u64, out2: *[2]u64, out3: *[2]u64, out4: *[1]u64, out5: *[1]u64, arg1: u64, arg2: [2]u64, arg3: [2]u64, arg4: [1]u64, arg5: [1]u64) void {
    @setRuntimeSafety(mode == .Debug);

    var x1: u64 = undefined;
    var x2: u1 = undefined;
    addcarryxU64(&x1, &x2, 0x0, (~arg1), cast(u64, 0x1));
    const x3 = (cast(u1, (x1 >> 63)) & cast(u1, ((arg3[0]) & cast(u64, 0x1))));
    var x4: u64 = undefined;
    var x5: u1 = undefined;
    addcarryxU64(&x4, &x5, 0x0, (~arg1), cast(u64, 0x1));
    var x6: u64 = undefined;
    cmovznzU64(&x6, x3, arg1, x4);
    var x7: u64 = undefined;
    cmovznzU64(&x7, x3, (arg2[0]), (arg3[0]));
    var x8: u64 = undefined;
    cmovznzU64(&x8, x3, (arg2[1]), (arg3[1]));
    var x9: u64 = undefined;
    var x10: u1 = undefined;
    addcarryxU64(&x9, &x10, 0x0, cast(u64, 0x1), (~(arg2[0])));
    var x11: u64 = undefined;
    var x12: u1 = undefined;
    addcarryxU64(&x11, &x12, x10, cast(u64, 0x0), (~(arg2[1])));
    var x13: u64 = undefined;
    cmovznzU64(&x13, x3, (arg3[0]), x9);
    var x14: u64 = undefined;
    cmovznzU64(&x14, x3, (arg3[1]), x11);
    var x15: u64 = undefined;
    cmovznzU64(&x15, x3, (arg4[0]), (arg5[0]));
    var x16: u64 = undefined;
    var x17: u1 = undefined;
    addcarryxU64(&x16, &x17, 0x0, x15, x15);
    var x18: u64 = undefined;
    var x19: u1 = undefined;
    subborrowxU64(&x18, &x19, 0x0, x16, 0x281);
    var x20: u64 = undefined;
    var x21: u1 = undefined;
    subborrowxU64(&x20, &x21, x19, cast(u64, x17), cast(u64, 0x0));
    const x22 = (arg4[0]);
    var x23: u64 = undefined;
    var x24: u1 = undefined;
    subborrowxU64(&x23, &x24, 0x0, cast(u64, 0x0), x22);
    var x25: u64 = undefined;
    cmovznzU64(&x25, x24, cast(u64, 0x0), 0xffffffffffffffff);
    var x26: u64 = undefined;
    var x27: u1 = undefined;
    addcarryxU64(&x26, &x27, 0x0, x23, (x25 & 0x281));
    var x28: u64 = undefined;
    cmovznzU64(&x28, x3, (arg5[0]), x26);
    const x29 = cast(u1, (x13 & cast(u64, 0x1)));
    var x30: u64 = undefined;
    cmovznzU64(&x30, x29, cast(u64, 0x0), x7);
    var x31: u64 = undefined;
    cmovznzU64(&x31, x29, cast(u64, 0x0), x8);
    var x32: u64 = undefined;
    var x33: u1 = undefined;
    addcarryxU64(&x32, &x33, 0x0, x13, x30);
    var x34: u64 = undefined;
    var x35: u1 = undefined;
    addcarryxU64(&x34, &x35, x33, x14, x31);
    var x36: u64 = undefined;
    cmovznzU64(&x36, x29, cast(u64, 0x0), x15);
    var x37: u64 = undefined;
    var x38: u1 = undefined;
    addcarryxU64(&x37, &x38, 0x0, x28, x36);
    var x39: u64 = undefined;
    var x40: u1 = undefined;
    subborrowxU64(&x39, &x40, 0x0, x37, 0x281);
    var x41: u64 = undefined;
    var x42: u1 = undefined;
    subborrowxU64(&x41, &x42, x40, cast(u64, x38), cast(u64, 0x0));
    var x43: u64 = undefined;
    var x44: u1 = undefined;
    addcarryxU64(&x43, &x44, 0x0, x6, cast(u64, 0x1));
    const x45 = ((x32 >> 1) | ((x34 << 63) & 0xffffffffffffffff));
    const x46 = ((x34 & 0x8000000000000000) | (x34 >> 1));
    var x47: u64 = undefined;
    cmovznzU64(&x47, x21, x18, x16);
    var x48: u64 = undefined;
    cmovznzU64(&x48, x42, x39, x37);
    out1.* = x43;
    out2[0] = x7;
    out2[1] = x8;
    out3[0] = x45;
    out3[1] = x46;
    out4[0] = x47;
    out5[0] = x48;
}
