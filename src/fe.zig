const std = @import("std");
const mem = std.mem;

/// Parameters to create a finite field type.
pub const FieldParams = struct {
    fiat: type,
    field_order: comptime_int,
    field_bits: comptime_int,
    saturated_bits: comptime_int,
    encoded_length: comptime_int,
};

pub fn Field(comptime params: FieldParams) type {
    const fiat = params.fiat;
    const MontgomeryDomainFieldElement = fiat.MontgomeryDomainFieldElement;
    // const NonMontgomeryDomainFieldElement = fiat.NonMontgomeryDomainFieldElement;

    return struct {
        const Fe = @This();

        limbs: MontgomeryDomainFieldElement,

        /// Field size.
        pub const field_order = params.field_order;

        /// Number of bits to represent the set of all elements.
        pub const field_bits = params.field_bits;

        /// Number of bits that can be saturated without overflowing.
        pub const saturated_bits = params.saturated_bits;

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
    };
}

pub const Vesta = Field(.{
    .fiat = @import("vesta.zig"),
    .field_order = 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001,
    .encoded_length = 256,
    .saturated_bits = 256,
    .field_bits = 256,
});
