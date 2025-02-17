const Field = @import("field_element.zig").Field;

pub const BLS12 = Field(.{
    .fiat = @import("curves/bls12_381.zig"),
    .field_order = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab,
    .encoded_length = 48,
    .field_bits = 381,
});

pub const Vesta = Field(.{
    .fiat = @import("curves/vesta.zig"),
    .field_order = 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001,
    .encoded_length = 32,
    .field_bits = 255,
});

pub const F641 = Field(.{
    .fiat = @import("curves/f641.zig"),
    .field_order = 641,
    .encoded_length = 2,
    .field_bits = 10,
});
