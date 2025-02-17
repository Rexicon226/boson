const std = @import("std");
const boson = @import("boson");

const BLS = boson.curves.BLS12;

pub fn main() !void {
    const lhs = try BLS.fromInt(10);
    const rhs = try BLS.fromInt(20);

    const result = lhs.sub(rhs);

    std.debug.print("reuslt: {}\n", .{result});
}
