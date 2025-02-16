const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const boson = b.addModule("boson", .{
        .target = target,
        .optimize = optimize,
        .root_source_file = b.path("src/boson.zig"),
    });

    const test_step = b.step("test", "Test the Boson library");
    const test_exe = b.addTest(.{
        .root_source_file = b.path("src/boson.zig"),
        .target = target,
        .optimize = optimize,
    });
    const test_run = b.addRunArtifact(test_exe);
    test_step.dependOn(&test_run.step);

    inline for (.{
        .{ "groth16", "examples/groth16.zig" },
    }) |entry| {
        const name, const path = entry;

        const example = b.addExecutable(.{
            .name = name,
            .target = target,
            .optimize = optimize,
            .root_source_file = b.path(path),
        });
        example.root_module.addImport("boson", boson);

        b.installArtifact(example);

        const run = b.addRunArtifact(example);
        if (b.args) |args| run.addArgs(args);
        const run_step = b.step("run-" ++ name, "Runs the " ++ name ++ " example");
        run_step.dependOn(&run.step);
    }
}
