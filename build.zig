const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "boson",
        .target = target,
        .optimize = optimize,
        .root_source_file = b.path("src/main.zig"),
    });
    b.installArtifact(exe);
    exe.linkLibC();

    const run = b.addRunArtifact(exe);
    if (b.args) |args| run.addArgs(args);
    const run_step = b.step("run", "");
    run_step.dependOn(&run.step);

    const test_step = b.step("test", "");
    const test_exe = b.addTest(.{
        .root_source_file = b.path("src/boson.zig"),
        .target = target,
        .optimize = optimize,
    });
    const test_run = b.addRunArtifact(test_exe);
    test_step.dependOn(&test_run.step);
}
