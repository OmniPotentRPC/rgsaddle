fn isolated_gpu_test(name: &str, disabled: bool) -> bool {
    if std::env::var("RGSADDLE_GPU_TEST_CHILD").as_deref() == Ok(name) { return false; }
    let mut command = std::process::Command::new(std::env::current_exe().expect("test executable"));
    command.args(["--exact", name, "--nocapture"])
        .env("RGSADDLE_GPU_TEST_CHILD", name)
        .env_remove("RGSADDLE_DISABLE_GPU")
        .env_remove("SELLA_DISABLE_GPU")
        .env_remove("RGSADDLE_GPU_MIN_DIM")
        .env_remove("SELLA_GPU_MIN_DIM");
    if disabled { command.env("SELLA_DISABLE_GPU", "1"); }
    let output = command.output().expect("isolated GPU test process");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "isolated GPU test {name} failed: {}\n{stdout}\n{stderr}", output.status);
    assert!(stdout.lines().any(|line| line.starts_with("test result: ok. 1 passed; 0 failed;")),
        "isolated GPU test {name} must execute exactly one case:\n{stdout}\n{stderr}");
    true
}
