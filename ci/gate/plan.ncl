# This is the source of both ci/gate/plan.json and
# .github/workflows/gate.yml. Run `pixi run gen-gate` after editing it.
let gate_matrix = [
  { arch = "x86_64", runner = "ubuntu-24.04", leg = "eessi" },
  { arch = "aarch64", runner = "ubuntu-24.04-arm", leg = "eessi" },
  { arch = "x86_64", runner = "ubuntu-24.04", leg = "pixi" },
  { arch = "aarch64", runner = "ubuntu-24.04-arm", leg = "pixi" },
] in
let plan = {
  schema = 1,
  max_cpus = 1,
  eessi = {
    stack_version = "2026.06",
    compiler = "GCCcore/15.2.0",
    rust_module_prefix = "Rust/",
    rust_module_suffix = "-GCCcore-15.2.0",
  },
  matrix = gate_matrix,
  legs = {
    eessi = {
      commands = [
        {
          name = "Cargo build (C ABI)",
          argv = ["cargo", "build", "--locked", "--features", "capi"],
          kind = "build",
        },
        {
          name = "Cargo test (full C ABI suite)",
          argv = ["cargo", "test", "--locked", "--features", "capi"],
          kind = "test",
        },
      ],
    },
    pixi = {
      commands = [
        {
          name = "Cargo build (all features)",
          argv = ["pixi", "run", "gate-cargo-build"],
          kind = "build",
        },
        {
          name = "Cargo test (all features)",
          argv = ["pixi", "run", "gate-cargo-test"],
          kind = "test",
        },
        {
          name = "Clippy",
          argv = ["pixi", "run", "gate-clippy"],
          kind = "lint",
        },
        {
          name = "Rustfmt",
          argv = ["pixi", "run", "gate-rustfmt"],
          kind = "lint",
        },
        {
          name = "C ABI build",
          argv = ["pixi", "run", "gate-capi-build"],
          kind = "build",
        },
        {
          name = "C ABI tests",
          argv = ["pixi", "run", "gate-capi-test"],
          kind = "test",
        },
        {
          name = "C ABI Meson subproject",
          argv = ["pixi", "run", "gate-capi-subproject"],
          kind = "test",
        },
        {
          name = "C ABI pkg-config consumer",
          argv = ["pixi", "run", "gate-capi-pkgconfig"],
          kind = "test",
        },
        {
          name = "Python extension build",
          argv = ["pixi", "run", "-e", "python", "gate-python-build"],
          kind = "build",
        },
        {
          name = "Python tests",
          argv = ["pixi", "run", "-e", "python", "gate-python-test"],
          kind = "test",
        },
      ],
    },
  },
} in
let install_eb_local = m%"
set -euo pipefail
version=3.1.0
file="eb-local-${version}-$(uname -m)-linux-musl"
install_dir="${RUNNER_TEMP}/eb-local"
mkdir -p "${install_dir}"
curl -fL "https://github.com/stackhands/eb-local/releases/download/v${version}/${file}" -o "${install_dir}/${file}"
curl -fL "https://github.com/stackhands/eb-local/releases/download/v${version}/SHA256SUMS" |
  (cd "${install_dir}" && sha256sum -c --ignore-missing)
mv "${install_dir}/${file}" "${install_dir}/eb-local"
chmod 755 "${install_dir}/eb-local"
echo "${install_dir}" >> "${GITHUB_PATH}"
"% in
let run_gate = m%"
set +e
python3 ci/gate/run.py --leg "${{ matrix.leg }}" --report "${RUNNER_TEMP}/gate.md"
status=$?
echo "status=${status}" >> "${GITHUB_OUTPUT}"
exit 0
"% in
let workflow = {
  name = "Reproducible build and test gate",
  on = {
    pull_request = null,
    workflow_dispatch = null,
  },
  permissions = {
    contents = "read",
  },
  jobs = {
    generated = {
      name = "Generated files are current",
      runs-on = "ubuntu-24.04",
      steps = [
        { uses = "actions/checkout@v4" },
        { uses = "prefix-dev/setup-pixi@v0.9.3", with = { cache = true } },
        { name = "Check generated gate files", run = "pixi run check-gate" },
      ],
    },
    gate = {
      name = m%"${{ matrix.leg }} / ${{ matrix.arch }}"%,
      runs-on = "${{ matrix.runner }}",
      strategy = {
        fail-fast = false,
        matrix = { include = gate_matrix },
      },
      steps = [
        { uses = "actions/checkout@v4" },
        {
          name = "Load EESSI",
          "if" = "matrix.leg == 'eessi'",
          uses = "EESSI/github-action-eessi@v3.4.0",
          with = { eessi_stack_version = "2026.06" },
        },
        {
          name = "Install eb-local",
          "if" = "matrix.leg == 'eessi'",
          shell = "bash",
          run = install_eb_local,
        },
        {
          name = "Set up pixi",
          "if" = "matrix.leg == 'pixi'",
          uses = "prefix-dev/setup-pixi@v0.9.3",
          with = { cache = true },
        },
        {
          name = "Run gate",
          id = "gate",
          shell = "bash",
          run = run_gate,
        },
        {
          name = "Upload gate report",
          "if" = "always()",
          uses = "actions/upload-artifact@v4",
          with = {
            name = m%"gate-report-${{ matrix.leg }}-${{ matrix.arch }}"%,
            path = "${{ runner.temp }}/gate.md",
            "if-no-files-found" = "error",
          },
        },
        {
          name = "Append report to job summary",
          "if" = "always()",
          shell = "bash",
          run = m%"cat "${RUNNER_TEMP}/gate.md" >> "${GITHUB_STEP_SUMMARY}""%,
        },
        {
          name = "Enforce gate result",
          "if" = "always()",
          shell = "bash",
          run = m%"test "${{ steps.gate.outputs.status }}" = 0"%,
        },
      ],
    },
  },
} in
let output_plan = plan in
let output_workflow = workflow in
{ plan = output_plan, workflow = output_workflow }
