//! SHA-256 of `eigh` and `partial_eigh` on every SIMD path and pool size.
//!
//! The digest covers little-endian `f64` eigenvalues, then the
//! eigenvector columns in column-major order. One golden file records
//! the scalar path. AVX2, AVX-512, NEON, and the scalar path must
//! match it, at pools 1, 2, 3, and 8.
//!
//! `RGSADDLE_WRITE_GOLDEN=1` rewrites the file from the scalar kernel
//! on one thread. Run this binary with `--test-threads=1`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use ndarray::Array2;
use sha2::{Digest, Sha256};

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/eigh.sha256")
}

fn sizes() -> Vec<usize> {
    let mut sizes = vec![48usize];
    if !cfg!(debug_assertions) {
        sizes.push(256);
        sizes.push(600);
    }
    sizes
}

fn supported_simd() -> Vec<&'static str> {
    let mut paths = vec!["scalar"];
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            paths.push("avx2");
        }
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
            paths.push("avx512");
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        paths.push("neon");
    }
    paths
}

/// Symmetric matrix with a negative zero, a subnormal, and a repeated
/// eigenvalue. The generator is `f64::mul_add`, matching hardware FMA.
fn matrix(n: usize) -> Array2<f64> {
    let mut state = 0xE161_u64.wrapping_add(n as u64);
    let mut a = Array2::zeros((n, n));
    for col in 0..n {
        for row in col..n {
            state = state
                .wrapping_mul(0x5851_F42D_4C95_7F2D)
                .wrapping_add(0x1405_7B7E_F767_814F);
            let unit = (state >> 11) as u32 as f64 / u32::MAX as f64;
            let value = unit * 2.0 - 1.0;
            a[(row, col)] = value;
            a[(col, row)] = value;
        }
    }
    a[(0, 0)] = -0.0;
    if n > 2 {
        let subnormal = f64::from_bits(1);
        a[(1, 2)] = subnormal;
        a[(2, 1)] = subnormal;
    }
    if n > 4 {
        let i = n - 2;
        let j = n - 1;
        for k in 0..n {
            a[(i, k)] = 0.0;
            a[(k, i)] = 0.0;
            a[(j, k)] = 0.0;
            a[(k, j)] = 0.0;
        }
        a[(i, i)] = 1.5;
        a[(j, j)] = 1.5;
    }
    a
}

fn digest(evals: &ndarray::Array1<f64>, vecs: &Array2<f64>) -> String {
    let mut hasher = Sha256::new();
    for value in evals.iter() {
        hasher.update(value.to_le_bytes());
    }
    let n = vecs.nrows();
    let k = vecs.ncols();
    for col in 0..k {
        for row in 0..n {
            hasher.update(vecs[(row, col)].to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

fn key(n: usize, op: &str, k: usize) -> String {
    format!("n={n} op={op} k={k}")
}

fn solve(a: &Array2<f64>, op: &str, k: usize) -> (ndarray::Array1<f64>, Array2<f64>) {
    match op {
        "eigh" => rgsaddle::eigensolve::eigh(a.view()).expect("eigh"),
        "partial" => rgsaddle::eigensolve::partial_eigh(a.view(), k).expect("partial"),
        _ => panic!("unknown op {op}"),
    }
}

fn cases() -> Vec<(usize, &'static str, usize)> {
    let mut out = Vec::new();
    for n in sizes() {
        out.push((n, "eigh", n));
        out.push((n, "partial", 1));
        out.push((n, "partial", 10));
    }
    out
}

fn load_golden() -> BTreeMap<String, String> {
    let text = std::fs::read_to_string(golden_path()).expect("golden file");
    let mut map = BTreeMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let case = parts.next().unwrap().to_string();
        let op = parts.next().unwrap();
        let k = parts.next().unwrap();
        let hash = parts.next().unwrap().to_string();
        map.insert(format!("{case} {op} {k}"), hash);
    }
    map
}

fn write_golden() {
    // SAFETY: this test is the only thread in the binary that reads
    // `RGSADDLE_SIMD`. The assignment happens before any solve.
    unsafe {
        std::env::set_var("RGSADDLE_SIMD", "scalar");
    }
    let mut lines = vec![
        "# SHA-256 of little-endian f64 eigenvalues, then column-major eigenvectors.".to_string(),
        "# Scalar f64::mul_add, one thread. AVX2+FMA, AVX-512, and NEON match these digests."
            .to_string(),
        "# Band width 64, WY width 64, chase half-width 128. Those widths are part of the bits."
            .to_string(),
        "# Partial spectra split where an off-diagonal is at most eps times the norm.".to_string(),
    ];
    for (n, op, k) in cases() {
        let a = matrix(n);
        let (evals, vecs) = solve(&a, op, k);
        lines.push(format!("{} {}", key(n, op, k), digest(&evals, &vecs)));
    }
    let path = golden_path();
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, lines.join("\n") + "\n").unwrap();
}

fn log_simd_path() {
    let requested = std::env::var("RGSADDLE_SIMD").unwrap_or_else(|_| "auto".to_string());
    #[cfg(target_arch = "x86_64")]
    let (avx512, avx2, fma) = (
        std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma"),
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma"),
        std::is_x86_feature_detected!("fma"),
    );
    #[cfg(not(target_arch = "x86_64"))]
    let (avx512, avx2, fma) = (false, false, false);
    let line = format!(
        "eigh-bits simd arch={} requested={requested} host_avx512={avx512} host_avx2={avx2} host_fma={fma} paths={:?}\n",
        std::env::consts::ARCH,
        supported_simd()
    );
    // libtest keeps eprintln in a capture buffer, so a passing run
    // never reaches the Actions log. A new open of /dev/stderr is the
    // process stderr the job record stores.
    let mut emitted = false;
    if let Ok(mut err) = std::fs::OpenOptions::new().write(true).open("/dev/stderr") {
        use std::io::Write;
        emitted = err.write_all(line.as_bytes()).is_ok();
    }
    if !emitted {
        eprint!("{line}");
    }
}

#[test]
fn eigenpairs_match_the_golden_file_on_every_simd_path() {
    log_simd_path();
    if std::env::var_os("RGSADDLE_WRITE_GOLDEN").is_some() {
        write_golden();
        return;
    }
    let golden = load_golden();
    let pools = [1usize, 2, 3, 8];
    for simd in supported_simd() {
        // SAFETY: one test, `--test-threads=1`. The variable is read
        // by `refresh_simd` at the start of each solve and by no
        // other thread outside this loop.
        unsafe {
            std::env::set_var("RGSADDLE_SIMD", simd);
        }
        for (n, op, k) in cases() {
            let a = matrix(n);
            let case = key(n, op, k);
            let expect = golden
                .get(&case)
                .unwrap_or_else(|| panic!("missing golden entry {case}"));
            for threads in pools {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .expect("pool");
                let (evals, vecs) = pool.install(|| solve(&a, op, k));
                let got = digest(&evals, &vecs);
                assert_eq!(
                    got, *expect,
                    "{case} simd={simd} threads={threads} left the golden digest"
                );
            }
        }
    }
}
