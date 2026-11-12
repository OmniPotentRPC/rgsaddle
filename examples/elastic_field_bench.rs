//! Time the direct pair sum and the tree field solver.
//!
//! The direct sum stops at 2000 defects. The tree walk continues through
//! 100000. Raw rows go to `benchmarks/elastic_field_scaling.csv`.
//!
//! ```text
//! cargo run --release --example elastic_field_bench
//! ```

use std::fs;
use std::time::Instant;

use rgsaddle::{DefectSource, DirectSolver, ElasticDipole, FieldSolver, FmmSolver, Kelvin};

fn dipole(value: f64) -> ElasticDipole {
    let mut tensor = [[0.0; 3]; 3];
    tensor[0][1] = value;
    tensor[1][0] = value;
    tensor[2][2] = 0.3 * value;
    ElasticDipole::from_tensor(tensor).unwrap()
}

fn defects(n: usize) -> Vec<DefectSource> {
    (0..n)
        .map(|i| {
            let t = i as f64 * 0.6180339887;
            DefectSource {
                position: [
                    (t * 12.989).sin() * 8.0,
                    (t * 7.123).cos() * 8.0,
                    ((i * 3) as f64 * 0.17).sin() * 8.0,
                ],
                dipole: dipole(0.4 + 0.2 * (i as f64 * 0.31).sin()),
            }
        })
        .collect()
}

fn main() {
    let green = Kelvin::new(1.0, 0.3).unwrap();
    let direct = DirectSolver { green };
    let fast = FmmSolver::new(green);
    let mut rows =
        String::from("n,direct_s,fmm_s,max_abs_strain_error,max_abs_strain,relative_error\n");
    for n in [
        100, 200, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000,
    ] {
        let sample = defects(n);
        if n <= 2000 {
            let started = Instant::now();
            let reference = direct.fields(&sample).unwrap();
            let direct_s = started.elapsed().as_secs_f64();
            let started = Instant::now();
            let approx = fast.fields(&sample).unwrap();
            let fmm_s = started.elapsed().as_secs_f64();
            let mut scale = 0.0_f64;
            let mut error = 0.0_f64;
            for (left, right) in reference.iter().zip(&approx) {
                for i in 0..3 {
                    for j in 0..3 {
                        scale = scale.max(left.strain.tensor()[i][j].abs());
                        error = error
                            .max((left.strain.tensor()[i][j] - right.strain.tensor()[i][j]).abs());
                    }
                }
            }
            let relative = error / scale.max(1e-30);
            rows.push_str(&format!(
                "{n},{direct_s},{fmm_s},{error},{scale},{relative}\n"
            ));
            println!(
                "n {n} direct {direct_s:.4}s fmm {fmm_s:.4}s error {error:.3e} relative {relative:.3e}"
            );
        } else {
            let started = Instant::now();
            let _approx = fast.fields(&sample).unwrap();
            let fmm_s = started.elapsed().as_secs_f64();
            rows.push_str(&format!("{n},,{fmm_s},,,\n"));
            println!("n {n} fmm {fmm_s:.4}s");
        }
    }
    let path = "benchmarks/elastic_field_scaling.csv";
    fs::write(path, rows).unwrap();
    println!("wrote {path}");
}
