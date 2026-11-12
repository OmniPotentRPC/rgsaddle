//! Read a strain-rate table and print k(ε, T).
//!
//! The copper table is `benchmarks/strain_rate_table_cu.json`. Helmholtz,
//! Gibbs and the dipole line stay in separate columns. A missing barrier
//! is reported as `no barrier` and is not replaced by another ensemble.
//!
//! ```text
//! cargo run --release --example strain_rate
//! cargo run --release --example strain_rate -- benchmarks/strain_rate_table_cu.json 300
//! ```

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use rgsaddle::{
    Ensemble, SaddleError, StrainRateTable, hydrostatic, shear_xz, shear_yx, uniaxial_x,
};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), SaddleError> {
    let mut args = env::args().skip(1);
    let path = args.next().map(PathBuf::from).unwrap_or_else(default_table);
    let temperature: f64 = args
        .next()
        .as_deref()
        .unwrap_or("300")
        .parse()
        .map_err(|_| SaddleError::Invalid("temperature must be a number of kelvin".into()))?;
    let table = StrainRateTable::open(&path)?;
    println!("table {}", path.display());
    println!("T = {temperature} K");
    println!("zero-strain barrier {:.8} eV", table.model().barrier0);
    if let Some(omega) = table.activation_volume() {
        let v = rgsaddle::strain_rate::voigt(omega);
        println!(
            "activation volume A3  xx {:.6} yy {:.6} zz {:.6} yz {:.6} xz {:.6} xy {:.6}",
            v[0], v[1], v[2], v[3], v[4], v[5]
        );
    }
    let queries = [
        ("shear_yx 0", shear_yx(0.0)),
        ("shear_yx 0.005", shear_yx(0.005)),
        ("shear_yx 0.01", shear_yx(0.01)),
        ("shear_yx 0.02", shear_yx(0.02)),
        ("shear_yx 0.05", shear_yx(0.05)),
        ("shear_xz 0.02", shear_xz(0.02)),
        ("uniaxial_x 0.01", uniaxial_x(0.01)),
        ("hydrostatic 0.01", hydrostatic(0.01)),
    ];
    for (label, strain) in queries {
        println!("{label}");
        for ensemble in [Ensemble::Linear, Ensemble::Helmholtz, Ensemble::Gibbs] {
            let name = match ensemble {
                Ensemble::Linear => "linear",
                Ensemble::Helmholtz => "helmholtz",
                Ensemble::Gibbs => "gibbs",
            };
            match table.lookup(strain, temperature, ensemble) {
                Ok(rate) => println!("  {name:<10} {rate}"),
                Err(err) => println!("  {name:<10} {err}"),
            }
        }
    }
    Ok(())
}

fn default_table() -> PathBuf {
    let relative = PathBuf::from("benchmarks/strain_rate_table_cu.json");
    if relative.exists() {
        return relative;
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/strain_rate_table_cu.json")
}
