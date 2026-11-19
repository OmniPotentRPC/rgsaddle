//! Compiles only when the consumer's pins and rgsaddle's resolve to one
//! rgmin and one linkcell.

// In the graph for its eindir-core and dlpk edges; rgsaddle exposes no
// rgpot-core type to compare against.
use rgpot_core as _;

pub fn trust_region(region: rgmin::TrustRegion) -> rgsaddle::trust::TrustRegion {
    region
}

pub fn quickmin(options: rgmin::QuickMinOptions) -> rgmin::QuickMinOptions {
    rgsaddle::band::BandConfig::default()
        .with_quickmin_options(options)
        .quickmin_options
}

pub fn cell(cell: linkcell::Cell) -> rgsaddle::mic::Cell {
    cell
}

pub fn minimage_cell(cell: minimage::Cell) -> rgsaddle::mic::Cell {
    cell
}
