// The FCC `<100>` dumbbell translation rule shared by the copper tests.

/// Axis of a `<100>` dumbbell after a nearest-neighbor translation.
///
/// Eight of the twelve FCC neighbor hops move the center and turn the axis
/// by a right angle. The new axis is the part of `hop` perpendicular to
/// `axis`, and that part lies along one cube edge. The four hops in the
/// plane normal to `axis` are not this mechanism: their perpendicular part
/// is not a `<100>` axis. The returned axis is unsigned, because a dumbbell
/// is unchanged by reversing it.
fn dumbbell_translation_axis(axis: [f64; 3], hop: [f64; 3]) -> Option<[f64; 3]> {
    let axis_norm2 = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
    let hop_norm2 = hop[0] * hop[0] + hop[1] * hop[1] + hop[2] * hop[2];
    if !(axis_norm2.is_finite() && hop_norm2.is_finite())
        || axis_norm2 < 1.0e-18
        || hop_norm2 < 1.0e-18
    {
        return None;
    }
    let dot = axis[0] * hop[0] + axis[1] * hop[1] + axis[2] * hop[2];
    let scale = dot / axis_norm2;
    let perp = [
        hop[0] - axis[0] * scale,
        hop[1] - axis[1] * scale,
        hop[2] - axis[2] * scale,
    ];
    let floor = 1.0e-8 * hop_norm2.sqrt();
    let mut nonzero = 0usize;
    let mut which = 0usize;
    for (component, value) in perp.iter().enumerate() {
        if value.abs() > floor {
            nonzero += 1;
            which = component;
        }
    }
    if nonzero != 1 {
        return None;
    }
    let mut out = [0.0; 3];
    out[which] = 1.0;
    Some(out)
}

/// Diffusivity of a `<100>` dumbbell on an FCC lattice.
///
/// `translation_rate` is the rate of one of the eight neighbor channels.
/// Each of those hops moves the center by `lattice / sqrt(2)` and turns the
/// axis by a right angle. The eight vectors from any axis sum to zero and
/// are centrosymmetric, so the turn does not correlate successive jumps and
/// `rotation_rate` (one `<110>` saddle; two of them reach each other axis)
/// does not change the long-time diffusivity. With `λ² = lattice² / 2`,
/// `D = (1/6) * 8 * translation_rate * λ² = (2/3) * translation_rate * lattice²`.
fn dumbbell_diffusivity(
    lattice: f64,
    translation_rate: f64,
    rotation_rate: f64,
) -> Result<f64, rgsaddle::SaddleError> {
    if !(lattice.is_finite() && lattice > 0.0) {
        return Err(rgsaddle::SaddleError::Invalid(
            "the lattice parameter must be finite and positive".into(),
        ));
    }
    if !(translation_rate.is_finite()
        && translation_rate >= 0.0
        && rotation_rate.is_finite()
        && rotation_rate >= 0.0)
    {
        return Err(rgsaddle::SaddleError::Invalid(
            "translation and rotation rates must be finite and non-negative".into(),
        ));
    }
    let _ = rotation_rate;
    Ok((2.0 / 3.0) * translation_rate * lattice * lattice)
}

/// Rate of one translation channel of a `<100>` dumbbell with defect
/// diffusivity `diffusivity`, the inverse of [`dumbbell_diffusivity`]:
/// `ν = D / (α a²)` with `α = 2/3`.
///
/// This is the SIA migration frequency in `K_iv = V_iv D_i / (α a²) = V_iv ν`
/// (Mwungeli, Liu, and Demkowicz, Phys. Rev. Materials 9, 023601 (2025),
/// section 1), so `D` must be the diffusivity of the defect itself. A
/// tracer diffusivity summed over atoms, `Σ_i ⟨|Δr_i|²⟩ / 6t`, is smaller by
/// the tracer correlation factor and gives a smaller `ν`.
fn dumbbell_channel_rate(lattice: f64, diffusivity: f64) -> Result<f64, rgsaddle::SaddleError> {
    if !(lattice.is_finite() && lattice > 0.0) {
        return Err(rgsaddle::SaddleError::Invalid(
            "the lattice parameter must be finite and positive".into(),
        ));
    }
    if !(diffusivity.is_finite() && diffusivity >= 0.0) {
        return Err(rgsaddle::SaddleError::Invalid(
            "the diffusivity must be finite and non-negative".into(),
        ));
    }
    Ok(diffusivity / ((2.0 / 3.0) * lattice * lattice))
}
