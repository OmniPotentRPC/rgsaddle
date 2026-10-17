//! The eight Python IntEnum tables and the Rust `to_abi` ordinals.

use rgsaddle::{
    CellChart, EigenDevice, ExpandKind, ForceGate, HessUpdate, IrcKind, MinModeKind, RestrictedKind,
};

#[test]
fn to_abi_matches_the_python_and_c_table() {
    assert_eq!(ForceGate::L2Norm.to_abi(), 0);
    assert_eq!(ForceGate::LinfNorm.to_abi(), 1);
    assert_eq!(ForceGate::MaxForceOnAtom.to_abi(), 2);

    assert_eq!(HessUpdate::Bfgs.to_abi(), 0);
    assert_eq!(HessUpdate::TsBfgs.to_abi(), 1);
    assert_eq!(HessUpdate::Bofill.to_abi(), 2);
    assert_eq!(HessUpdate::try_from_abi(2), Some(HessUpdate::Bofill));
    assert_eq!(HessUpdate::try_from_abi(99), None);

    assert_eq!(IrcKind::Gs2.to_abi(), 0);
    assert_eq!(IrcKind::Morokuma.to_abi(), 1);

    assert_eq!(MinModeKind::Dimer.to_abi(), 0);
    assert_eq!(MinModeKind::Lanczos.to_abi(), 1);

    assert_eq!(EigenDevice::Host.to_abi(), 0);
    assert_eq!(EigenDevice::Dlpk.to_abi(), 1);

    assert_eq!(ExpandKind::Lanczos.to_abi(), 0);
    assert_eq!(ExpandKind::Gd.to_abi(), 1);
    assert_eq!(ExpandKind::Jd0.to_abi(), 2);
    assert_eq!(ExpandKind::Jd0Alt.to_abi(), 3);
    assert_eq!(ExpandKind::Mjd0.to_abi(), 4);
    assert_eq!(ExpandKind::Mjd0Alt.to_abi(), 5);

    assert_eq!(CellChart::Entries.to_abi(), 0);
    assert_eq!(CellChart::LogDeform.to_abi(), 1);

    assert_eq!(RestrictedKind::TrustRegion.to_abi(), 0);
    assert_eq!(RestrictedKind::MaxInternalStep.to_abi(), 1);
    assert_eq!(RestrictedKind::RestrictedAtomicStep.to_abi(), 2);
}
