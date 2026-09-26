//! Dest client of vocn auto-find.
//!
//! Bond / angle / dihedral *topology* stays in vocn. Dest loads the
//! primitive set vocn returns into a [`Constraints`] chart. Dest
//! does not generate Bond / Angle / Dihedral primitives.

use ndarray::ArrayView1;

use crate::SaddleError;
use crate::constraints::Constraints;

/// One vocn primitive. Dest stores the atom indices only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Primitive {
    Bond([usize; 2]),
    Angle([usize; 3]),
    Dihedral([usize; 4]),
}

/// Primitive set returned by vocn auto-find.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Found {
    primitives: Vec<Primitive>,
}

impl Found {
    /// Empty set. [`InternalPes::from_find`][crate::InternalPes::from_find]
    /// refuses it.
    pub fn new() -> Self {
        Self::default()
    }

    /// Pack bonds, then angles, then dihedrals (Sella internals order).
    pub fn from_parts(
        bonds: impl IntoIterator<Item = [usize; 2]>,
        angles: impl IntoIterator<Item = [usize; 3]>,
        dihedrals: impl IntoIterator<Item = [usize; 4]>,
    ) -> Self {
        let mut primitives = Vec::new();
        for atoms in bonds {
            primitives.push(Primitive::Bond(atoms));
        }
        for atoms in angles {
            primitives.push(Primitive::Angle(atoms));
        }
        for atoms in dihedrals {
            primitives.push(Primitive::Dihedral(atoms));
        }
        Self { primitives }
    }

    /// Active primitives, Sella packing order.
    pub fn primitives(&self) -> &[Primitive] {
        &self.primitives
    }

    /// Chart length dest loads into [`crate::InternalPes`].
    pub fn n_int(&self) -> usize {
        self.primitives.len()
    }

    pub fn is_empty(&self) -> bool {
        self.primitives.is_empty()
    }

    /// Push the vocn set onto a chart. Targets are the current values.
    pub fn load(&self, chart: &mut Constraints, x: ArrayView1<f64>) -> Result<(), SaddleError> {
        for p in &self.primitives {
            match p {
                Primitive::Bond(atoms) => chart.fix_bond(*atoms, x, None)?,
                Primitive::Angle(atoms) => chart.fix_angle(*atoms, x, None)?,
                Primitive::Dihedral(atoms) => chart.fix_dihedral(*atoms, x, None)?,
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internal::pack_cart;

    #[test]
    fn from_parts_counts_sella_order() {
        let found = Found::from_parts([[0, 1], [0, 2]], [[1, 0, 2]], []);
        assert_eq!(found.n_int(), 3);
        assert!(!found.is_empty());
        match found.primitives() {
            [
                Primitive::Bond([0, 1]),
                Primitive::Bond([0, 2]),
                Primitive::Angle([1, 0, 2]),
            ] => {}
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn load_writes_the_chart() {
        let x = pack_cart(&[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]);
        let found = Found::from_parts([[0, 1], [0, 2]], [[1, 0, 2]], []);
        let mut chart = Constraints::new(3).unwrap();
        found.load(&mut chart, x.view()).unwrap();
        let c = chart.counts();
        assert_eq!(c.nbonds, 2);
        assert_eq!(c.nangles, 1);
        assert_eq!(c.ndihedrals, 0);
        assert_eq!(chart.equalities().len(), found.n_int());
    }
}
