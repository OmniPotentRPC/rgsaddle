use ndarray::{Array1, ArrayView1, array};
use rgsaddle::{PointSurface, SaddleError, action_pair_blocks};

struct Identity;

impl PointSurface for Identity {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        Ok((0.0, Array1::zeros(x.len())))
    }

    fn hessian_vector(
        &self,
        _x: ArrayView1<f64>,
        v: ArrayView1<f64>,
    ) -> Result<Option<Array1<f64>>, SaddleError> {
        Ok(Some(v.to_owned()))
    }
}

#[test]
fn action_blocks_read_the_hessian_product() {
    let x = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let blocks = action_pair_blocks(&Identity, x.view(), &[(0, 0), (0, 1)]).unwrap();
    assert_eq!(blocks.len(), 2);
    let mut diagonal = [0.0; 9];
    for axis in 0..3 {
        diagonal[axis * 3 + axis] = 1.0;
    }
    assert_eq!(blocks[0], diagonal);
    assert_eq!(blocks[1], [0.0; 9]);
}

#[test]
fn a_point_that_is_not_cartesian_is_refused() {
    let x = array![0.0, 0.0];
    let error = action_pair_blocks(&Identity, x.view(), &[]).unwrap_err();
    assert!(matches!(error, SaddleError::Shape(_)));
}
