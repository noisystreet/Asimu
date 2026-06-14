//! 不可压缩闭域压力参考处理。

use crate::core::Real;
use crate::mesh::StructuredMesh3d;

pub(crate) fn volume_weighted_pressure_mean(values: &[Real], mesh: &StructuredMesh3d) -> Real {
    let mut weighted_sum = 0.0;
    let mut volume_sum = 0.0;
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let cell = mesh.cell_index(i, j, k);
                let volume = mesh.cell_metric(i, j, k).volume;
                weighted_sum += values[cell] * volume;
                volume_sum += volume;
            }
        }
    }
    if volume_sum <= Real::EPSILON {
        0.0
    } else {
        weighted_sum / volume_sum
    }
}
