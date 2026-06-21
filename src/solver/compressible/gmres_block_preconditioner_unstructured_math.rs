//! block LU-SGS 小型块/向量运算 helper。

use crate::core::Real;
use crate::solver::compressible::gmres_implicit_3d::CONSERVED_COMPONENTS_3D;

pub(super) fn cell_vector(values: &[Real], cell: usize) -> [Real; CONSERVED_COMPONENTS_3D] {
    let start = cell * CONSERVED_COMPONENTS_3D;
    [
        values[start],
        values[start + 1],
        values[start + 2],
        values[start + 3],
        values[start + 4],
    ]
}

pub(super) fn block_slice(blocks: &[Real], cell: usize) -> &[Real] {
    let start = cell * CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D;
    &blocks[start..start + CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D]
}

pub(super) fn subtract_block_product(
    out: &mut [Real; CONSERVED_COMPONENTS_3D],
    block: &[Real],
    vector: &[Real; CONSERVED_COMPONENTS_3D],
    neighbor_damping: Real,
) {
    let damping = if neighbor_damping >= 1.0 - Real::EPSILON {
        1.0
    } else {
        neighbor_damping.max(0.0)
    };
    for row in 0..CONSERVED_COMPONENTS_3D {
        let mut value = 0.0;
        for col in 0..CONSERVED_COMPONENTS_3D {
            value += block[row * CONSERVED_COMPONENTS_3D + col] * vector[col];
        }
        out[row] -= damping * value;
    }
}

pub(super) fn block_vector_product(
    block: &[Real],
    vector: &[Real; CONSERVED_COMPONENTS_3D],
) -> [Real; CONSERVED_COMPONENTS_3D] {
    let mut out = [0.0; CONSERVED_COMPONENTS_3D];
    for row in 0..CONSERVED_COMPONENTS_3D {
        for col in 0..CONSERVED_COMPONENTS_3D {
            out[row] += block[row * CONSERVED_COMPONENTS_3D + col] * vector[col];
        }
    }
    out
}

pub(super) fn write_cell_vector_from_block_product(
    out: &mut [Real],
    cell: usize,
    block: &[Real],
    vector: &[Real; CONSERVED_COMPONENTS_3D],
) {
    let start = cell * CONSERVED_COMPONENTS_3D;
    let product = block_vector_product(block, vector);
    out[start..start + CONSERVED_COMPONENTS_3D].copy_from_slice(&product);
}
