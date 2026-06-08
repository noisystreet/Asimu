//! 着色桶内四路面批静态几何 SoA（init-time 预处理）。

use crate::exec::batch::ExecFaceBatchStatic4;

use super::UnstructuredInteriorFace;

/// 四内面静态几何 SoA（exec 类型；discretization 兼容别名）。
pub type InteriorFaceBatchStatic4 = ExecFaceBatchStatic4;

/// 单着色桶的四路批布局：完整四路批 + 桶尾余面。
#[derive(Debug, Clone, Default)]
pub struct InteriorFaceBucketBatchLayout {
    pub full_batches: Vec<InteriorFaceBatchStatic4>,
    pub remainder: Vec<usize>,
}

impl InteriorFaceBucketBatchLayout {
    #[must_use]
    pub fn num_faces(&self) -> usize {
        self.full_batches.len() * 4 + self.remainder.len()
    }
}

pub(super) fn build_bucket_batch_layouts(
    buckets: &[Vec<usize>],
    interior: &[UnstructuredInteriorFace],
) -> Vec<InteriorFaceBucketBatchLayout> {
    buckets
        .iter()
        .map(|bucket| build_one_bucket_batch_layout(bucket, interior))
        .collect()
}

fn build_one_bucket_batch_layout(
    bucket: &[usize],
    interior: &[UnstructuredInteriorFace],
) -> InteriorFaceBucketBatchLayout {
    let mut layout = InteriorFaceBucketBatchLayout::default();
    let full = bucket.len() - bucket.len() % 4;
    for chunk in bucket[..full].chunks_exact(4) {
        layout
            .full_batches
            .push(static_batch4_from_face_indices(chunk, interior));
    }
    layout.remainder.extend_from_slice(&bucket[full..]);
    layout
}

fn static_batch4_from_face_indices(
    face_indices: &[usize],
    interior: &[UnstructuredInteriorFace],
) -> InteriorFaceBatchStatic4 {
    let idx = [
        face_indices[0],
        face_indices[1],
        face_indices[2],
        face_indices[3],
    ];
    let mut batch = ExecFaceBatchStatic4 {
        face_indices: idx,
        owners: [0; 4],
        neighbors: [0; 4],
        nx: [0.0; 4],
        ny: [0.0; 4],
        nz: [0.0; 4],
        owner_rhs_scale: [0.0; 4],
        neighbor_rhs_scale: [0.0; 4],
        area: [0.0; 4],
        owner_volume: [0.0; 4],
        neighbor_volume: [0.0; 4],
    };
    for (lane, &face_idx) in idx.iter().enumerate() {
        let face = &interior[face_idx];
        batch.owners[lane] = face.owner;
        batch.neighbors[lane] = face.neighbor;
        batch.nx[lane] = face.normal.x;
        batch.ny[lane] = face.normal.y;
        batch.nz[lane] = face.normal.z;
        batch.owner_rhs_scale[lane] = face.owner_rhs_scale;
        batch.neighbor_rhs_scale[lane] = face.neighbor_rhs_scale;
        batch.area[lane] = face.area;
        batch.owner_volume[lane] = face.owner_volume;
        batch.neighbor_volume[lane] = face.neighbor_volume;
    }
    batch
}

#[cfg(test)]
mod tests {
    use crate::core::Vector3;

    use super::*;

    fn sample_interior_face() -> UnstructuredInteriorFace {
        UnstructuredInteriorFace {
            owner: 0,
            neighbor: 1,
            area: 1.0,
            normal: Vector3::new(1.0, 0.0, 0.0),
            owner_volume: 1.0,
            neighbor_volume: 1.0,
            inv_owner_volume: 1.0,
            inv_neighbor_volume: 1.0,
            owner_rhs_scale: -1.0,
            neighbor_rhs_scale: 1.0,
            lsq_dr: Vector3::new(1.0, 0.0, 0.0),
            lsq_w: 1.0,
            dr_owner_to_face: Vector3::new(0.5, 0.0, 0.0),
            dr_neighbor_to_face: Vector3::new(-0.5, 0.0, 0.0),
        }
    }

    #[test]
    fn bucket_batch_layout_splits_remainder() {
        let interior = (0..6).map(|_| sample_interior_face()).collect::<Vec<_>>();
        let bucket = vec![0usize, 1, 2, 3, 4, 5];
        let layout = build_one_bucket_batch_layout(&bucket, &interior);
        assert_eq!(layout.full_batches.len(), 1);
        assert_eq!(layout.remainder, vec![4, 5]);
        assert!(layout.full_batches[0].simd_eligible());
    }
}
