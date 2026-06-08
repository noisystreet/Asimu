//! CPU 并行调度（`rayon` 仅经本模块；discretization / solver 不得直接依赖 rayon）。

#[cfg(feature = "parallel-fvm")]
use rayon::prelude::*;

use crate::error::Result;

/// 着色桶内并行 map、桶间串行。
#[cfg(feature = "parallel-fvm")]
pub fn par_map_colored_buckets<T, F>(buckets: &[Vec<usize>], min_len: usize, f: F) -> Vec<Vec<T>>
where
    T: Send,
    F: Fn(usize) -> T + Sync,
{
    buckets
        .iter()
        .map(|bucket| {
            bucket
                .par_iter()
                .with_min_len(min_len)
                .map(|&face_idx| f(face_idx))
                .collect()
        })
        .collect()
}

/// 面索引 slice 并行 map。
#[cfg(feature = "parallel-fvm")]
pub fn par_map_face_indices<T, F>(indices: &[usize], min_len: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync,
{
    indices
        .par_iter()
        .with_min_len(min_len)
        .map(|&face_idx| f(face_idx))
        .collect()
}

/// 面索引 slice 并行 map（可失败）。
#[cfg(feature = "parallel-fvm")]
pub fn par_try_map_face_indices<T, E, F>(
    indices: &[usize],
    min_len: usize,
    f: F,
) -> std::result::Result<Vec<T>, E>
where
    T: Send,
    F: Fn(usize) -> std::result::Result<T, E> + Sync,
    E: Send,
{
    indices
        .par_iter()
        .with_min_len(min_len)
        .map(|&face_idx| f(face_idx))
        .collect()
}

/// 静态 batch slice 并行 map（可失败）。
#[cfg(feature = "parallel-fvm")]
pub fn par_try_map_batches<T, B, E, F>(
    batches: &[B],
    min_len: usize,
    f: F,
) -> std::result::Result<Vec<T>, E>
where
    T: Send,
    B: Sync,
    F: Fn(&B) -> std::result::Result<T, E> + Sync + Send,
    E: Send,
{
    batches.par_iter().with_min_len(min_len).map(f).collect()
}

/// 三 mut/只读 slice 并行 zip。
#[cfg(feature = "parallel-fvm")]
pub fn par_for_each_zip3_mut<A, B, C, F>(a: &mut [A], b: &mut [B], c: &[C], f: F)
where
    A: Send,
    B: Send,
    C: Sync,
    F: Fn(&mut A, &mut B, &C) + Sync,
{
    a.par_iter_mut()
        .zip(b.par_iter_mut())
        .zip(c.par_iter())
        .for_each(|((x, y), z)| f(x, y, z));
}

/// 双 mut slice 与只读 slice 并行 zip。
#[cfg(feature = "parallel-fvm")]
pub fn par_for_each_zip_mut2<A, B, F>(a: &mut [A], b: &[B], f: F)
where
    A: Send,
    B: Sync,
    F: Fn(&mut A, &B) + Sync,
{
    a.par_iter_mut()
        .zip(b.par_iter())
        .for_each(|(x, y)| f(x, y));
}

/// 三 mut/只读 slice 并行 zip（可失败）。
#[cfg(feature = "parallel-fvm")]
pub fn par_try_for_each_zip3<A, B, C, E, F>(
    a: &mut [A],
    b: &mut [B],
    c: &[C],
    f: F,
) -> std::result::Result<(), E>
where
    A: Send,
    B: Send,
    C: Sync,
    F: Fn(&mut A, &mut B, &C) -> std::result::Result<(), E> + Sync,
    E: Send,
{
    a.par_iter_mut()
        .zip(b.par_iter_mut())
        .zip(c.par_iter())
        .try_for_each(|((x, y), z)| f(x, y, z))
}

/// 单元索引并行：五路 mut RHS + enumerate（IDWLS inviscid）。
#[cfg(feature = "parallel-fvm")]
pub fn par_try_for_each_cell_rhs5<E, F>(
    br: &mut [crate::core::Vector3],
    bp: &mut [crate::core::Vector3],
    bu: &mut [crate::core::Vector3],
    bv: &mut [crate::core::Vector3],
    bw: &mut [crate::core::Vector3],
    f: F,
) -> std::result::Result<(), E>
where
    F: Fn(
            usize,
            &mut crate::core::Vector3,
            &mut crate::core::Vector3,
            &mut crate::core::Vector3,
            &mut crate::core::Vector3,
            &mut crate::core::Vector3,
        ) -> std::result::Result<(), E>
        + Sync,
    E: Send,
{
    (
        br.par_iter_mut(),
        bp.par_iter_mut(),
        bu.par_iter_mut(),
        bv.par_iter_mut(),
        bw.par_iter_mut(),
    )
        .into_par_iter()
        .enumerate()
        .try_for_each(|(cell, (br, bp, bu, bv, bw))| f(cell, br, bp, bu, bv, bw))
}

/// 单元索引并行：四路 mut RHS + enumerate（IDWLS viscous）。
#[cfg(feature = "parallel-fvm")]
pub fn par_try_for_each_cell_rhs4<E, F>(
    bu: &mut [crate::core::Vector3],
    bv: &mut [crate::core::Vector3],
    bw: &mut [crate::core::Vector3],
    bt: &mut [crate::core::Vector3],
    f: F,
) -> std::result::Result<(), E>
where
    F: Fn(
            usize,
            &mut crate::core::Vector3,
            &mut crate::core::Vector3,
            &mut crate::core::Vector3,
            &mut crate::core::Vector3,
        ) -> std::result::Result<(), E>
        + Sync,
    E: Send,
{
    (
        bu.par_iter_mut(),
        bv.par_iter_mut(),
        bw.par_iter_mut(),
        bt.par_iter_mut(),
    )
        .into_par_iter()
        .enumerate()
        .try_for_each(|(cell, (bu, bv, bw, bt))| f(cell, bu, bv, bw, bt))
}

/// 谱半径等：单 mut slice enumerate 并行（可失败）。
#[cfg(feature = "parallel-fvm")]
pub fn par_try_for_each_enumerated<T, E, F>(items: &mut [T], f: F) -> std::result::Result<(), E>
where
    T: Send,
    F: Fn(usize, &mut T) -> std::result::Result<(), E> + Sync,
    E: Send,
{
    items
        .par_iter_mut()
        .enumerate()
        .try_for_each(|(i, item)| f(i, item))
}

/// 粘性 batch4：geoms/fluxes 按 4  chunk 与 batch_counts、static batch 并行 compute。
#[cfg(feature = "parallel-fvm")]
pub fn par_for_each_viscous_batch4_chunks<G, F, B, C>(
    geoms: &mut [G],
    fluxes: &mut [F],
    batch_counts: &mut [u8],
    batches: &[B],
    min_len: usize,
    compute: C,
) where
    G: Send,
    F: Send,
    B: Sync,
    C: Fn(&mut [G], &mut [F], &B) -> u8 + Sync,
{
    geoms
        .par_chunks_mut(4)
        .zip(fluxes.par_chunks_mut(4))
        .zip(batch_counts.par_iter_mut())
        .zip(batches.par_iter())
        .with_min_len(min_len)
        .for_each(|(((geom_chunk, flux_chunk), count), batch)| {
            *count = compute(geom_chunk, flux_chunk, batch);
        });
}

/// 桶尾余面：geom/flux/valid 与面索引并行 compute。
#[cfg(feature = "parallel-fvm")]
pub fn par_for_each_viscous_remainder<G, Fl, C>(
    geoms: &mut [G],
    fluxes: &mut [Fl],
    valid: &mut [bool],
    face_indices: &[usize],
    min_len: usize,
    compute: C,
) where
    G: Send,
    Fl: Send,
    C: Fn(usize, &mut G, &mut Fl, &mut bool) + Sync,
{
    geoms
        .par_iter_mut()
        .zip(fluxes.par_iter_mut())
        .zip(valid.par_iter_mut())
        .zip(face_indices.par_iter())
        .with_min_len(min_len)
        .for_each(|(((geom, flux), valid), &face_idx)| {
            compute(face_idx, geom, flux, valid);
        });
}

/// 非 SIMD 并行桶：全槽并行 compute。
#[cfg(feature = "parallel-fvm")]
pub fn par_for_each_viscous_face_slots<G, Fl, C>(
    geoms: &mut [G],
    fluxes: &mut [Fl],
    valid: &mut [bool],
    face_indices: &[usize],
    min_len: usize,
    compute: C,
) where
    G: Send,
    Fl: Send,
    C: Fn(usize, &mut G, &mut Fl, &mut bool) + Sync,
{
    par_for_each_viscous_remainder(geoms, fluxes, valid, face_indices, min_len, compute);
}

/// `Result` 别名包装（solver / discretization 热路径）。
#[cfg(feature = "parallel-fvm")]
pub fn par_try_for_each_enumerated_result<T, F>(items: &mut [T], f: F) -> Result<()>
where
    T: Send,
    F: Fn(usize, &mut T) -> Result<()> + Sync,
{
    par_try_for_each_enumerated(items, f)
}
