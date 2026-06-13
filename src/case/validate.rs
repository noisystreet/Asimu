//! 算例编排层语义校验（run 阶段；多块 parse 校验见 `io::case_validate`）。

use std::collections::HashSet;

use tracing::warn;

use crate::boundary::BoundarySet;
use crate::core::{FaceId, Real};
use crate::discretization::ReconstructionKind;
use crate::error::{AsimuError, Result};
use crate::io::CaseSpec;
use crate::mesh::UnstructuredMesh3d;
use crate::solver::TimeIntegrationScheme;

/// 非结构可压缩离散与时间格式约束。
pub fn unstructured_compressible(case: &CaseSpec) -> Result<()> {
    let disc = case.compressible_discretization()?;
    let inviscid = disc.inviscid();
    match inviscid.reconstruction {
        ReconstructionKind::FirstOrder => {}
        ReconstructionKind::Muscl => {
            if inviscid.unstructured_gradient_limiter.is_none() {
                if disc.limiter.is_some() {
                    return Err(AsimuError::Config(
                        "非结构二阶线性重构须设置 unstructured_limiter = barth_jespersen | venkatakrishnan；\
                         结构化 limiter（minmod/van_leer/van_albada）不可在非结构 case 中复用（见 ADR 0012）"
                            .to_string(),
                    ));
                }
                return Err(AsimuError::Config(
                    "非结构二阶线性重构须设置 unstructured_limiter = barth_jespersen | venkatakrishnan"
                        .to_string(),
                ));
            }
            if disc.limiter.is_some() {
                warn!(
                    limiter = ?disc.limiter,
                    unstructured_limiter = ?disc.unstructured_limiter,
                    "非结构二阶线性重构忽略 [euler].limiter，使用 unstructured_limiter"
                );
            }
            if let Some(name) = disc.unstructured_limiter.as_deref() {
                if crate::discretization::UnstructuredGradientLimiter::parse(name).is_none() {
                    return Err(AsimuError::Config(format!(
                        "未知 unstructured_limiter \"{name}\"；可选 barth_jespersen | venkatakrishnan"
                    )));
                }
            }
        }
    }
    if case.time.residual_smoothing_config().enabled {
        warn!("非结构网格暂不支持结构化方向分裂残差光顺；本次忽略 residual_smoothing");
    }
    if case.time.resolved_time_scheme() == TimeIntegrationScheme::Gmres {
        return Err(AsimuError::Config(
            "非结构网格暂不支持 time.scheme = \"gmres\"".to_string(),
        ));
    }
    Ok(())
}

/// 非结构边界面须被 patch 完整覆盖且无内部面引用。
pub fn unstructured_boundary_coverage(
    mesh: &UnstructuredMesh3d,
    boundary: &BoundarySet,
) -> Result<()> {
    let mut covered = HashSet::new();
    for patch in boundary.patches() {
        for &face in &patch.face_ids {
            if mesh.face_neighbor(face)?.is_some() {
                return Err(AsimuError::Boundary(format!(
                    "非结构边界 patch {} 引用了内部面 FaceId({})",
                    patch.name,
                    face.index()
                )));
            }
            covered.insert(face.index());
        }
    }
    let mut boundary_faces = 0usize;
    for face in 0..mesh.num_faces() {
        if mesh.face_neighbor(FaceId(face as u32))?.is_none() {
            boundary_faces += 1;
        }
    }
    if covered.len() != boundary_faces {
        return Err(AsimuError::Boundary(format!(
            "非结构边界 patch 覆盖 {}/{} 个边界面，求解前须完整覆盖",
            covered.len(),
            boundary_faces
        )));
    }
    Ok(())
}

/// log₁₀(RMS(ρ̇)) 早停容差（`[time].tolerance`）。
#[must_use]
pub fn residual_tolerance(case: &CaseSpec) -> Option<Real> {
    case.resolved_tolerance()
}
