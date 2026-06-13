//! 算例语义校验（Parse → **Validate** → Build 之 Validate 阶段，可压多块）。

use crate::error::{AsimuError, Result};
use crate::io::{CaseMesh, CaseSpec};
use crate::solver::TimeIntegrationScheme;

/// 多块 1-to-1 接口的可压缩算例约束（LU-SGS 对角隐式）。
pub fn multiblock_compressible(case: &CaseSpec) -> Result<()> {
    if !case.is_compressible() || (case.euler.is_none() && case.navier_stokes.is_none()) {
        return Ok(());
    }
    let CaseMesh::MultiBlockStructured3d(mesh) = &case.mesh else {
        return Ok(());
    };
    if mesh.interfaces().is_empty() {
        return Ok(());
    }
    if case.time.resolved_time_scheme() != TimeIntegrationScheme::LuSgs {
        return Err(AsimuError::Config(
            "有 1-to-1 接口的多块 3D 可压缩算例当前仅支持 time.scheme = \"lu_sgs\"".to_string(),
        ));
    }
    if case.time.resolved_lusgs_config()?.sweep {
        return Err(AsimuError::Config(
            "有 1-to-1 接口的多块 3D 可压缩算例暂不支持 lusgs_sweep = true".to_string(),
        ));
    }
    Ok(())
}
