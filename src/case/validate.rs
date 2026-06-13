//! 算例编排层语义校验（run 阶段；多块 parse 校验见 `io::case_validate`）。

use std::collections::HashSet;

use tracing::warn;

use crate::boundary::BoundarySet;
use crate::core::{ComputePrecision, ExecDevice, FaceId, Real};
use crate::discretization::ReconstructionKind;
use crate::error::{AsimuError, Result};
use crate::io::{CaseMesh, CaseSpec};
use crate::mesh::UnstructuredMesh3d;
use crate::solver::TimeIntegrationScheme;

/// 核心计算精度与当前 solver 能力是否匹配（ADR 0016 P3）。
pub fn compute_precision(case: &CaseSpec) -> Result<()> {
    if case.numerics.compute_precision == ComputePrecision::F64 {
        return Ok(());
    }
    validate_f32_capabilities(case)
}

/// `[numerics].backend` 与编译 feature / 求解器能力是否匹配（ADR 0017 G0）。
pub fn exec_backend(case: &CaseSpec) -> Result<()> {
    match case.numerics.exec_device {
        ExecDevice::Cpu => Ok(()),
        ExecDevice::GpuCuda => validate_gpu_cuda_backend(case),
    }
}

fn validate_gpu_cuda_backend(case: &CaseSpec) -> Result<()> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = case;
        Err(AsimuError::Config(
            "backend = \"cuda\" 需要以 Cargo feature cuda 编译 asimu".to_string(),
        ))
    }
    #[cfg(feature = "cuda")]
    {
        validate_gpu_cuda_backend_enabled(case)
    }
}

#[cfg(feature = "cuda")]
fn validate_gpu_cuda_backend_enabled(case: &CaseSpec) -> Result<()> {
    if case.numerics.compute_precision != ComputePrecision::F32 {
        return Err(gpu_cuda_unsupported(
            "cuda 首版仅支持 compute_precision = \"f32\"",
        ));
    }
    if !matches!(case.mesh, CaseMesh::Unstructured3d(_)) {
        return Err(gpu_cuda_unsupported("cuda 首版仅支持非结构 3D 可压缩路径"));
    }
    if !case.is_compressible() {
        return Err(gpu_cuda_unsupported("cuda 仅支持可压缩求解路径"));
    }
    let disc = case.compressible_discretization()?;
    if disc.inviscid().reconstruction != ReconstructionKind::FirstOrder {
        return Err(gpu_cuda_unsupported(
            "cuda 首版（G1）仅支持 reconstruction = first_order",
        ));
    }
    if case.physics.viscous.is_some() || case.navier_stokes.is_some() {
        return Err(gpu_cuda_unsupported("cuda 首版不支持粘性 / Navier-Stokes"));
    }
    match case.time.resolved_time_scheme() {
        TimeIntegrationScheme::Rk4 | TimeIntegrationScheme::Euler => {}
        scheme => {
            return Err(gpu_cuda_unsupported(&format!(
                "cuda 首版不支持 time.scheme = \"{}\"",
                scheme.label()
            )));
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn gpu_cuda_unsupported(detail: &str) -> AsimuError {
    AsimuError::Config(format!("backend = \"cuda\"：{detail}"))
}

fn validate_f32_capabilities(case: &CaseSpec) -> Result<()> {
    if case.numerics.compute_precision != ComputePrecision::F32 {
        return Ok(());
    }
    if case.is_compressible() {
        let supported_3d = matches!(
            case.mesh,
            CaseMesh::MultiBlockStructured3d(_) | CaseMesh::Unstructured3d(_)
        );
        if !supported_3d {
            return Err(f32_unsupported(
                "仅 3D 可压缩 structured/unstructured 路径支持 f32",
            ));
        }
    } else {
        return Err(f32_unsupported("仅 3D 可压缩 Euler 路径支持 f32"));
    }
    if case.navier_stokes.is_some() && !matches!(case.mesh, CaseMesh::Unstructured3d(_)) {
        return Err(f32_unsupported("Navier-Stokes f32 暂仅支持非结构 3D 路径"));
    }
    if case.physics.viscous.is_some() && !matches!(case.mesh, CaseMesh::Unstructured3d(_)) {
        return Err(f32_unsupported("粘性通量 f32 暂仅支持非结构 3D 路径"));
    }
    validate_f32_discretization(case)?;
    validate_f32_time_scheme(case)?;
    if case.time.residual_smoothing_config().enabled {
        return Err(f32_unsupported("f32 暂不支持 residual_smoothing"));
    }
    if let CaseMesh::MultiBlockStructured3d(mesh) = &case.mesh {
        if !mesh.interfaces().is_empty() {
            return Err(f32_unsupported("f32 暂不支持多块 1-to-1 接口通量"));
        }
    }
    Ok(())
}

fn validate_f32_discretization(case: &CaseSpec) -> Result<()> {
    let disc = case.compressible_discretization()?;
    match disc.inviscid().reconstruction {
        ReconstructionKind::FirstOrder => Ok(()),
        ReconstructionKind::Muscl => {
            if !matches!(case.mesh, CaseMesh::Unstructured3d(_)) {
                return Err(f32_unsupported("f32 二阶 MUSCL 暂仅支持非结构 3D 路径"));
            }
            if disc.inviscid().unstructured_gradient_limiter.is_none() {
                return Err(f32_unsupported(
                    "f32 非结构 MUSCL 须设置 unstructured_limiter = barth_jespersen | venkatakrishnan",
                ));
            }
            Ok(())
        }
    }
}

fn validate_f32_time_scheme(case: &CaseSpec) -> Result<()> {
    match case.time.resolved_time_scheme() {
        TimeIntegrationScheme::Rk4 | TimeIntegrationScheme::Euler => Ok(()),
        TimeIntegrationScheme::LuSgs => validate_f32_lusgs_time(case),
        TimeIntegrationScheme::Gmres => validate_f32_gmres_time(case),
        scheme => Err(f32_unsupported(&format!(
            "f32 暂不支持 time.scheme = \"{}\"",
            scheme.label()
        ))),
    }
}

fn validate_f32_lusgs_time(case: &CaseSpec) -> Result<()> {
    if !case.time.uses_local_time_step() {
        return Err(f32_unsupported("f32 lu_sgs 须配合 local_time_step = true"));
    }
    let lu_sgs = case.time.resolved_lusgs_config()?;
    if lu_sgs.sweep {
        return Err(f32_unsupported("f32 暂不支持 lusgs_sweep = true"));
    }
    Ok(())
}

fn validate_f32_gmres_time(case: &CaseSpec) -> Result<()> {
    if !case.time.uses_local_time_step() {
        return Err(f32_unsupported("f32 gmres 须配合 local_time_step = true"));
    }
    if matches!(case.mesh, CaseMesh::Unstructured3d(_)) {
        return Err(f32_unsupported("f32 gmres 暂仅支持结构化 3D 路径"));
    }
    if case.physics.viscous.is_some() {
        return Err(f32_unsupported("f32 gmres 暂不支持粘性通量"));
    }
    Ok(())
}

fn f32_unsupported(detail: &str) -> AsimuError {
    AsimuError::Config(format!("compute_precision = \"f32\"：{detail}"))
}

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

#[cfg(test)]
mod compute_precision_tests {
    use super::*;
    use std::path::Path;

    use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};
    use crate::core::ComputePrecision;
    use crate::io::{CaseNumericsConfig, load_case};
    use crate::mesh::{CellKind, UnstructuredCell, UnstructuredMesh3d};

    fn attach_single_tet_farfield(case: &mut CaseSpec) {
        let mesh = UnstructuredMesh3d::new(
            "tet",
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            vec![UnstructuredCell::new(CellKind::Tet, vec![0, 1, 2, 3]).expect("cell")],
        )
        .expect("mesh");
        let faces = (0..mesh.num_faces())
            .map(|face| FaceId(face as u32))
            .collect::<Vec<_>>();
        let fs = case.freestream.expect("freestream");
        case.mesh = CaseMesh::Unstructured3d(mesh);
        case.boundary = BoundarySet::new(vec![BoundaryPatch::new(
            "farfield",
            faces,
            BoundaryKind::Farfield {
                mach: fs.mach,
                pressure: fs.pressure,
                temperature: fs.temperature,
                alpha: fs.alpha,
                beta: fs.beta,
            },
        )]);
    }

    #[test]
    fn f64_passes_validate() {
        let case = load_case(Path::new(
            "tests/benchmarks/1d_diffusion_analytical/case.toml",
        ))
        .expect("case");
        assert_eq!(case.numerics.compute_precision, ComputePrecision::F64);
        compute_precision(&case).expect("f64");
    }

    #[test]
    fn f32_rejected_for_unsupported_paths() {
        let mut case = load_case(Path::new(
            "tests/benchmarks/1d_diffusion_analytical/case.toml",
        ))
        .expect("case");
        case.numerics = CaseNumericsConfig {
            compute_precision: ComputePrecision::F32,
            ..CaseNumericsConfig::default()
        };
        let err = compute_precision(&case).expect_err("f32 diffusion");
        assert!(err.to_string().contains("f32"));
    }

    #[test]
    fn f32_accepts_unstructured_first_order_case() {
        let mut case = load_case(Path::new(
            "tests/benchmarks/unstructured_freestream/case.toml",
        ))
        .expect("case");
        case.numerics = CaseNumericsConfig {
            compute_precision: ComputePrecision::F32,
            ..CaseNumericsConfig::default()
        };
        compute_precision(&case).expect("unstructured freestream f32");
    }

    #[test]
    fn f32_accepts_unstructured_muscl_case() {
        let mut case = load_case(Path::new(
            "tests/benchmarks/unstructured_freestream/case.toml",
        ))
        .expect("case");
        case.numerics = CaseNumericsConfig {
            compute_precision: ComputePrecision::F32,
            ..CaseNumericsConfig::default()
        };
        attach_single_tet_farfield(&mut case);
        if let Some(euler) = case.euler.as_mut() {
            euler.reconstruction = Some("muscl".to_string());
            euler.unstructured_limiter = Some("barth_jespersen".to_string());
        }
        compute_precision(&case).expect("unstructured muscl f32");
    }

    #[test]
    fn f32_rejects_structured_muscl_case() {
        let mut case = load_case(Path::new(
            "tests/benchmarks/unstructured_freestream/case.toml",
        ))
        .expect("case");
        case.numerics = CaseNumericsConfig {
            compute_precision: ComputePrecision::F32,
            ..CaseNumericsConfig::default()
        };
        let block_mesh = crate::mesh::StructuredMesh3d::uniform_box("box", 2, 2, 2, 1.0, 1.0, 1.0)
            .expect("mesh");
        case.mesh = CaseMesh::MultiBlockStructured3d(
            crate::mesh::MultiBlockStructuredMesh3d::from_single_mesh(block_mesh).expect("mb"),
        );
        if let Some(euler) = case.euler.as_mut() {
            euler.reconstruction = Some("muscl".to_string());
            euler.unstructured_limiter = Some("barth_jespersen".to_string());
        }
        let err = compute_precision(&case).expect_err("structured muscl f32");
        assert!(err.to_string().contains("MUSCL"));
    }

    #[test]
    fn f32_rejects_structured_viscous_case() {
        let mut case = load_case(Path::new(
            "tests/benchmarks/unstructured_freestream/case.toml",
        ))
        .expect("case");
        case.numerics = CaseNumericsConfig {
            compute_precision: ComputePrecision::F32,
            ..CaseNumericsConfig::default()
        };
        let block_mesh = crate::mesh::StructuredMesh3d::uniform_box("box", 2, 2, 2, 1.0, 1.0, 1.0)
            .expect("mesh");
        case.mesh = CaseMesh::MultiBlockStructured3d(
            crate::mesh::MultiBlockStructuredMesh3d::from_single_mesh(block_mesh).expect("mb"),
        );
        case.physics.viscous = Some(crate::physics::ViscousPhysicsConfig::default());
        let err = compute_precision(&case).expect_err("structured viscous f32");
        assert!(err.to_string().contains("非结构"));
    }

    #[test]
    fn f32_accepts_unstructured_navier_stokes_case() {
        let mut case = load_case(Path::new(
            "tests/benchmarks/unstructured_freestream/case.toml",
        ))
        .expect("case");
        case.numerics = CaseNumericsConfig {
            compute_precision: ComputePrecision::F32,
            ..CaseNumericsConfig::default()
        };
        attach_single_tet_farfield(&mut case);
        case.navier_stokes = case.euler.take();
        case.physics.viscous = Some(crate::physics::ViscousPhysicsConfig::default());
        compute_precision(&case).expect("unstructured navier_stokes f32");
    }

    #[test]
    fn f32_accepts_structured_lusgs_case() {
        let mut case = load_case(Path::new(
            "tests/benchmarks/unstructured_freestream/case.toml",
        ))
        .expect("case");
        case.numerics = CaseNumericsConfig {
            compute_precision: ComputePrecision::F32,
            ..CaseNumericsConfig::default()
        };
        let block_mesh = crate::mesh::StructuredMesh3d::uniform_box("box", 2, 2, 2, 1.0, 1.0, 1.0)
            .expect("mesh");
        case.mesh = CaseMesh::MultiBlockStructured3d(
            crate::mesh::MultiBlockStructuredMesh3d::from_single_mesh(block_mesh).expect("mb"),
        );
        case.time.scheme = Some(TimeIntegrationScheme::LuSgs);
        case.time.local_time_step = true;
        compute_precision(&case).expect("structured lusgs f32");
    }
}

#[cfg(test)]
mod exec_backend_tests {
    use super::*;
    use std::path::Path;

    use crate::core::{ComputePrecision, ExecDevice};
    use crate::io::{CaseNumericsConfig, load_case};

    #[test]
    fn cpu_backend_always_valid() {
        let case = load_case(Path::new(
            "tests/benchmarks/1d_diffusion_analytical/case.toml",
        ))
        .expect("case");
        exec_backend(&case).expect("cpu default");
    }

    #[test]
    fn gpu_cuda_rejected_without_feature() {
        let mut case = load_case(Path::new(
            "tests/benchmarks/unstructured_freestream/case.toml",
        ))
        .expect("case");
        case.numerics = CaseNumericsConfig {
            compute_precision: ComputePrecision::F32,
            exec_device: ExecDevice::GpuCuda,
        };
        #[cfg(not(feature = "cuda"))]
        {
            let err = exec_backend(&case).expect_err("no feature");
            assert!(err.to_string().contains("cuda"));
        }
        #[cfg(feature = "cuda")]
        {
            exec_backend(&case).expect("cuda feature enabled");
        }
    }

    #[test]
    fn gpu_cuda_rejects_f64_precision() {
        let mut case = load_case(Path::new(
            "tests/benchmarks/unstructured_freestream/case.toml",
        ))
        .expect("case");
        case.numerics = CaseNumericsConfig {
            compute_precision: ComputePrecision::F64,
            exec_device: ExecDevice::GpuCuda,
        };
        #[cfg(feature = "cuda")]
        {
            let err = exec_backend(&case).expect_err("f64");
            assert!(err.to_string().contains("f32"));
        }
    }
}
