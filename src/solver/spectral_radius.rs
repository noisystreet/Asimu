//! 单元谱半径 \(\sigma_i\)（RK4 当地时间步、LU-SGS 伪时间共用）。

use crate::boundary::BoundarySet;
use crate::core::{FaceId, Real, Vector3};
use crate::discretization::BoundaryGhostBuffer;
use crate::error::{AsimuError, Result};
use crate::field::{PrimitiveFields, primitive_from_conserved_relaxed};
use crate::mesh::{BoundaryMesh3d, LogicalFace3d, StructuredMesh3d};
use crate::physics::{IdealGasEoS, PrimitiveState, ViscousPhysicsConfig};

const DEGENERATE_VOLUME: Real = 1.0e-30;

/// 3D 中心差分抛物型（粘性/热传导）对 \(\sigma\) 的贡献系数。
///
/// Blazek §6.1.4 的结构网格形式可写为
/// \(\Delta t_i=\mathrm{CFL}\,\Omega_i/(\Lambda_i^c+C\Lambda_i^v)\)；
/// 本模块内部使用 \(\sigma_i=(\Lambda_i^c+C\Lambda_i^v)/\Omega_i\)，
/// 因此粘性面贡献为 \(C\,\max(\nu,\alpha)\,A_f^2/\Omega_i^2\)。
const PARABOLIC_SPECTRAL_FACTOR_3D: Real = 6.0;

/// 谱半径求值上下文。
pub struct SpectralRadius3dParams<'a> {
    pub mesh: &'a StructuredMesh3d,
    pub boundary_mesh: &'a dyn BoundaryMesh3d,
    pub boundaries: &'a BoundarySet,
    pub ghosts: &'a BoundaryGhostBuffer,
    pub primitives: &'a PrimitiveFields,
    pub eos: &'a IdealGasEoS,
    pub min_pressure: Real,
    /// 若启用 Navier–Stokes，在双曲 \(\sigma\) 上叠加粘性/热传导抛物型项。
    pub viscous: Option<&'a ViscousPhysicsConfig>,
}

/// \(\sigma_i = \frac{1}{V_i}\sum_f (|u_n|+a)_f\,A_f + \sigma_i^v\)（内界面 + 边界面 owner 侧）。
///
/// 调用前须已刷新 `ghosts`（与 `evaluate_rhs` 一致）。
pub fn cell_spectral_radius_3d(params: &SpectralRadius3dParams<'_>) -> Result<Vec<Real>> {
    let mesh = params.mesh;
    let n = mesh.num_cells();
    if params.primitives.num_cells() != n {
        return Err(AsimuError::Solver(format!(
            "cell_spectral_radius_3d: PrimitiveFields 长度 {} 与网格 {n} 不一致",
            params.primitives.num_cells()
        )));
    }
    let gamma = params.eos.gamma;
    let mut sigma = vec![0.0; n];
    accumulate_i_face_sigma(mesh, params.primitives, gamma, &mut sigma);
    accumulate_j_face_sigma(mesh, params.primitives, gamma, &mut sigma);
    accumulate_k_face_sigma(mesh, params.primitives, gamma, &mut sigma);
    accumulate_boundary_face_sigma(params, &mut sigma)?;
    if let Some(viscous) = params.viscous {
        let diff = cell_viscous_diffusivity_max(params.primitives, params.eos, viscous)?;
        add_viscous_parabolic_sigma(params, &diff, &mut sigma)?;
    }
    for s in &mut sigma {
        *s = s.max(Real::EPSILON);
    }
    Ok(sigma)
}

/// 每单元 \(\max(\nu,\alpha)\)，\(\nu=\mu/\rho\)，\(\alpha=\mu/(\rho\,\mathrm{Pr})\)。
pub fn cell_viscous_diffusivity_max(
    primitives: &PrimitiveFields,
    eos: &IdealGasEoS,
    viscous: &ViscousPhysicsConfig,
) -> Result<Vec<Real>> {
    let n = primitives.num_cells();
    let mut diff = Vec::with_capacity(n);
    for i in 0..n {
        let rho = primitives.density.values()[i].max(1.0e-30);
        let pressure = primitives.pressure.values()[i].max(1.0e-30);
        let t_star = viscous.static_temperature(pressure, rho, eos);
        let (mu_eff, _lambda) = viscous.face_transport_coefficients(t_star, t_star, eos)?;
        let nu = mu_eff / rho;
        let alpha = mu_eff / (rho * viscous.prandtl);
        diff.push(nu.max(alpha));
    }
    Ok(diff)
}

fn add_viscous_parabolic_sigma(
    params: &SpectralRadius3dParams<'_>,
    diffusivity: &[Real],
    sigma: &mut [Real],
) -> Result<()> {
    debug_assert_eq!(sigma.len(), diffusivity.len());
    let mesh = params.mesh;
    add_i_viscous_parabolic_sigma(mesh, diffusivity, sigma);
    add_j_viscous_parabolic_sigma(mesh, diffusivity, sigma);
    add_k_viscous_parabolic_sigma(mesh, diffusivity, sigma);
    add_boundary_viscous_parabolic_sigma(params, diffusivity, sigma)
}

fn add_i_viscous_parabolic_sigma(
    mesh: &StructuredMesh3d,
    diffusivity: &[Real],
    sigma: &mut [Real],
) {
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx.saturating_sub(1) {
                let owner = mesh.cell_index(i, j, k);
                let neighbor = mesh.cell_index(i + 1, j, k);
                let face = mesh.i_face_metric(i, j, k);
                add_viscous_face_contribution(
                    sigma,
                    diffusivity,
                    owner,
                    face.area,
                    mesh.cell_metric(i, j, k).volume,
                );
                add_viscous_face_contribution(
                    sigma,
                    diffusivity,
                    neighbor,
                    face.area,
                    mesh.cell_metric(i + 1, j, k).volume,
                );
            }
        }
    }
}

fn add_j_viscous_parabolic_sigma(
    mesh: &StructuredMesh3d,
    diffusivity: &[Real],
    sigma: &mut [Real],
) {
    for k in 0..mesh.nz {
        for j in 0..mesh.ny.saturating_sub(1) {
            for i in 0..mesh.nx {
                let owner = mesh.cell_index(i, j, k);
                let neighbor = mesh.cell_index(i, j + 1, k);
                let face = mesh.j_face_metric(i, j, k);
                add_viscous_face_contribution(
                    sigma,
                    diffusivity,
                    owner,
                    face.area,
                    mesh.cell_metric(i, j, k).volume,
                );
                add_viscous_face_contribution(
                    sigma,
                    diffusivity,
                    neighbor,
                    face.area,
                    mesh.cell_metric(i, j + 1, k).volume,
                );
            }
        }
    }
}

fn add_k_viscous_parabolic_sigma(
    mesh: &StructuredMesh3d,
    diffusivity: &[Real],
    sigma: &mut [Real],
) {
    for k in 0..mesh.nz.saturating_sub(1) {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let owner = mesh.cell_index(i, j, k);
                let neighbor = mesh.cell_index(i, j, k + 1);
                let face = mesh.k_face_metric(i, j, k);
                add_viscous_face_contribution(
                    sigma,
                    diffusivity,
                    owner,
                    face.area,
                    mesh.cell_metric(i, j, k).volume,
                );
                add_viscous_face_contribution(
                    sigma,
                    diffusivity,
                    neighbor,
                    face.area,
                    mesh.cell_metric(i, j, k + 1).volume,
                );
            }
        }
    }
}

fn add_boundary_viscous_parabolic_sigma(
    params: &SpectralRadius3dParams<'_>,
    diffusivity: &[Real],
    sigma: &mut [Real],
) -> Result<()> {
    let mesh = params.mesh;
    for patch in params.boundaries.patches() {
        for &face in &patch.face_ids {
            let owner = params.boundary_mesh.face_owner(face)?.index() as usize;
            let geom = params.boundary_mesh.face_geometry_3d(face)?;
            let (logical, local) = LogicalFace3d::decode(face)?;
            let (i, j, k) = mesh.face_ij(logical, local)?;
            add_viscous_face_contribution(
                sigma,
                diffusivity,
                owner,
                geom.area,
                mesh.cell_metric(i, j, k).volume,
            );
        }
    }
    Ok(())
}

fn add_viscous_face_contribution(
    sigma: &mut [Real],
    diffusivity: &[Real],
    cell: usize,
    area: Real,
    volume: Real,
) {
    let d = diffusivity[cell];
    if d > 0.0 && area > Real::EPSILON && volume > DEGENERATE_VOLUME {
        sigma[cell] += PARABOLIC_SPECTRAL_FACTOR_3D * d * area * area / (volume * volume);
    }
}

fn accumulate_boundary_face_sigma(
    params: &SpectralRadius3dParams<'_>,
    sigma: &mut [Real],
) -> Result<()> {
    let mesh = params.mesh;
    let gamma = params.eos.gamma;
    for patch in params.boundaries.patches() {
        for &face in &patch.face_ids {
            add_boundary_face_sigma(params, face, mesh, gamma, sigma)?;
        }
    }
    Ok(())
}

fn add_boundary_face_sigma(
    params: &SpectralRadius3dParams<'_>,
    face: FaceId,
    mesh: &StructuredMesh3d,
    gamma: Real,
    sigma: &mut [Real],
) -> Result<()> {
    let owner = params.boundary_mesh.face_owner(face)?.index() as usize;
    let geom = params.boundary_mesh.face_geometry_3d(face)?;
    let ghost = params.ghosts.get_face(face).ok_or_else(|| {
        AsimuError::Boundary(format!(
            "cell_spectral_radius_3d: 边界面 FaceId({}) 缺少 ghost",
            face.index()
        ))
    })?;
    let prim_owner = params.primitives.cell_primitive(owner);
    let prim_ghost =
        primitive_from_conserved_relaxed(params.eos, &ghost.conserved, params.min_pressure)?;
    let lam = face_spectral_radius(&prim_owner, &prim_ghost, geom.normal, gamma);
    let (logical, local) = LogicalFace3d::decode(face)?;
    let (i, j, k) = mesh.face_ij(logical, local)?;
    let volume = mesh.cell_metric(i, j, k).volume;
    if volume > DEGENERATE_VOLUME {
        sigma[owner] += lam * geom.area / volume;
    }
    Ok(())
}

fn accumulate_i_face_sigma(
    mesh: &StructuredMesh3d,
    primitives: &PrimitiveFields,
    gamma: Real,
    sigma: &mut [Real],
) {
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx.saturating_sub(1) {
                let owner = mesh.cell_index(i, j, k);
                let neighbor = mesh.cell_index(i + 1, j, k);
                let face = mesh.i_face_metric(i, j, k);
                let lam = face_spectral_radius(
                    &primitives.cell_primitive(owner),
                    &primitives.cell_primitive(neighbor),
                    face.normal,
                    gamma,
                );
                add_face_contribution(
                    sigma,
                    mesh,
                    FaceSigmaContribution {
                        owner,
                        neighbor,
                        lambda: lam,
                        area: face.area,
                        owner_ijk: [i, j, k],
                        neighbor_ijk: [i + 1, j, k],
                    },
                );
            }
        }
    }
}

fn accumulate_j_face_sigma(
    mesh: &StructuredMesh3d,
    primitives: &PrimitiveFields,
    gamma: Real,
    sigma: &mut [Real],
) {
    for k in 0..mesh.nz {
        for j in 0..mesh.ny.saturating_sub(1) {
            for i in 0..mesh.nx {
                let owner = mesh.cell_index(i, j, k);
                let neighbor = mesh.cell_index(i, j + 1, k);
                let face = mesh.j_face_metric(i, j, k);
                let lam = face_spectral_radius(
                    &primitives.cell_primitive(owner),
                    &primitives.cell_primitive(neighbor),
                    face.normal,
                    gamma,
                );
                add_face_contribution(
                    sigma,
                    mesh,
                    FaceSigmaContribution {
                        owner,
                        neighbor,
                        lambda: lam,
                        area: face.area,
                        owner_ijk: [i, j, k],
                        neighbor_ijk: [i, j + 1, k],
                    },
                );
            }
        }
    }
}

fn accumulate_k_face_sigma(
    mesh: &StructuredMesh3d,
    primitives: &PrimitiveFields,
    gamma: Real,
    sigma: &mut [Real],
) {
    for k in 0..mesh.nz.saturating_sub(1) {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let owner = mesh.cell_index(i, j, k);
                let neighbor = mesh.cell_index(i, j, k + 1);
                let face = mesh.k_face_metric(i, j, k);
                let lam = face_spectral_radius(
                    &primitives.cell_primitive(owner),
                    &primitives.cell_primitive(neighbor),
                    face.normal,
                    gamma,
                );
                add_face_contribution(
                    sigma,
                    mesh,
                    FaceSigmaContribution {
                        owner,
                        neighbor,
                        lambda: lam,
                        area: face.area,
                        owner_ijk: [i, j, k],
                        neighbor_ijk: [i, j, k + 1],
                    },
                );
            }
        }
    }
}

struct FaceSigmaContribution {
    owner: usize,
    neighbor: usize,
    lambda: Real,
    area: Real,
    owner_ijk: [usize; 3],
    neighbor_ijk: [usize; 3],
}

fn add_face_contribution(sigma: &mut [Real], mesh: &StructuredMesh3d, face: FaceSigmaContribution) {
    let FaceSigmaContribution {
        owner,
        neighbor,
        lambda,
        area,
        owner_ijk: [oi, oj, ok],
        neighbor_ijk: [ni, nj, nk],
    } = face;
    let v_owner = mesh.cell_metric(oi, oj, ok).volume;
    let v_neighbor = mesh.cell_metric(ni, nj, nk).volume;
    if v_owner > DEGENERATE_VOLUME {
        sigma[owner] += lambda * area / v_owner;
    }
    if v_neighbor > DEGENERATE_VOLUME {
        sigma[neighbor] += lambda * area / v_neighbor;
    }
}

/// 内界面法向谱半径 \((|u_n|+a)_L+(|u_n|+a)_R)/2\)（扫掠耦合用）。
pub fn face_spectral_radius(
    prim_l: &PrimitiveState,
    prim_r: &PrimitiveState,
    normal: Vector3,
    gamma: Real,
) -> Real {
    let lam_l = normal_speed_plus_sound(prim_l, normal, gamma);
    let lam_r = normal_speed_plus_sound(prim_r, normal, gamma);
    0.5 * (lam_l + lam_r)
}

fn normal_speed_plus_sound(prim: &PrimitiveState, normal: Vector3, gamma: Real) -> Real {
    let rho = prim.density.max(1.0e-30);
    let u_n =
        prim.velocity[0] * normal.x + prim.velocity[1] * normal.y + prim.velocity[2] * normal.z;
    let a = (gamma * prim.pressure.max(1.0e-30) / rho).sqrt();
    u_n.abs() + a
}

/// Blazek 局部时间步（第 6.1.4/9.1 节）：\(\Delta t_i = \mathrm{CFL}/\sigma_i\)，\(\sigma_i=(\Lambda_i^c+C\Lambda_i^v)/V_i\)。
///
/// 无粘时等价于 \(\Delta t_i = \mathrm{CFL}\,V_i / \sum_f \lambda_f A_f\)。
/// 显式 RK、LU-SGS 伪时间共用；\(\lambda_f \approx \frac{1}{2}(|u_n|+a)_L+\frac{1}{2}(|u_n|+a)_R\)，
/// Navier-Stokes 还叠加 `cell_spectral_radius_3d` 中的粘性 face-sum 项。
pub fn cell_local_dt_spectral(volumes: &[Real], sigma: &[Real], cfl: Real) -> Result<Vec<Real>> {
    if volumes.len() != sigma.len() {
        return Err(AsimuError::Solver(
            "cell_local_dt_spectral: volumes 与 sigma 长度不一致".to_string(),
        ));
    }
    if cfl <= 0.0 {
        return Err(AsimuError::Solver(
            "cell_local_dt_spectral: CFL 须为正".to_string(),
        ));
    }
    let mut dt = Vec::with_capacity(volumes.len());
    for (&v, &s) in volumes.iter().zip(sigma.iter()) {
        if v <= 0.0 || s <= 0.0 {
            return Err(AsimuError::Solver(
                "cell_local_dt_spectral: 体积与谱半径须为正".to_string(),
            ));
        }
        dt.push(cfl / s);
    }
    Ok(dt)
}

/// 3D 逐单元局部时间步：先算 \(\sigma_i\)，再按 Blazek 公式得 \(\Delta t_i\)。
pub fn cell_local_dt_cfl_3d(params: &SpectralRadius3dParams<'_>, cfl: Real) -> Result<Vec<Real>> {
    let volumes = params.mesh.cell_volumes();
    let sigma = cell_spectral_radius_3d(params)?;
    cell_local_dt_spectral(&volumes, &sigma, cfl)
}

/// 兼容旧名（LU-SGS 伪时间步）。
pub fn local_pseudo_dt_lusgs(volumes: &[Real], sigma: &[Real], cfl: Real) -> Result<Vec<Real>> {
    cell_local_dt_spectral(volumes, sigma, cfl)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};
    use crate::discretization::freestream_pair::{FreestreamPairFixture, UniformFarfieldSide};
    use crate::discretization::{BoundaryGhostBuffer, apply_compressible_boundary_conditions};
    use crate::field::{ConservedFields, PrimitiveFields};
    use crate::mesh::{BoundaryMesh, StructuredMesh3d};
    use crate::physics::{FreestreamContext, FreestreamParams};

    fn uniform_box_sigma(
        boundary_patches: Vec<BoundaryPatch>,
        side: &UniformFarfieldSide<'_>,
    ) -> Vec<Real> {
        let mesh = StructuredMesh3d::uniform_box("box", 3, 3, 3, 1.0, 1.0, 1.0).expect("mesh");
        let fields = ConservedFields::from_freestream_context(mesh.num_cells(), &side.ctx, side.fs)
            .expect("fields");
        let boundary_set = BoundarySet::new(boundary_patches);
        let mut ghosts = BoundaryGhostBuffer::new();
        apply_compressible_boundary_conditions(
            &mesh,
            &boundary_set,
            &fields,
            &mut ghosts,
            &side.ctx,
            side.fs,
            side.viscous,
        )
        .expect("bc");
        let mut primitives = PrimitiveFields::zeros(mesh.num_cells()).expect("prim");
        primitives
            .fill_from_conserved(&fields, side.eos, side.min_pressure)
            .expect("fill");
        cell_spectral_radius_3d(&SpectralRadius3dParams {
            mesh: &mesh,
            boundary_mesh: &mesh,
            boundaries: &boundary_set,
            ghosts: &ghosts,
            primitives: &primitives,
            eos: side.eos,
            min_pressure: side.min_pressure,
            viscous: side.viscous,
        })
        .expect("sigma")
    }

    #[test]
    fn boundary_faces_increase_owner_spectral_radius() {
        let pair = FreestreamPairFixture::air_sutherland(0.2);
        let side = pair.inviscid_side();
        let mesh = StructuredMesh3d::uniform_box("box", 3, 3, 3, 1.0, 1.0, 1.0).expect("mesh");
        let faces = mesh.resolve_logical_boundary("i_min").expect("i_min");
        let farfield_patches = vec![BoundaryPatch::new(
            "i_min",
            faces,
            BoundaryKind::Farfield {
                mach: side.fs.mach,
                pressure: side.fs.pressure,
                temperature: side.fs.temperature,
                alpha: 0.0,
                beta: 0.0,
            },
        )];
        let sigma_boundary = uniform_box_sigma(farfield_patches, &side);
        let sigma_interior_only = uniform_box_sigma(Vec::new(), &side);
        let imin_owner = mesh.cell_index(0, 0, 0);
        assert!(sigma_boundary[imin_owner] > sigma_interior_only[imin_owner]);
    }

    /// 圆柱 CGNS：`--nocapture` 打印 \(\sigma\)、\(V\)、\(\Delta t\) 分布及最小 dt 单元。
    #[cfg(feature = "io-cgns")]
    #[test]
    fn cylinder_mach8_timestep_diagnostic_when_present() {
        use std::path::PathBuf;

        use crate::io::{CaseMesh, load_case};

        let case_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("case_cylinder/case.toml");
        if !case_path.is_file() {
            return;
        }
        let case = load_case(&case_path).expect("load case");
        let CaseMesh::Structured3d(mesh) = &case.mesh else {
            panic!("expected 3d mesh");
        };
        let eos = case.physics.eos().expect("eos");
        let fs = case.freestream.expect("freestream");
        let cfl = case.time.cfl.expect("cfl");
        let fields = ConservedFields::from_freestream(mesh.num_cells(), &eos, &fs).expect("fields");
        let mut ghosts = BoundaryGhostBuffer::new();
        let fs_ctx = FreestreamContext::new(&eos, case.reference.as_ref(), None);
        apply_compressible_boundary_conditions(
            mesh,
            &case.boundary,
            &fields,
            &mut ghosts,
            &fs_ctx,
            &fs,
            None,
        )
        .expect("bc");
        let mut primitives = PrimitiveFields::zeros(mesh.num_cells()).expect("prim");
        let p_floor = crate::field::positivity_pressure_floor(fs.pressure);
        primitives
            .fill_from_conserved(&fields, &eos, p_floor)
            .expect("fill");
        let params = SpectralRadius3dParams {
            mesh,
            boundary_mesh: mesh,
            boundaries: &case.boundary,
            ghosts: &ghosts,
            primitives: &primitives,
            eos: &eos,
            min_pressure: p_floor,
            viscous: None,
        };
        let sigma = cell_spectral_radius_3d(&params).expect("sigma");
        let volumes = mesh.cell_volumes();
        let dts = cell_local_dt_spectral(&volumes, &sigma, cfl).expect("dts");
        let lengths = mesh.cell_cfl_lengths().expect("lengths");
        let gamma = eos.gamma;
        let n = mesh.num_cells();
        let mut h_over_lam = Vec::with_capacity(n);
        for (i, length) in lengths.iter().enumerate().take(n) {
            let prim = primitives.cell_primitive(i);
            let rho = prim.density.max(1.0e-30);
            let u_mag =
                (prim.velocity[0].powi(2) + prim.velocity[1].powi(2) + prim.velocity[2].powi(2))
                    .sqrt();
            let speed = u_mag + (gamma * prim.pressure.max(1.0e-30) / rho).sqrt();
            h_over_lam.push(*length / speed.max(1.0e-30));
        }
        let dts_h = h_over_lam.iter().map(|&hl| cfl * hl).collect::<Vec<_>>();
        let min_idx = dts
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .expect("min dt");
        let pct = |v: &[Real], p: f64| {
            let mut s = v.to_vec();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            s[(p * (s.len() as f64 - 1.0)).round() as usize]
        };
        eprintln!("=== cylinder dt 诊断 (cfl={cfl}) ===");
        eprintln!(
            "dt[spectral]: min={:.6e} p50={:.6e} p99={:.6e} max={:.6e}",
            dts.iter().copied().fold(f64::INFINITY, f64::min),
            pct(&dts, 0.5),
            pct(&dts, 0.99),
            dts.iter().copied().fold(0.0_f64, f64::max),
        );
        eprintln!(
            "dt[length estimate, comparison only]: min={:.6e} p50={:.6e} p99={:.6e} max={:.6e}",
            dts_h.iter().copied().fold(f64::INFINITY, f64::min),
            pct(&dts_h, 0.5),
            pct(&dts_h, 0.99),
            dts_h.iter().copied().fold(0.0_f64, f64::max),
        );
        eprintln!(
            "min-dt cell idx={min_idx}: V={:.6e} sigma={:.6e} h_min={:.6e} V/sigma={:.6e}",
            volumes[min_idx],
            sigma[min_idx],
            lengths[min_idx],
            volumes[min_idx] / sigma[min_idx],
        );
        assert!(
            dts[min_idx] > 0.0 && dts[min_idx].is_finite(),
            "min dt must be positive finite"
        );
    }

    #[test]
    fn viscous_diffusion_increases_unified_spectral_radius_and_tightens_dt() {
        use crate::physics::{ViscosityModel, ViscousPhysicsConfig};

        let mesh = StructuredMesh3d::uniform_box("box", 2, 2, 2, 1.0, 1.0, 1.0).expect("mesh");
        let eos = IdealGasEoS::AIR_STANDARD;
        let fs = FreestreamParams::default();
        let fields = ConservedFields::from_freestream(mesh.num_cells(), &eos, &fs).expect("fields");
        let boundary_set = BoundarySet::new(Vec::new());
        let ghosts = BoundaryGhostBuffer::new();
        let mut primitives = PrimitiveFields::zeros(mesh.num_cells()).expect("prim");
        primitives
            .fill_from_conserved(&fields, &eos, 1.0e-6)
            .expect("fill");
        let cfl = 0.5;
        let base = SpectralRadius3dParams {
            mesh: &mesh,
            boundary_mesh: &mesh,
            boundaries: &boundary_set,
            ghosts: &ghosts,
            primitives: &primitives,
            eos: &eos,
            min_pressure: 1.0e-6,
            viscous: None,
        };
        let sigma_inv = cell_spectral_radius_3d(&base).expect("sigma inv");
        let volumes = mesh.cell_volumes();
        let dt_inv = cell_local_dt_spectral(&volumes, &sigma_inv, cfl).expect("dt inv");
        let viscous = ViscousPhysicsConfig::new(ViscosityModel::constant(1.0).expect("mu"), 0.72)
            .expect("visc cfg");
        let sigma_visc = cell_spectral_radius_3d(&SpectralRadius3dParams {
            viscous: Some(&viscous),
            ..base
        })
        .expect("sigma visc");
        let dt_visc = cell_local_dt_spectral(&volumes, &sigma_visc, cfl).expect("dt visc");
        for i in 0..mesh.num_cells() {
            assert!(
                sigma_visc[i] > sigma_inv[i],
                "cell {i}: viscous sigma should exceed inviscid"
            );
            assert!(
                dt_visc[i] < dt_inv[i],
                "cell {i}: viscous dt should be smaller"
            );
        }
    }

    #[test]
    fn viscous_volume_spectral_radius_exceeds_inviscid_on_uniform_box() {
        use crate::physics::{ViscosityModel, ViscousPhysicsConfig};

        let mesh = StructuredMesh3d::uniform_box("box", 2, 2, 2, 1.0, 1.0, 1.0).expect("mesh");
        let eos = IdealGasEoS::AIR_STANDARD;
        let fs = FreestreamParams::default();
        let fields = ConservedFields::from_freestream(mesh.num_cells(), &eos, &fs).expect("fields");
        let boundary_set = BoundarySet::new(Vec::new());
        let mut ghosts = BoundaryGhostBuffer::new();
        let fs_ctx = FreestreamContext::new(&eos, None, None);
        apply_compressible_boundary_conditions(
            &mesh,
            &boundary_set,
            &fields,
            &mut ghosts,
            &fs_ctx,
            &fs,
            None,
        )
        .expect("bc");
        let mut primitives = PrimitiveFields::zeros(mesh.num_cells()).expect("prim");
        primitives
            .fill_from_conserved(&fields, &eos, 1.0e-6)
            .expect("fill");
        let base = SpectralRadius3dParams {
            mesh: &mesh,
            boundary_mesh: &mesh,
            boundaries: &boundary_set,
            ghosts: &ghosts,
            primitives: &primitives,
            eos: &eos,
            min_pressure: 1.0e-6,
            viscous: None,
        };
        let sigma_inv = cell_spectral_radius_3d(&base).expect("sigma");
        let viscous = ViscousPhysicsConfig::new(ViscosityModel::constant(0.1).expect("mu"), 0.72)
            .expect("visc");
        let sigma_visc = cell_spectral_radius_3d(&SpectralRadius3dParams {
            viscous: Some(&viscous),
            ..base
        })
        .expect("sigma visc");
        assert!(
            sigma_visc
                .iter()
                .zip(sigma_inv.iter())
                .all(|(a, b)| *a > *b)
        );
    }

    #[test]
    fn spectral_radius_positive_on_uniform_freestream() {
        let pair = FreestreamPairFixture::air_sutherland(0.2);
        pair.for_each_inviscid_side(|side| {
            let mesh = StructuredMesh3d::uniform_box("box", 2, 2, 2, 1.0, 1.0, 1.0).expect("mesh");
            let mut patches = Vec::new();
            for name in ["i_min", "i_max", "j_min", "j_max", "k_min", "k_max"] {
                let faces = mesh.resolve_logical_boundary(name).expect("faces");
                patches.push(BoundaryPatch::new(
                    name,
                    faces,
                    BoundaryKind::Farfield {
                        mach: side.fs.mach,
                        pressure: side.fs.pressure,
                        temperature: side.fs.temperature,
                        alpha: 0.0,
                        beta: 0.0,
                    },
                ));
            }
            let sigma = uniform_box_sigma(patches, side);
            assert!(
                sigma.iter().all(|&s| s.is_finite() && s > 0.0),
                "{} spectral radius",
                side.label
            );
        });
    }
}
