//! Taylor–Green 涡初场与动能衰减 V&V（ADR 0015 I3）。
//!
//! 理论：[`docs/theory/incompressible_simplec_piso.md`](../../../docs/theory/incompressible_simplec_piso.md)

use std::f64::consts::PI;

use tracing::info;

use crate::boundary::BoundarySet;
use crate::core::{Real, format_log_sci4};
use crate::discretization::{
    IncompressibleFaceFluxField, IncompressibleMomentumPredictorConfig,
    assemble_incompressible_momentum_predictor_with_boundary_and_flux_3d,
};
use crate::error::Result;
use crate::field::{IncompressibleFields, ScalarField};
use crate::io::IncompressibleCaseConfig;
use crate::mesh::StructuredMesh3d;
use crate::solver::{
    IncompressibleProjectionConfig, project_incompressible_fields_divergence_free_with_d_3d,
};

const TWO_PI: Real = 2.0 * PI;

/// 装配 Taylor–Green 初场（结构化域，坐标已无量纲化至 \([0,1]^d\)）。
pub fn taylor_green_initial_fields(mesh: &StructuredMesh3d) -> Result<IncompressibleFields> {
    let n = mesh.num_cells();
    let mut pressure = Vec::with_capacity(n);
    let mut ux = Vec::with_capacity(n);
    let mut uy = Vec::with_capacity(n);
    let mut uz = Vec::with_capacity(n);
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let center = mesh.cell_metric(i, j, k).center;
                let x = TWO_PI * center.x;
                let y = TWO_PI * center.y;
                let z = TWO_PI * center.z;
                let cos_z = z.cos();
                ux.push(x.sin() * y.cos() * cos_z);
                uy.push(-x.cos() * y.sin() * cos_z);
                uz.push(0.0);
                let cos_2x = (2.0 * x).cos();
                let cos_2y = (2.0 * y).cos();
                let cos_2z = (2.0 * z).cos();
                let p = -(cos_2x + cos_2y) * (cos_2z + 2.0) / 16.0;
                pressure.push(p);
            }
        }
    }
    Ok(IncompressibleFields {
        pressure: ScalarField::from_values(pressure)?,
        velocity_x: ScalarField::from_values(ux)?,
        velocity_y: ScalarField::from_values(uy)?,
        velocity_z: ScalarField::from_values(uz)?,
    })
}

/// Taylor–Green 初场预处理结果：Rhie-Chow 压力投影后的场与一致面通量。
#[derive(Debug, Clone, PartialEq)]
pub struct TaylorGreenPreparedInitial {
    pub fields: IncompressibleFields,
    pub face_flux: IncompressibleFaceFluxField,
}

/// Taylor–Green 初场：Rhie-Chow 压力投影（固定解析速度，调整压力）并播种 div-free 面通量。
pub fn taylor_green_prepare_initial_fields(
    mesh: &StructuredMesh3d,
    config: &IncompressibleCaseConfig,
    boundary: &BoundarySet,
    pseudo_time_step: Real,
    fields: IncompressibleFields,
) -> Result<TaylorGreenPreparedInitial> {
    let predictor_config =
        IncompressibleMomentumPredictorConfig::new(config.kinematic_viscosity, pseudo_time_step)?
            .with_body_force(config.body_force)?
            .with_velocity_under_relaxation(1.0)?
            .with_convection_scheme(config.convection_scheme);
    let momentum = assemble_incompressible_momentum_predictor_with_boundary_and_flux_3d(
        mesh,
        &fields,
        boundary,
        predictor_config,
        None,
    )?;
    let (projected, stats) = project_incompressible_fields_divergence_free_with_d_3d(
        fields,
        &momentum.d_coefficient,
        IncompressibleProjectionConfig::rhie_chow_pressure_only(
            mesh,
            boundary,
            config.density,
            config.linear_solvers.pressure,
            12,
            1.0e-6,
        ),
    )?;
    info!(
        iterations = stats.iterations,
        max_abs_divergence_before = %format_log_sci4(stats.max_abs_divergence_before),
        max_abs_divergence_after = %format_log_sci4(stats.max_abs_divergence_after),
        pressure_converged = stats.pressure_solve_converged,
        "Taylor–Green 初场 Rhie-Chow 散度投影"
    );
    let face_flux = IncompressibleFaceFluxField::from_rhie_chow(
        mesh,
        &projected,
        &momentum.d_coefficient,
        boundary,
    )?;
    Ok(TaylorGreenPreparedInitial {
        fields: projected,
        face_flux,
    })
}

/// 体积平均 kinetic energy \(E=\frac{1}{V}\int \frac{1}{2}\rho|\mathbf{u}|^2\,\mathrm{d}V\)。
#[must_use]
pub fn kinetic_energy(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    density: Real,
) -> Real {
    let mut integral = 0.0;
    let mut volume = 0.0;
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let idx = mesh.cell_index(i, j, k);
                let cell_volume = mesh.cell_metric(i, j, k).volume;
                let u = fields.velocity_x.values()[idx];
                let v = fields.velocity_y.values()[idx];
                let w = fields.velocity_z.values()[idx];
                integral += 0.5 * density * (u * u + v * v + w * w) * cell_volume;
                volume += cell_volume;
            }
        }
    }
    if volume > Real::EPSILON {
        integral / volume
    } else {
        0.0
    }
}

/// 解析动能比 \(E(t)/E(0)=\exp(-4\nu t)\)（Brachet et al. 1983）。
///
/// 网格坐标已缩至 \([0,1]^d\) 且动量扩散用 \(\nu^*=1/Re\) 时，无量纲时间 \(t^*\) 下
/// \(\exp(-4\nu t)=\exp(-4\nu^* L_{\mathrm{ref}}^2 t^*)\)。
#[must_use]
pub fn analytical_kinetic_energy_ratio(
    inv_reynolds: Real,
    reference_length: Real,
    nondimensional_time: Real,
) -> Real {
    (-4.0 * inv_reynolds * reference_length * reference_length * nondimensional_time).exp()
}

/// 在 spin-up 后区间估计 \(-\mathrm{d}\ln E/\mathrm{d}t^*\)，与解析 \(4\nu^* L_{\mathrm{ref}}^2\) 对比。
pub(crate) fn taylor_green_decay_rates(
    enabled: bool,
    inv_reynolds: Real,
    reference_length: Real,
    time_step: Real,
    history: &[Real],
) -> (Option<Real>, Option<Real>) {
    if !enabled {
        return (None, None);
    }
    const SPIN_UP_STEPS: usize = 10;
    let analytical = Some(4.0 * inv_reynolds * reference_length * reference_length);
    if history.len() <= SPIN_UP_STEPS + 1 {
        return (None, analytical);
    }
    let start = history[SPIN_UP_STEPS];
    let end = *history.last().expect("history has end");
    let elapsed = time_step * (history.len() - 1 - SPIN_UP_STEPS) as Real;
    if start <= Real::EPSILON || end <= Real::EPSILON || elapsed <= Real::EPSILON {
        return (None, analytical);
    }
    let rate = -(end / start).ln() / elapsed;
    (Some(rate), analytical)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::approx_eq;
    use crate::mesh::StructuredMesh3d;

    fn mesh_8x8x1() -> StructuredMesh3d {
        StructuredMesh3d::uniform_box("tg", 8, 8, 1, 1.0, 1.0, 0.1).expect("mesh")
    }

    #[test]
    fn taylor_green_initial_field_is_divergence_free_on_periodic_topology() {
        let mesh = mesh_8x8x1();
        let fields = taylor_green_initial_fields(&mesh).expect("fields");
        let energy = kinetic_energy(&mesh, &fields, 1.0);
        assert!(energy > 0.05);
        assert!(energy < 0.5);
    }

    #[test]
    fn analytical_decay_matches_reference_slope_at_small_time() {
        let inv_re = 0.01;
        let dt = 0.1;
        let ratio = analytical_kinetic_energy_ratio(inv_re, 1.0, dt);
        assert!(approx_eq(ratio, (-0.004_f64).exp(), 1.0e-12));
    }

    #[test]
    fn rhie_chow_projection_reduces_initial_divergence() {
        use crate::discretization::compute_incompressible_rhie_chow_divergence_3d;
        use crate::field::ScalarField;
        use crate::io::parse_case_str;
        use crate::solver::{
            IncompressibleProjectionConfig, project_incompressible_fields_divergence_free_3d,
        };

        let case = parse_case_str(
            r#"
name = "tg_projection"
benchmark_id = "taylor_green_3d"

[mesh]
kind = "structured_3d"
nx = 8
ny = 8
nz = 1
lx = 6.283185307179586
ly = 6.283185307179586
lz = 0.1

[physics]

[incompressible]
pressure = 0.0
velocity = [0.0, 0.0, 0.0]
density = 1.0
kinematic_viscosity = 0.1

[incompressible.reference]
length = 6.283185307179586
velocity = 1.0

[boundary.i_min]
kind = "periodic"
partner = "i_max"

[boundary.i_max]
kind = "periodic"
partner = "i_min"

[boundary.j_min]
kind = "periodic"
partner = "j_max"

[boundary.j_max]
kind = "periodic"
partner = "j_min"

[boundary.k_min]
kind = "symmetry"

[boundary.k_max]
kind = "symmetry"
"#,
        )
        .expect("parse");
        let mesh = case.mesh.as_3d().expect("mesh");
        let config = case.incompressible.expect("inc");
        let fields = taylor_green_initial_fields(mesh).expect("fields");
        let d = ScalarField::uniform(mesh.num_cells(), 1.0).expect("d");
        let div_before =
            compute_incompressible_rhie_chow_divergence_3d(mesh, &fields, &d, &case.boundary)
                .expect("div");
        let max_before = div_before
            .values()
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        assert!(max_before > 1.0e-4, "max_before={max_before}");

        let (projected, stats) = project_incompressible_fields_divergence_free_3d(
            fields,
            IncompressibleProjectionConfig::rhie_chow_pressure_only(
                mesh,
                &case.boundary,
                config.density,
                config.linear_solvers.pressure,
                12,
                1.0e-6,
            ),
        )
        .expect("project");
        assert!(stats.iterations >= 1);
        assert!(stats.max_abs_divergence_after < max_before * 0.01);
        assert!(stats.max_abs_divergence_after < 1.0e-3);
        assert!(stats.pressure_solve_converged);
        let div_after =
            compute_incompressible_rhie_chow_divergence_3d(mesh, &projected, &d, &case.boundary)
                .expect("div after");
        let max_after = div_after
            .values()
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        assert!(approx_eq(
            stats.max_abs_divergence_after,
            max_after,
            1.0e-12
        ));
    }

    #[test]
    fn projection_then_boundary_keeps_low_divergence_with_consistent_d() {
        use crate::discretization::{
            IncompressibleMomentumPredictorConfig, apply_incompressible_boundary_conditions_3d,
            assemble_incompressible_momentum_predictor_with_boundary_and_flux_3d,
            compute_incompressible_rhie_chow_divergence_3d,
        };
        use crate::io::parse_case_str;
        use crate::solver::{
            IncompressibleProjectionConfig, project_incompressible_fields_divergence_free_with_d_3d,
        };

        let case = parse_case_str(
            r#"
name = "tg_projection_boundary"
benchmark_id = "taylor_green_3d"

[mesh]
kind = "structured_3d"
nx = 16
ny = 16
nz = 1
lx = 6.283185307179586
ly = 6.283185307179586
lz = 0.1

[physics]

[time]
mode = "transient"
dt = 0.05

[incompressible]
pressure = 0.0
velocity = [0.0, 0.0, 0.0]
density = 1.0
kinematic_viscosity = 0.1

[incompressible.reference]
length = 6.283185307179586
velocity = 1.0

[boundary.i_min]
kind = "periodic"
partner = "i_max"

[boundary.i_max]
kind = "periodic"
partner = "i_min"

[boundary.j_min]
kind = "periodic"
partner = "j_max"

[boundary.j_max]
kind = "periodic"
partner = "j_min"

[boundary.k_min]
kind = "symmetry"

[boundary.k_max]
kind = "symmetry"
"#,
        )
        .expect("parse");
        let mesh = case.mesh.as_3d().expect("mesh");
        let config = case.incompressible.expect("inc");
        let fields = taylor_green_initial_fields(mesh).expect("fields");
        let predictor_config = IncompressibleMomentumPredictorConfig::new(
            config.kinematic_viscosity,
            case.time.dt.expect("dt"),
        )
        .expect("predictor")
        .with_body_force(config.body_force)
        .expect("body_force")
        .with_velocity_under_relaxation(1.0)
        .expect("urf")
        .with_convection_scheme(config.convection_scheme);
        let momentum = assemble_incompressible_momentum_predictor_with_boundary_and_flux_3d(
            mesh,
            &fields,
            &case.boundary,
            predictor_config,
            None,
        )
        .expect("momentum");
        let (mut projected, _) = project_incompressible_fields_divergence_free_with_d_3d(
            fields,
            &momentum.d_coefficient,
            IncompressibleProjectionConfig::rhie_chow_pressure_only(
                mesh,
                &case.boundary,
                config.density,
                config.linear_solvers.pressure,
                12,
                1.0e-6,
            ),
        )
        .expect("project");
        let div_before_bc = compute_incompressible_rhie_chow_divergence_3d(
            mesh,
            &projected,
            &momentum.d_coefficient,
            &case.boundary,
        )
        .expect("div before bc");
        let max_before_bc = div_before_bc
            .values()
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        apply_incompressible_boundary_conditions_3d(mesh, &mut projected, &case.boundary)
            .expect("apply bc");
        let div_after_bc = compute_incompressible_rhie_chow_divergence_3d(
            mesh,
            &projected,
            &momentum.d_coefficient,
            &case.boundary,
        )
        .expect("div after bc");
        let max_after_bc = div_after_bc
            .values()
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        assert!(
            max_after_bc <= max_before_bc * 2.0 + 1.0e-8,
            "max_before_bc={max_before_bc:e}, max_after_bc={max_after_bc:e}"
        );
    }

    #[test]
    fn initial_pressure_projection_reduces_but_not_eliminates_step1_predicted_divergence() {
        use crate::io::parse_case_str;
        use crate::solver::{
            IncompressiblePressureVelocityConfig, run_incompressible_pressure_velocity,
        };

        let case = parse_case_str(
            r#"
name = "tg_step1_probe"
benchmark_id = "taylor_green_3d"

[mesh]
kind = "structured_3d"
nx = 16
ny = 16
nz = 1
lx = 6.283185307179586
ly = 6.283185307179586
lz = 0.1

[physics]

[time]
mode = "transient"
dt = 0.05

[incompressible]
pressure = 0.0
velocity = [0.0, 0.0, 0.0]
density = 1.0
kinematic_viscosity = 0.1
convection_scheme = "central"
piso_correctors = 2

[incompressible.reference]
length = 6.283185307179586
velocity = 1.0

[boundary.i_min]
kind = "periodic"
partner = "i_max"

[boundary.i_max]
kind = "periodic"
partner = "i_min"

[boundary.j_min]
kind = "periodic"
partner = "j_max"

[boundary.j_max]
kind = "periodic"
partner = "j_min"

[boundary.k_min]
kind = "symmetry"

[boundary.k_max]
kind = "symmetry"
"#,
        )
        .expect("parse");
        let mesh = case.mesh.as_3d().expect("mesh");
        let config = case.incompressible.expect("inc");
        let dt = case.time.dt.expect("dt");
        let base_fields = taylor_green_initial_fields(mesh).expect("base");
        let prepared = taylor_green_prepare_initial_fields(
            mesh,
            &config,
            &case.boundary,
            dt,
            base_fields.clone(),
        )
        .expect("projected");
        let projected_fields = prepared.fields;
        let base_solver_config = IncompressiblePressureVelocityConfig {
            mesh,
            density: config.density,
            kinematic_viscosity: config.kinematic_viscosity,
            body_force: config.body_force,
            velocity_under_relaxation: 1.0,
            pressure_under_relaxation: 1.0,
            pseudo_time_step: dt,
            convection_scheme: config.convection_scheme,
            pressure_correctors: config.piso_correctors.max(1),
            boundary: &case.boundary,
            max_iterations: 1,
            min_iterations: 0,
            tolerance: None,
            require_velocity_convergence: false,
            convergence_window: 1,
            snapshot_interval: None,
            linear_solvers: config.linear_solvers,
            transient_mode: true,
            initial_face_flux: None,
        };
        let projected_solver_config = IncompressiblePressureVelocityConfig {
            initial_face_flux: Some(prepared.face_flux),
            ..base_solver_config.clone()
        };
        let base_step1 = run_incompressible_pressure_velocity(&base_fields, base_solver_config)
            .expect("base step1");
        let projected_step1 =
            run_incompressible_pressure_velocity(&projected_fields, projected_solver_config)
                .expect("projected step1");
        assert!(
            projected_step1.max_abs_predicted_divergence
                < 0.5 * base_step1.max_abs_predicted_divergence,
            "base={} projected={}",
            base_step1.max_abs_predicted_divergence,
            projected_step1.max_abs_predicted_divergence
        );
        assert!(
            projected_step1.max_abs_predicted_divergence < 1.0e-3,
            "projected={}",
            projected_step1.max_abs_predicted_divergence
        );
    }

    #[test]
    fn step1_predicted_divergence_is_small_after_initial_coupling() {
        use crate::io::parse_case_str;
        use crate::solver::{
            IncompressiblePressureVelocityConfig, run_incompressible_pressure_velocity,
        };

        let case = parse_case_str(
            r#"
name = "tg_step1_coupled"
benchmark_id = "taylor_green_3d"

[mesh]
kind = "structured_3d"
nx = 16
ny = 16
nz = 1
lx = 6.283185307179586
ly = 6.283185307179586
lz = 0.1

[physics]

[time]
mode = "transient"
dt = 0.005

[incompressible]
pressure = 0.0
velocity = [0.0, 0.0, 0.0]
density = 1.0
kinematic_viscosity = 0.1
convection_scheme = "central"
piso_correctors = 2

[incompressible.reference]
length = 6.283185307179586
velocity = 1.0

[boundary.i_min]
kind = "periodic"
partner = "i_max"

[boundary.i_max]
kind = "periodic"
partner = "i_min"

[boundary.j_min]
kind = "periodic"
partner = "j_max"

[boundary.j_max]
kind = "periodic"
partner = "j_min"

[boundary.k_min]
kind = "symmetry"

[boundary.k_max]
kind = "symmetry"
"#,
        )
        .expect("parse");
        let mesh = case.mesh.as_3d().expect("mesh");
        let config = case.incompressible.expect("inc");
        let dt = case.time.dt.expect("dt");
        let prepared = taylor_green_prepare_initial_fields(
            mesh,
            &config,
            &case.boundary,
            dt,
            taylor_green_initial_fields(mesh).expect("base"),
        )
        .expect("projected");
        let step1 = run_incompressible_pressure_velocity(
            &prepared.fields,
            IncompressiblePressureVelocityConfig {
                mesh,
                density: config.density,
                kinematic_viscosity: config.kinematic_viscosity,
                body_force: config.body_force,
                velocity_under_relaxation: 1.0,
                pressure_under_relaxation: 1.0,
                pseudo_time_step: dt,
                convection_scheme: config.convection_scheme,
                pressure_correctors: config.piso_correctors.max(1),
                boundary: &case.boundary,
                max_iterations: 1,
                min_iterations: 0,
                tolerance: None,
                require_velocity_convergence: false,
                convergence_window: 1,
                snapshot_interval: None,
                linear_solvers: config.linear_solvers,
                transient_mode: true,
                initial_face_flux: Some(prepared.face_flux),
            },
        )
        .expect("step1");
        assert!(
            step1.max_abs_predicted_divergence < 1.0e-4,
            "predicted={}",
            step1.max_abs_predicted_divergence
        );
    }
}
