//! 单元中心梯度（Green-Gauss）。

#![allow(clippy::too_many_arguments)]

use crate::boundary::BoundarySet;
use crate::core::{Real, Vector3};
use crate::discretization::BoundaryGhostBuffer;
use crate::error::{AsimuError, Result};
use crate::field::{PrimitiveFields, ScalarField, primitive_from_conserved_relaxed};
use crate::mesh::{BoundaryMesh, LogicalFace3d, StructuredMesh3d};
use crate::physics::IdealGasEoS;

/// 速度分量与温度的单元中心梯度（SoA）。
#[derive(Debug, Clone, PartialEq)]
pub struct GradientFields {
    pub du_dx: ScalarField,
    pub du_dy: ScalarField,
    pub du_dz: ScalarField,
    pub dv_dx: ScalarField,
    pub dv_dy: ScalarField,
    pub dv_dz: ScalarField,
    pub dw_dx: ScalarField,
    pub dw_dy: ScalarField,
    pub dw_dz: ScalarField,
    pub dt_dx: ScalarField,
    pub dt_dy: ScalarField,
    pub dt_dz: ScalarField,
}

impl GradientFields {
    pub fn zeros(num_cells: usize) -> Result<Self> {
        Ok(Self {
            du_dx: ScalarField::uniform(num_cells, 0.0)?,
            du_dy: ScalarField::uniform(num_cells, 0.0)?,
            du_dz: ScalarField::uniform(num_cells, 0.0)?,
            dv_dx: ScalarField::uniform(num_cells, 0.0)?,
            dv_dy: ScalarField::uniform(num_cells, 0.0)?,
            dv_dz: ScalarField::uniform(num_cells, 0.0)?,
            dw_dx: ScalarField::uniform(num_cells, 0.0)?,
            dw_dy: ScalarField::uniform(num_cells, 0.0)?,
            dw_dz: ScalarField::uniform(num_cells, 0.0)?,
            dt_dx: ScalarField::uniform(num_cells, 0.0)?,
            dt_dy: ScalarField::uniform(num_cells, 0.0)?,
            dt_dz: ScalarField::uniform(num_cells, 0.0)?,
        })
    }

    #[must_use]
    pub fn num_cells(&self) -> usize {
        self.du_dx.len()
    }

    #[must_use]
    pub fn velocity_grad_at(&self, cell: usize) -> VelocityGradient {
        VelocityGradient {
            du: [
                self.du_dx.values()[cell],
                self.du_dy.values()[cell],
                self.du_dz.values()[cell],
            ],
            dv: [
                self.dv_dx.values()[cell],
                self.dv_dy.values()[cell],
                self.dv_dz.values()[cell],
            ],
            dw: [
                self.dw_dx.values()[cell],
                self.dw_dy.values()[cell],
                self.dw_dz.values()[cell],
            ],
            dt: [
                self.dt_dx.values()[cell],
                self.dt_dy.values()[cell],
                self.dt_dz.values()[cell],
            ],
        }
    }
}

/// 单元 \((u,v,w,T)\) 梯度张量分量。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VelocityGradient {
    pub du: [Real; 3],
    pub dv: [Real; 3],
    pub dw: [Real; 3],
    pub dt: [Real; 3],
}

/// Green-Gauss：\(\nabla\phi \approx \frac{1}{V}\sum_f \phi_f \mathbf{S}_f\)，\(\phi_f=\frac{1}{2}(\phi_L+\phi_R)\)。
pub fn compute_green_gauss_gradients_3d(
    mesh: &StructuredMesh3d,
    primitives: &PrimitiveFields,
    eos: &IdealGasEoS,
    boundaries: &BoundarySet,
    ghosts: &BoundaryGhostBuffer,
    min_pressure: Real,
    out: &mut GradientFields,
) -> Result<()> {
    let n = mesh.num_cells();
    if primitives.num_cells() != n || out.num_cells() != n {
        return Err(AsimuError::Field(
            "梯度场与原始变量场尺寸不一致".to_string(),
        ));
    }
    out.clear();
    let temperatures = cell_temperatures(primitives, eos)?;
    accumulate_green_gauss_interior(mesh, primitives, &temperatures, out)?;
    accumulate_green_gauss_boundaries(
        mesh,
        primitives,
        eos,
        boundaries,
        ghosts,
        min_pressure,
        &temperatures,
        out,
    )
}

fn accumulate_green_gauss_interior(
    mesh: &StructuredMesh3d,
    primitives: &PrimitiveFields,
    temperatures: &[Real],
    out: &mut GradientFields,
) -> Result<()> {
    let nx = mesh.nx;
    let ny = mesh.ny;
    let nz = mesh.nz;
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx.saturating_sub(1) {
                let left = mesh.cell_index(i, j, k);
                let right = mesh.cell_index(i + 1, j, k);
                let face = mesh.i_face_metric(i, j, k);
                accumulate_scalar_pair(
                    out,
                    left,
                    right,
                    primitives,
                    temperatures,
                    face.area_vector,
                    mesh.cell_metric(i, j, k).volume,
                    mesh.cell_metric(i + 1, j, k).volume,
                )?;
            }
        }
    }
    for k in 0..nz {
        for j in 0..ny.saturating_sub(1) {
            for i in 0..nx {
                let lower = mesh.cell_index(i, j, k);
                let upper = mesh.cell_index(i, j + 1, k);
                let face = mesh.j_face_metric(i, j, k);
                accumulate_scalar_pair(
                    out,
                    lower,
                    upper,
                    primitives,
                    temperatures,
                    face.area_vector,
                    mesh.cell_metric(i, j, k).volume,
                    mesh.cell_metric(i, j + 1, k).volume,
                )?;
            }
        }
    }
    for k in 0..nz.saturating_sub(1) {
        for j in 0..ny {
            for i in 0..nx {
                let lower = mesh.cell_index(i, j, k);
                let upper = mesh.cell_index(i, j, k + 1);
                let face = mesh.k_face_metric(i, j, k);
                accumulate_scalar_pair(
                    out,
                    lower,
                    upper,
                    primitives,
                    temperatures,
                    face.area_vector,
                    mesh.cell_metric(i, j, k).volume,
                    mesh.cell_metric(i, j, k + 1).volume,
                )?;
            }
        }
    }
    Ok(())
}

fn accumulate_green_gauss_boundaries(
    mesh: &StructuredMesh3d,
    primitives: &PrimitiveFields,
    eos: &IdealGasEoS,
    boundaries: &BoundarySet,
    ghosts: &BoundaryGhostBuffer,
    min_pressure: Real,
    temperatures: &[Real],
    out: &mut GradientFields,
) -> Result<()> {
    for patch in boundaries.patches() {
        for &face in &patch.face_ids {
            let owner = BoundaryMesh::face_owner(mesh, face)?.index() as usize;
            let (logical, local) = LogicalFace3d::decode(face)?;
            let (i, j, k) = mesh.face_ij(logical, local)?;
            let geom = mesh.boundary_face_metric(logical, i, j, k);
            let volume = mesh.cell_metric(i, j, k).volume;
            let ghost = ghosts.get_face(face).ok_or_else(|| {
                AsimuError::Boundary(format!("边界面 FaceId({}) 缺少 ghost", face.index()))
            })?;
            let ghost_prim = primitive_from_conserved_relaxed(eos, &ghost.conserved, min_pressure)?;
            let t_ghost = ghost_prim.pressure / (ghost_prim.density * eos.gas_constant);
            accumulate_boundary_cell(
                out,
                owner,
                primitives,
                temperatures,
                &ghost_prim,
                t_ghost,
                geom.area_vector,
                volume,
            )?;
        }
    }
    Ok(())
}

fn cell_temperatures(primitives: &PrimitiveFields, eos: &IdealGasEoS) -> Result<Vec<Real>> {
    let n = primitives.num_cells();
    let mut t = vec![0.0; n];
    for (i, ti) in t.iter_mut().enumerate().take(n) {
        let rho = primitives.density.values()[i];
        let p = primitives.pressure.values()[i];
        if rho <= 0.0 || p <= 0.0 {
            return Err(AsimuError::Field(
                "密度或压力非正，无法计算温度".to_string(),
            ));
        }
        *ti = p / (rho * eos.gas_constant);
    }
    Ok(t)
}

fn accumulate_scalar_pair(
    out: &mut GradientFields,
    left: usize,
    right: usize,
    prim: &PrimitiveFields,
    temp: &[Real],
    area_vector: Vector3,
    vol_left: Real,
    vol_right: Real,
) -> Result<()> {
    let u_l = prim.velocity_x.values()[left];
    let u_r = prim.velocity_x.values()[right];
    let v_l = prim.velocity_y.values()[left];
    let v_r = prim.velocity_y.values()[right];
    let w_l = prim.velocity_z.values()[left];
    let w_r = prim.velocity_z.values()[right];
    let t_l = temp[left];
    let t_r = temp[right];
    let samples = FaceScalarSamples {
        u_l,
        u_r,
        v_l,
        v_r,
        w_l,
        w_r,
        t_l,
        t_r,
    };
    add_face_contribution(out, left, samples, area_vector, vol_left, 1.0)?;
    add_face_contribution(out, right, samples, area_vector, vol_right, -1.0)?;
    Ok(())
}

fn accumulate_boundary_cell(
    out: &mut GradientFields,
    cell: usize,
    prim: &PrimitiveFields,
    temp: &[Real],
    ghost_prim: &crate::physics::PrimitiveState,
    t_ghost: Real,
    area_vector: Vector3,
    volume: Real,
) -> Result<()> {
    add_face_contribution(
        out,
        cell,
        FaceScalarSamples {
            u_l: prim.velocity_x.values()[cell],
            u_r: ghost_prim.velocity[0],
            v_l: prim.velocity_y.values()[cell],
            v_r: ghost_prim.velocity[1],
            w_l: prim.velocity_z.values()[cell],
            w_r: ghost_prim.velocity[2],
            t_l: temp[cell],
            t_r: t_ghost,
        },
        area_vector,
        volume,
        1.0,
    )?;
    Ok(())
}

#[derive(Clone, Copy)]
struct FaceScalarSamples {
    u_l: Real,
    u_r: Real,
    v_l: Real,
    v_r: Real,
    w_l: Real,
    w_r: Real,
    t_l: Real,
    t_r: Real,
}

fn add_face_contribution(
    out: &mut GradientFields,
    cell: usize,
    samples: FaceScalarSamples,
    area_vector: Vector3,
    volume: Real,
    sign: Real,
) -> Result<()> {
    let FaceScalarSamples {
        u_l,
        u_r,
        v_l,
        v_r,
        w_l,
        w_r,
        t_l,
        t_r,
    } = samples;
    if volume <= 0.0 {
        return Err(AsimuError::Mesh("单元体积非正".to_string()));
    }
    let scale = sign / volume;
    let u_f = 0.5 * (u_l + u_r);
    let v_f = 0.5 * (v_l + v_r);
    let w_f = 0.5 * (w_l + w_r);
    let t_f = 0.5 * (t_l + t_r);
    out.du_dx.values_mut()[cell] += scale * u_f * area_vector.x;
    out.du_dy.values_mut()[cell] += scale * u_f * area_vector.y;
    out.du_dz.values_mut()[cell] += scale * u_f * area_vector.z;
    out.dv_dx.values_mut()[cell] += scale * v_f * area_vector.x;
    out.dv_dy.values_mut()[cell] += scale * v_f * area_vector.y;
    out.dv_dz.values_mut()[cell] += scale * v_f * area_vector.z;
    out.dw_dx.values_mut()[cell] += scale * w_f * area_vector.x;
    out.dw_dy.values_mut()[cell] += scale * w_f * area_vector.y;
    out.dw_dz.values_mut()[cell] += scale * w_f * area_vector.z;
    out.dt_dx.values_mut()[cell] += scale * t_f * area_vector.x;
    out.dt_dy.values_mut()[cell] += scale * t_f * area_vector.y;
    out.dt_dz.values_mut()[cell] += scale * t_f * area_vector.z;
    Ok(())
}

impl GradientFields {
    fn clear(&mut self) {
        for f in [
            &mut self.du_dx,
            &mut self.du_dy,
            &mut self.du_dz,
            &mut self.dv_dx,
            &mut self.dv_dy,
            &mut self.dv_dz,
            &mut self.dw_dx,
            &mut self.dw_dy,
            &mut self.dw_dz,
            &mut self.dt_dx,
            &mut self.dt_dy,
            &mut self.dt_dz,
        ] {
            for v in f.values_mut() {
                *v = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{BoundaryKind, BoundaryPatch};
    use crate::discretization::apply_compressible_boundary_conditions;
    use crate::field::ConservedFields;
    use crate::mesh::StructuredMesh3d;
    use crate::physics::FreestreamParams;

    #[test]
    fn uniform_flow_has_zero_velocity_gradient() {
        let mesh = StructuredMesh3d::uniform_box("box", 4, 4, 4, 1.0, 1.0, 1.0).expect("mesh");
        let eos = IdealGasEoS::AIR_STANDARD;
        let fs = FreestreamParams {
            mach: 0.1,
            ..FreestreamParams::default()
        };
        let fields = ConservedFields::from_freestream(mesh.num_cells(), &eos, &fs).expect("fields");
        let mut patches = Vec::new();
        for name in ["i_min", "i_max", "j_min", "j_max", "k_min", "k_max"] {
            patches.push(BoundaryPatch::new(
                name,
                mesh.resolve_logical_boundary(name).expect("faces"),
                BoundaryKind::Farfield {
                    mach: fs.mach,
                    pressure: fs.pressure,
                    temperature: fs.temperature,
                    alpha: 0.0,
                    beta: 0.0,
                },
            ));
        }
        let boundary = crate::boundary::BoundarySet::new(patches);
        let mut ghosts = BoundaryGhostBuffer::new();
        apply_compressible_boundary_conditions(
            &mesh,
            &boundary,
            &fields,
            &mut ghosts,
            &eos,
            &fs,
            None,
        )
        .expect("bc");
        let mut prim = PrimitiveFields::zeros(mesh.num_cells()).expect("prim");
        prim.fill_from_conserved(&fields, &eos, 1.0e-6)
            .expect("fill");
        let mut grad = GradientFields::zeros(mesh.num_cells()).expect("grad");
        compute_green_gauss_gradients_3d(&mesh, &prim, &eos, &boundary, &ghosts, 1.0e-6, &mut grad)
            .expect("grad");
        for i in 0..mesh.num_cells() {
            let g = grad.velocity_grad_at(i);
            for comp in [g.du, g.dv, g.dw] {
                assert!(comp.iter().all(|&x| x.abs() < 1.0e-10));
            }
        }
    }
}
