//! 无量纲均匀来流 + 远场 BC 测试 fixture（仅 `cfg(test)` 编译）。

use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};
use crate::core::Real;
use crate::discretization::{BoundaryGhostBuffer, apply_compressible_boundary_conditions};
use crate::field::ConservedFields;
use crate::mesh::{BoundaryMesh, StructuredMesh3d};
use crate::physics::{
    FreestreamContext, FreestreamParams, IdealGasEoS, ReferenceScales, ViscosityModel,
    ViscousPhysicsConfig,
};

const DIM_PRESSURE: Real = 101_325.0;
const DIM_TEMPERATURE: Real = 300.0;

/// 由来流 Mach 构造的无量纲来流 fixture。
pub struct FreestreamPairFixture {
    pub eos: IdealGasEoS,
    pub fs: FreestreamParams,
    pub reference: ReferenceScales,
    pub nd_viscous: ViscousPhysicsConfig,
}

/// 单套无量纲来流 + BC 上下文。
pub struct UniformFarfieldSide<'a> {
    pub label: &'static str,
    pub eos: &'a IdealGasEoS,
    pub fs: &'a FreestreamParams,
    pub ctx: FreestreamContext<'a>,
    pub min_pressure: Real,
    pub viscous: Option<&'a ViscousPhysicsConfig>,
}

impl FreestreamPairFixture {
    pub fn air_sutherland(mach: Real) -> Self {
        let dim_eos = IdealGasEoS::AIR_STANDARD;
        let dim_fs = FreestreamParams {
            mach,
            pressure: DIM_PRESSURE,
            temperature: DIM_TEMPERATURE,
            ..FreestreamParams::default()
        };
        let dim_viscous =
            ViscousPhysicsConfig::new(ViscosityModel::AIR_SUTHERLAND, 0.72).expect("visc");
        let reference =
            ReferenceScales::from_freestream(&dim_eos, &dim_fs, Some(&dim_viscous)).expect("ref");
        let mut nd_viscous = dim_viscous.clone();
        nd_viscous.inv_reynolds = reference.inv_reynolds();
        nd_viscous.viscosity_ref = Some(reference.viscosity);
        nd_viscous.temperature_ref = Some(reference.temperature);
        let mut eos = dim_eos;
        eos.gas_constant = reference.nondimensional_gas_constant();
        let fs = FreestreamParams {
            mach,
            pressure: 1.0 / dim_eos.gamma,
            temperature: 1.0,
            ..FreestreamParams::default()
        };
        Self {
            eos,
            fs,
            reference,
            nd_viscous,
        }
    }

    #[must_use]
    pub fn min_pressure(&self) -> Real {
        crate::field::positivity_pressure_floor(self.fs.pressure)
    }

    /// 无粘无量纲来流侧。
    pub fn inviscid_side<'a>(&'a self) -> UniformFarfieldSide<'a> {
        UniformFarfieldSide {
            label: "nondimensional",
            eos: &self.eos,
            fs: &self.fs,
            ctx: FreestreamContext::new(&self.eos, Some(&self.reference), None),
            min_pressure: self.min_pressure(),
            viscous: None,
        }
    }

    /// NS 无量纲来流侧。
    pub fn viscous_side<'a>(&'a self) -> UniformFarfieldSide<'a> {
        UniformFarfieldSide {
            label: "nondimensional",
            eos: &self.eos,
            fs: &self.fs,
            ctx: FreestreamContext::new(&self.eos, Some(&self.reference), Some(&self.nd_viscous)),
            min_pressure: self.min_pressure(),
            viscous: Some(&self.nd_viscous),
        }
    }

    /// 依次执行无粘回调（仅无量纲侧）。
    pub fn for_each_inviscid_side<F>(&self, mut f: F)
    where
        F: FnMut(&UniformFarfieldSide<'_>),
    {
        f(&self.inviscid_side());
    }

    /// 依次执行 NS 回调（仅无量纲侧）。
    pub fn for_each_viscous_side<F>(&self, mut f: F)
    where
        F: FnMut(&UniformFarfieldSide<'_>),
    {
        f(&self.viscous_side());
    }
}

/// 均匀盒网格 + 六面远场 BC + ghost。
pub fn uniform_farfield_box(
    nx: usize,
    ny: usize,
    nz: usize,
    lx: Real,
    ly: Real,
    lz: Real,
    side: &UniformFarfieldSide<'_>,
) -> (
    StructuredMesh3d,
    BoundarySet,
    ConservedFields,
    BoundaryGhostBuffer,
) {
    let mesh = StructuredMesh3d::uniform_box("box", nx, ny, nz, lx, ly, lz).expect("mesh");
    let mut patches = Vec::new();
    for name in ["i_min", "i_max", "j_min", "j_max", "k_min", "k_max"] {
        patches.push(BoundaryPatch::new(
            name,
            mesh.resolve_logical_boundary(name).expect("faces"),
            BoundaryKind::Farfield {
                mach: side.fs.mach,
                pressure: side.fs.pressure,
                temperature: side.fs.temperature,
                alpha: 0.0,
                beta: 0.0,
            },
        ));
    }
    let boundary = BoundarySet::new(patches);
    let fields = ConservedFields::from_freestream_context(mesh.num_cells(), &side.ctx, side.fs)
        .expect("fields");
    let mut ghosts = BoundaryGhostBuffer::new();
    apply_compressible_boundary_conditions(
        &mesh,
        &boundary,
        &fields,
        &mut ghosts,
        &side.ctx,
        side.fs,
        side.viscous,
    )
    .expect("bc");
    (mesh, boundary, fields, ghosts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_freestream_density_is_unity() {
        let pair = FreestreamPairFixture::air_sutherland(0.5);
        let side = pair.inviscid_side();
        let prim = side.ctx.primitive(side.fs).expect("prim");
        assert!((prim.density - 1.0).abs() < 1.0e-12);
        assert!((prim.temperature - 1.0).abs() < 1.0e-12);
    }
}
