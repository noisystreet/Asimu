//! 有量纲 / 无量纲均匀来流 + 远场 BC 成对测试 fixture（仅 `cfg(test)` 编译）。

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

/// 由来流 Mach 构造的有量纲 / 无量纲对称 fixture。
pub struct FreestreamPairFixture {
    pub dim_eos: IdealGasEoS,
    pub dim_fs: FreestreamParams,
    pub nd_eos: IdealGasEoS,
    pub nd_fs: FreestreamParams,
    pub reference: ReferenceScales,
    dim_viscous: ViscousPhysicsConfig,
    pub nd_viscous: ViscousPhysicsConfig,
}

/// 单套来流 + BC 上下文（有量纲或无量纲之一）。
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
        let mut nd_eos = dim_eos;
        nd_eos.gas_constant = reference.nondimensional_gas_constant();
        let nd_fs = FreestreamParams {
            mach,
            pressure: 1.0 / dim_eos.gamma,
            temperature: 1.0,
            ..FreestreamParams::default()
        };
        Self {
            dim_eos,
            dim_fs,
            nd_eos,
            nd_fs,
            reference,
            dim_viscous,
            nd_viscous,
        }
    }

    #[must_use]
    pub fn min_pressure_dimensional(&self) -> Real {
        1.0e-6
    }

    #[must_use]
    pub fn min_pressure_nondimensional(&self) -> Real {
        1.0e-6 / self.dim_eos.gamma
    }

    /// 无粘成对：有量纲侧。
    pub fn inviscid_dimensional<'a>(&'a self) -> UniformFarfieldSide<'a> {
        UniformFarfieldSide {
            label: "dimensional",
            eos: &self.dim_eos,
            fs: &self.dim_fs,
            ctx: FreestreamContext::dimensional(&self.dim_eos),
            min_pressure: self.min_pressure_dimensional(),
            viscous: None,
        }
    }

    /// 无粘成对：无量纲侧。
    pub fn inviscid_nondimensional<'a>(&'a self) -> UniformFarfieldSide<'a> {
        UniformFarfieldSide {
            label: "nondimensional",
            eos: &self.nd_eos,
            fs: &self.nd_fs,
            ctx: FreestreamContext::new(&self.nd_eos, Some(&self.reference), None),
            min_pressure: self.min_pressure_nondimensional(),
            viscous: None,
        }
    }

    /// NS 成对：有量纲侧。
    pub fn viscous_dimensional<'a>(&'a self) -> UniformFarfieldSide<'a> {
        UniformFarfieldSide {
            label: "dimensional",
            eos: &self.dim_eos,
            fs: &self.dim_fs,
            ctx: FreestreamContext::dimensional(&self.dim_eos),
            min_pressure: self.min_pressure_dimensional(),
            viscous: Some(&self.dim_viscous),
        }
    }

    /// NS 成对：无量纲侧。
    pub fn viscous_nondimensional<'a>(&'a self) -> UniformFarfieldSide<'a> {
        UniformFarfieldSide {
            label: "nondimensional",
            eos: &self.nd_eos,
            fs: &self.nd_fs,
            ctx: FreestreamContext::new(
                &self.nd_eos,
                Some(&self.reference),
                Some(&self.nd_viscous),
            ),
            min_pressure: self.min_pressure_nondimensional(),
            viscous: Some(&self.nd_viscous),
        }
    }

    /// 依次执行有量纲 / 无量纲回调（单测内成对断言用）。
    pub fn for_each_inviscid_side<F>(&self, mut f: F)
    where
        F: FnMut(&UniformFarfieldSide<'_>),
    {
        f(&self.inviscid_dimensional());
        f(&self.inviscid_nondimensional());
    }

    /// 依次执行 NS 有量纲 / 无量纲回调。
    pub fn for_each_viscous_side<F>(&self, mut f: F)
    where
        F: FnMut(&UniformFarfieldSide<'_>),
    {
        f(&self.viscous_dimensional());
        f(&self.viscous_nondimensional());
    }
}

/// 均匀盒网格 + 六面远场 BC + ghost（来流经 `FreestreamContext` 构造）。
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
    fn pair_fixture_freestream_density_is_unity_in_nondimensional_mode() {
        let pair = FreestreamPairFixture::air_sutherland(0.5);
        let nd = pair.inviscid_nondimensional();
        let prim = nd.ctx.primitive(nd.fs).expect("prim");
        assert!((prim.density - 1.0).abs() < 1.0e-12);
        assert!((prim.temperature - 1.0).abs() < 1.0e-12);
    }
}
