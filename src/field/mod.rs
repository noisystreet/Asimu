//! 物理场存储（SoA，v0.2 骨架）。
//!
//! 理论：[`docs/theory/fvm_diffusion.md`](../../docs/theory/fvm_diffusion.md)

mod algebra;
mod conserved;
mod incompressible;
mod initial;
mod lusgs_diagonal;
mod positivity;
mod primitive;
mod residual;
mod scalar_field;

use std::collections::BTreeMap;

use crate::error::Result;
use crate::mesh::StructuredMesh1d;
use crate::physics::IdealGasEoS;

pub use conserved::{
    ConservedFields, ConservedFieldsT, clamp_conserved_positivity, positivity_pressure_floor,
    primitive_from_conserved, primitive_from_conserved_relaxed,
    primitive_from_conserved_relaxed_f32, primitive_from_conserved_relaxed_f32_from_state,
};
pub use incompressible::IncompressibleFields;
pub use initial::{
    FluidInitialConfig, InitialKind, InitialSet, ScalarInitial, build_scalar_initial,
};
pub use lusgs_diagonal::{
    LusgsDiagonalCoeffs, LusgsDiagonalCoeffsF32, LusgsDiagonalUpdateBackend,
    assign_lusgs_diagonal_update_f32,
};
pub use positivity::{
    is_physical_conserved, is_physical_conserved_f32, max_physical_increment_scale,
    max_physical_increment_scale_f32, state_after_increment, state_after_increment_f32,
};
pub use primitive::{PrimitiveFields, PrimitiveFieldsT, PrimitiveFillFromConserved};
pub use residual::{ConservedResidual, ConservedResidualT};
pub use scalar_field::{ScalarField, ScalarFieldT};

/// 命名标量场集合（与 DATA_MODEL `Fields` 对齐，v0.2 仅标量）。
#[derive(Debug, Clone, PartialEq)]
pub struct Fields {
    scalars: BTreeMap<String, ScalarField>,
}

impl Fields {
    #[must_use]
    pub fn new(scalars: BTreeMap<String, ScalarField>) -> Self {
        Self { scalars }
    }

    pub fn from_initial_set(mesh: &StructuredMesh1d, initial: &InitialSet) -> Result<Self> {
        initial.validate()?;
        let mut scalars = BTreeMap::new();
        for scalar in initial.scalars() {
            scalars.insert(scalar.name.clone(), scalar.build_on_mesh(mesh)?);
        }
        Ok(Self { scalars })
    }

    pub fn get(&self, name: &str) -> Option<&ScalarField> {
        self.scalars.get(name)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &ScalarField)> {
        self.scalars.iter().map(|(k, v)| (k.as_str(), v))
    }

    pub fn build_conserved(
        num_cells: usize,
        eos: &IdealGasEoS,
        config: &FluidInitialConfig,
    ) -> Result<ConservedFields> {
        config.build_conserved(num_cells, eos)
    }
}
