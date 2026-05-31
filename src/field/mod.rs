//! 物理场存储（SoA，v0.2 骨架）。
//!
//! 理论：[`docs/theory/fvm_diffusion.md`](../../docs/theory/fvm_diffusion.md)

mod algebra;
mod conserved;
mod initial;
mod primitive;
mod residual;

use std::collections::BTreeMap;

use crate::core::Real;
use crate::error::{AsimuError, Result};
use crate::mesh::StructuredMesh1d;
use crate::physics::IdealGasEoS;

pub use conserved::{
    ConservedFields, clamp_conserved_positivity, positivity_pressure_floor,
    primitive_from_conserved, primitive_from_conserved_relaxed,
};
pub use initial::{
    FluidInitialConfig, InitialKind, InitialSet, ScalarInitial, build_scalar_initial,
};
pub use primitive::PrimitiveFields;
pub use residual::ConservedResidual;

/// 标量场，长度与网格单元数一致。
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarField {
    values: Vec<Real>,
}

impl ScalarField {
    /// 构造常值场；`num_cells` 必须大于 0。
    pub fn uniform(num_cells: usize, value: Real) -> Result<Self> {
        if num_cells == 0 {
            return Err(AsimuError::Field("num_cells 必须大于 0".to_string()));
        }
        Ok(Self {
            values: vec![value; num_cells],
        })
    }

    /// 从已有数据构造；拒绝空向量。
    pub fn from_values(values: Vec<Real>) -> Result<Self> {
        if values.is_empty() {
            return Err(AsimuError::Field("values 不能为空".to_string()));
        }
        Ok(Self { values })
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn values(&self) -> &[Real] {
        &self.values
    }

    pub fn values_mut(&mut self) -> &mut [Real] {
        &mut self.values
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_field() {
        assert!(matches!(
            ScalarField::from_values(vec![]).unwrap_err(),
            AsimuError::Field(_)
        ));
    }

    #[test]
    fn uniform_field_has_length() {
        let field = ScalarField::uniform(4, 1.5).expect("field");
        assert_eq!(field.len(), 4);
        assert!(field.values().iter().all(|&v| v == 1.5));
    }
}
