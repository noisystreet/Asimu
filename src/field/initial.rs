//! 初始条件数据与场构建。

use crate::core::Real;
use crate::error::{AsimuError, Result};
use crate::mesh::StructuredMesh1d;
use crate::physics::{FreestreamParams, IdealGasEoS};

use super::{ConservedFields, ScalarField};

/// 标量初始条件类型。
#[derive(Debug, Clone, PartialEq)]
pub enum InitialKind {
    /// 常值 \(\phi = c\)。
    Uniform { value: Real },
    /// 线性分布：\(\phi(x) = \text{left} + (\text{right}-\text{left})\cdot\frac{x-\text{origin}}{\text{length}}\)。
    Linear { left: Real, right: Real },
    /// 逐单元给定（长度须等于 `num_cells`）。
    Values { data: Vec<Real> },
}

impl InitialKind {
    /// TOML `kind` 字段解析。
    pub fn from_toml(
        kind: &str,
        value: Option<Real>,
        left: Option<Real>,
        right: Option<Real>,
        data: Option<Vec<Real>>,
    ) -> Option<Self> {
        match kind {
            "uniform" => value.map(|v| Self::Uniform { value: v }),
            "linear" => match (left, right) {
                (Some(left), Some(right)) => Some(Self::Linear { left, right }),
                _ => None,
            },
            "values" => data.map(|data| Self::Values { data }),
            _ => None,
        }
    }
}

/// 命名标量初始条件（如 `phi`）。
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarInitial {
    pub name: String,
    pub kind: InitialKind,
}

impl ScalarInitial {
    pub fn new(name: impl Into<String>, kind: InitialKind) -> Self {
        Self {
            name: name.into(),
            kind,
        }
    }

    /// 在 1D 均匀网格上生成标量场。
    pub fn build_on_mesh(&self, mesh: &StructuredMesh1d) -> Result<ScalarField> {
        build_scalar_initial(mesh, &self.kind)
    }
}

/// 算例全部初始条件。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct InitialSet {
    scalars: Vec<ScalarInitial>,
}

impl InitialSet {
    #[must_use]
    pub fn new(scalars: Vec<ScalarInitial>) -> Self {
        Self { scalars }
    }

    #[must_use]
    pub fn scalars(&self) -> &[ScalarInitial] {
        &self.scalars
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.scalars.is_empty()
    }

    pub fn find(&self, name: &str) -> Option<&ScalarInitial> {
        self.scalars.iter().find(|s| s.name == name)
    }

    /// 未指定时返回全零场。
    pub fn build_scalar_or_zero(
        &self,
        name: &str,
        mesh: &StructuredMesh1d,
    ) -> Result<ScalarField> {
        match self.find(name) {
            Some(initial) => initial.build_on_mesh(mesh),
            None => ScalarField::uniform(mesh.num_cells(), 0.0),
        }
    }

    /// 校验名称非空、无重复。
    pub fn validate(&self) -> Result<()> {
        let mut seen = std::collections::BTreeSet::new();
        for scalar in &self.scalars {
            if scalar.name.trim().is_empty() {
                return Err(AsimuError::Field("初始条件名称不能为空".to_string()));
            }
            if !seen.insert(scalar.name.clone()) {
                return Err(AsimuError::Field(format!(
                    "重复的初始条件 \"{}\"",
                    scalar.name
                )));
            }
        }
        Ok(())
    }
}

/// 可压缩流初始条件配置（freestream / 标量场）。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FluidInitialConfig {
    pub freestream: Option<FreestreamParams>,
    pub scalars: InitialSet,
}

impl FluidInitialConfig {
    pub fn build_conserved(&self, num_cells: usize, eos: &IdealGasEoS) -> Result<ConservedFields> {
        if let Some(fs) = &self.freestream {
            return ConservedFields::from_freestream(num_cells, eos, fs);
        }
        Err(AsimuError::Field(
            "可压缩算例须指定 [freestream] 或 [restart]".to_string(),
        ))
    }
}

/// 由 `InitialKind` 在 1D 网格上构建标量场。
pub fn build_scalar_initial(mesh: &StructuredMesh1d, kind: &InitialKind) -> Result<ScalarField> {
    let n = mesh.num_cells();
    match kind {
        InitialKind::Uniform { value } => ScalarField::uniform(n, *value),
        InitialKind::Linear { left, right } => {
            let dx = mesh.dx();
            let origin = mesh.origin;
            let length = mesh.length;
            if length <= 0.0 {
                return Err(AsimuError::Field("域长度必须大于 0".to_string()));
            }
            let mut values = Vec::with_capacity(n);
            for i in 0..n {
                let x = origin + (i as Real + 0.5) * dx;
                let t = (x - origin) / length;
                values.push(left + (right - left) * t);
            }
            ScalarField::from_values(values)
        }
        InitialKind::Values { data } => {
            if data.len() != n {
                return Err(AsimuError::Field(format!(
                    "初始 values 长度 {} 与单元数 {n} 不一致",
                    data.len()
                )));
            }
            ScalarField::from_values(data.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::approx_eq;

    #[test]
    fn linear_initial_matches_endpoints() {
        let mesh = StructuredMesh1d::new("line", 4, 0.0, 1.0).expect("mesh");
        let kind = InitialKind::Linear {
            left: 0.0,
            right: 1.0,
        };
        let field = build_scalar_initial(&mesh, &kind).expect("build");
        assert!(approx_eq(field.values()[0], 0.125, 1.0e-12));
        assert!(approx_eq(field.values()[3], 0.875, 1.0e-12));
    }

    #[test]
    fn values_initial_checks_length() {
        let mesh = StructuredMesh1d::new("line", 3, 0.0, 1.0).expect("mesh");
        let err = build_scalar_initial(
            &mesh,
            &InitialKind::Values {
                data: vec![0.0, 1.0],
            },
        )
        .unwrap_err();
        assert!(matches!(err, AsimuError::Field(_)));
    }
}
