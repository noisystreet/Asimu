use crate::core::Real;
use crate::error::{AsimuError, Result};

use super::{Preconditioner, ensure_vector_len};

/// 恒等预条件器。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdentityPreconditioner {
    n: usize,
}

impl IdentityPreconditioner {
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self { n }
    }
}

impl Preconditioner for IdentityPreconditioner {
    fn dimension(&self) -> usize {
        self.n
    }

    fn apply(&self, rhs: &[Real], out: &mut [Real]) -> Result<()> {
        ensure_vector_len(rhs, self.n, "identity preconditioner rhs")?;
        ensure_vector_len(out, self.n, "identity preconditioner out")?;
        out.copy_from_slice(rhs);
        Ok(())
    }
}

/// LU-SGS 对角预条件器：按分量缩放 \(M^{-1}r\)。
#[derive(Debug, Clone, PartialEq)]
pub struct LusgsDiagonalPreconditioner {
    scales: Vec<Real>,
}

impl LusgsDiagonalPreconditioner {
    pub fn from_scales(scales: Vec<Real>) -> Result<Self> {
        if scales.is_empty() {
            return Err(AsimuError::Linalg(
                "LU-SGS 对角预条件器尺寸必须大于 0".to_string(),
            ));
        }
        if scales.iter().any(|s| !s.is_finite() || *s < 0.0) {
            return Err(AsimuError::Linalg(
                "LU-SGS 对角预条件器 scale 必须为有限非负数".to_string(),
            ));
        }
        Ok(Self { scales })
    }

    /// 从单元 \(\Delta t_i,\sigma_i\) 构造，并按 `components_per_cell` 重复到守恒分量。
    pub fn from_lusgs_diagonal(
        dt: &[Real],
        sigma: &[Real],
        omega: Real,
        components_per_cell: usize,
    ) -> Result<Self> {
        if dt.len() != sigma.len() || dt.is_empty() || components_per_cell == 0 {
            return Err(AsimuError::Linalg(
                "LU-SGS 对角预条件器 dt/sigma/分量数不一致".to_string(),
            ));
        }
        if !omega.is_finite() || omega <= 0.0 || omega > 1.0 {
            return Err(AsimuError::Linalg(
                "LU-SGS 对角预条件器 omega 须在 (0,1] 内".to_string(),
            ));
        }
        let mut scales = Vec::with_capacity(dt.len() * components_per_cell);
        for (&dt_i, &sigma_i) in dt.iter().zip(sigma.iter()) {
            if !dt_i.is_finite() || dt_i <= 0.0 || !sigma_i.is_finite() || sigma_i < 0.0 {
                return Err(AsimuError::Linalg(
                    "LU-SGS 对角预条件器 dt 须为正且 sigma 非负".to_string(),
                ));
            }
            let scale = omega * dt_i / (1.0 + dt_i * sigma_i);
            for _ in 0..components_per_cell {
                scales.push(scale);
            }
        }
        Self::from_scales(scales)
    }
}

impl Preconditioner for LusgsDiagonalPreconditioner {
    fn dimension(&self) -> usize {
        self.scales.len()
    }

    fn apply(&self, rhs: &[Real], out: &mut [Real]) -> Result<()> {
        ensure_vector_len(rhs, self.dimension(), "lusgs diagonal rhs")?;
        ensure_vector_len(out, self.dimension(), "lusgs diagonal out")?;
        for ((dst, src), scale) in out.iter_mut().zip(rhs.iter()).zip(self.scales.iter()) {
            *dst = scale * src;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lusgs_diagonal_preconditioner_repeats_cell_scales() {
        let p = LusgsDiagonalPreconditioner::from_lusgs_diagonal(&[0.5], &[3.0], 0.5, 5)
            .expect("precond");
        let mut out = [0.0; 5];
        p.apply(&[1.0, 2.0, 3.0, 4.0, 5.0], &mut out)
            .expect("apply");
        let scale = 0.5 * 0.5 / (1.0 + 0.5 * 3.0);
        assert!((out[4] - 5.0 * scale).abs() < 1.0e-12);
    }
}
