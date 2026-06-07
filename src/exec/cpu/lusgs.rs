//! LU-SGS 对角更新：`base + scale·R`（5 个 SoA 场）。

use crate::core::Real;

/// 5 分量守恒 SoA 只读 slice。
pub struct ConservedSoA<'a> {
    pub rho: &'a [Real],
    pub mx: &'a [Real],
    pub my: &'a [Real],
    pub mz: &'a [Real],
    pub energy: &'a [Real],
}

/// 5 分量守恒 SoA 可变 slice。
pub struct ConservedSoAMut<'a> {
    pub rho: &'a mut [Real],
    pub mx: &'a mut [Real],
    pub my: &'a mut [Real],
    pub mz: &'a mut [Real],
    pub energy: &'a mut [Real],
}

/// LU-SGS 对角更新输入。
pub struct LusgsDiagonalUpdate<'a> {
    pub out: ConservedSoAMut<'a>,
    pub base: ConservedSoA<'a>,
    pub residual: ConservedSoA<'a>,
    pub scale: &'a [Real],
}

/// `out ← base + scale[i] * residual`（逐单元 scale）。
pub fn assign_lusgs_diagonal_update(update: LusgsDiagonalUpdate<'_>) {
    debug_assert_eq!(update.out.rho.len(), update.scale.len());
    #[cfg(feature = "simd-fvm")]
    {
        let mut update = update;
        assign_lusgs_diagonal_update_simd(&mut update);
    }
    #[cfg(not(feature = "simd-fvm"))]
    {
        let n = update.scale.len();
        for i in 0..n {
            let s = update.scale[i];
            update.out.rho[i] = update.base.rho[i] + s * update.residual.rho[i];
            update.out.mx[i] = update.base.mx[i] + s * update.residual.mx[i];
            update.out.my[i] = update.base.my[i] + s * update.residual.my[i];
            update.out.mz[i] = update.base.mz[i] + s * update.residual.mz[i];
            update.out.energy[i] = update.base.energy[i] + s * update.residual.energy[i];
        }
    }
}

#[cfg(feature = "simd-fvm")]
fn assign_lusgs_diagonal_update_simd(update: &mut LusgsDiagonalUpdate<'_>) {
    use wide::f64x4;

    let n = update.scale.len();
    let mut i = 0;
    while i + 4 <= n {
        let s = f64x4::new([
            update.scale[i],
            update.scale[i + 1],
            update.scale[i + 2],
            update.scale[i + 3],
        ]);
        axpy4(
            &mut update.out.rho[i..i + 4],
            &update.base.rho[i..i + 4],
            &update.residual.rho[i..i + 4],
            s,
        );
        axpy4(
            &mut update.out.mx[i..i + 4],
            &update.base.mx[i..i + 4],
            &update.residual.mx[i..i + 4],
            s,
        );
        axpy4(
            &mut update.out.my[i..i + 4],
            &update.base.my[i..i + 4],
            &update.residual.my[i..i + 4],
            s,
        );
        axpy4(
            &mut update.out.mz[i..i + 4],
            &update.base.mz[i..i + 4],
            &update.residual.mz[i..i + 4],
            s,
        );
        axpy4(
            &mut update.out.energy[i..i + 4],
            &update.base.energy[i..i + 4],
            &update.residual.energy[i..i + 4],
            s,
        );
        i += 4;
    }
    while i < n {
        let s = update.scale[i];
        update.out.rho[i] = update.base.rho[i] + s * update.residual.rho[i];
        update.out.mx[i] = update.base.mx[i] + s * update.residual.mx[i];
        update.out.my[i] = update.base.my[i] + s * update.residual.my[i];
        update.out.mz[i] = update.base.mz[i] + s * update.residual.mz[i];
        update.out.energy[i] = update.base.energy[i] + s * update.residual.energy[i];
        i += 1;
    }
}

#[cfg(feature = "simd-fvm")]
#[inline]
fn axpy4(out: &mut [Real], base: &[Real], inc: &[Real], scale: wide::f64x4) {
    let b = wide::f64x4::new([base[0], base[1], base[2], base[3]]);
    let r = wide::f64x4::new([inc[0], inc[1], inc[2], inc[3]]);
    let y = b + scale * r;
    let arr = y.to_array();
    out.copy_from_slice(&arr);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::approx_eq;

    fn reference_update(base: [Real; 5], res: [Real; 5], scale: Real) -> [Real; 5] {
        [
            base[0] + scale * res[0],
            base[1] + scale * res[1],
            base[2] + scale * res[2],
            base[3] + scale * res[3],
            base[4] + scale * res[4],
        ]
    }

    #[test]
    fn lusgs_diagonal_update_matches_reference() {
        let n = 13;
        let base_rho = (0..n).map(|i| i as Real + 1.0).collect::<Vec<_>>();
        let base_mx = base_rho.iter().map(|v| v * 0.1).collect::<Vec<_>>();
        let base_my = base_rho.iter().map(|v| v * 0.2).collect::<Vec<_>>();
        let base_mz = base_rho.iter().map(|v| v * 0.3).collect::<Vec<_>>();
        let base_energy = base_rho.iter().map(|v| v * 2.0).collect::<Vec<_>>();
        let res_rho = (0..n).map(|i| (i as Real) * 0.01).collect::<Vec<_>>();
        let res_mx = res_rho.clone();
        let res_my = res_rho.iter().map(|v| v * 2.0).collect::<Vec<_>>();
        let res_mz = res_rho.iter().map(|v| v * 3.0).collect::<Vec<_>>();
        let res_energy = res_rho.iter().map(|v| v * 4.0).collect::<Vec<_>>();
        let scale: Vec<Real> = (0..n).map(|i| 0.5 + i as Real * 0.03).collect();

        let mut out_rho = vec![0.0; n];
        let mut out_mx = vec![0.0; n];
        let mut out_my = vec![0.0; n];
        let mut out_mz = vec![0.0; n];
        let mut out_energy = vec![0.0; n];
        assign_lusgs_diagonal_update(LusgsDiagonalUpdate {
            out: ConservedSoAMut {
                rho: &mut out_rho,
                mx: &mut out_mx,
                my: &mut out_my,
                mz: &mut out_mz,
                energy: &mut out_energy,
            },
            base: ConservedSoA {
                rho: &base_rho,
                mx: &base_mx,
                my: &base_my,
                mz: &base_mz,
                energy: &base_energy,
            },
            residual: ConservedSoA {
                rho: &res_rho,
                mx: &res_mx,
                my: &res_my,
                mz: &res_mz,
                energy: &res_energy,
            },
            scale: &scale,
        });

        for i in 0..n {
            let exp = reference_update(
                [
                    base_rho[i],
                    base_mx[i],
                    base_my[i],
                    base_mz[i],
                    base_energy[i],
                ],
                [res_rho[i], res_mx[i], res_my[i], res_mz[i], res_energy[i]],
                scale[i],
            );
            assert!(approx_eq(out_rho[i], exp[0], 1.0e-12));
            assert!(approx_eq(out_mx[i], exp[1], 1.0e-12));
            assert!(approx_eq(out_my[i], exp[2], 1.0e-12));
            assert!(approx_eq(out_mz[i], exp[3], 1.0e-12));
            assert!(approx_eq(out_energy[i], exp[4], 1.0e-12));
        }
    }
}
