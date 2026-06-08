//! 着色桶 residual scatter（ADR 0013）。

mod atomic;
mod contribution;
mod inviscid;
mod ptr;
mod serial;
mod viscous;

pub use contribution::{
    InviscidPairScatter, InviscidResidualMut, InviscidScatterOp, ViscousRangeScatter,
    ViscousResidualMut, ViscousScatterOp, ViscousValidSlotScatter,
};
pub use inviscid::scatter_inviscid_pairs;
pub use serial::run_bucket_scatter;
pub use viscous::{scatter_viscous_bucket_range, scatter_viscous_valid_slots};

#[cfg(test)]
mod tests {
    use crate::core::{Real, approx_eq};
    use crate::exec::context::{ExecConfig, ExecutionContext, ResolvedScatterMode, ScatterMode};
    use crate::exec::metrics::MeshExecMetrics;

    use super::{
        InviscidPairScatter, InviscidResidualMut, InviscidScatterOp, ViscousRangeScatter,
        ViscousResidualMut, ViscousScatterOp, scatter_inviscid_pairs, scatter_viscous_bucket_range,
    };

    fn atomic_test_context(bucket_len: usize) -> ExecutionContext {
        ExecutionContext::new(
            ExecConfig {
                scatter_mode: ScatterMode::ParallelUnsafeAtomics,
                parallel_min_len: 1,
                ..ExecConfig::default()
            },
            MeshExecMetrics::new(bucket_len, bucket_len, bucket_len),
        )
    }

    #[test]
    fn scatter_viscous_serial_matches_atomic_parallel() {
        if !cfg!(feature = "parallel-fvm") {
            return;
        }
        #[derive(Copy, Clone)]
        struct Geom {
            owner: usize,
            neighbor: usize,
            owner_scale: Real,
            neighbor_scale: Real,
        }
        #[derive(Copy, Clone)]
        struct Flux {
            mx: Real,
            my: Real,
            mz: Real,
            energy: Real,
        }
        let geoms = [
            Geom {
                owner: 0,
                neighbor: 1,
                owner_scale: 1.0,
                neighbor_scale: -1.0,
            },
            Geom {
                owner: 2,
                neighbor: 3,
                owner_scale: 0.5,
                neighbor_scale: -0.5,
            },
        ];
        let fluxes = [
            Flux {
                mx: 1.0,
                my: 2.0,
                mz: 3.0,
                energy: 4.0,
            },
            Flux {
                mx: 10.0,
                my: 20.0,
                mz: 30.0,
                energy: 40.0,
            },
        ];
        let extract = |g: &Geom, f: &Flux| ViscousScatterOp {
            owner: g.owner,
            neighbor: g.neighbor,
            owner_scale: g.owner_scale,
            neighbor_scale: g.neighbor_scale,
            flux_mx: f.mx,
            flux_my: f.my,
            flux_mz: f.mz,
            flux_energy: f.energy,
        };

        let mut serial_mx = vec![0.0; 4];
        let mut serial_my = vec![0.0; 4];
        let mut serial_mz = vec![0.0; 4];
        let mut serial_energy = vec![0.0; 4];
        let serial_ctx = ExecutionContext::new(
            ExecConfig {
                scatter_mode: ScatterMode::Serial,
                ..ExecConfig::default()
            },
            MeshExecMetrics::new(2, 2, 2),
        );
        scatter_viscous_bucket_range(
            ViscousRangeScatter {
                ctx: &serial_ctx,
                bucket_len: 2,
                geoms: &geoms,
                fluxes: &fluxes,
                range: 0..2,
                residual: ViscousResidualMut {
                    mx: &mut serial_mx,
                    my: &mut serial_my,
                    mz: &mut serial_mz,
                    energy: &mut serial_energy,
                },
            },
            extract,
        );

        let mut atomic_mx = vec![0.0; 4];
        let mut atomic_my = vec![0.0; 4];
        let mut atomic_mz = vec![0.0; 4];
        let mut atomic_energy = vec![0.0; 4];
        let atomic_ctx = atomic_test_context(2);
        assert_eq!(
            atomic_ctx.resolved_scatter_mode(),
            ResolvedScatterMode::ParallelUnsafeAtomics
        );
        scatter_viscous_bucket_range(
            ViscousRangeScatter {
                ctx: &atomic_ctx,
                bucket_len: 2,
                geoms: &geoms,
                fluxes: &fluxes,
                range: 0..2,
                residual: ViscousResidualMut {
                    mx: &mut atomic_mx,
                    my: &mut atomic_my,
                    mz: &mut atomic_mz,
                    energy: &mut atomic_energy,
                },
            },
            extract,
        );

        for i in 0..4 {
            assert!(approx_eq(serial_mx[i], atomic_mx[i], 1.0e-12));
            assert!(approx_eq(serial_my[i], atomic_my[i], 1.0e-12));
            assert!(approx_eq(serial_mz[i], atomic_mz[i], 1.0e-12));
            assert!(approx_eq(serial_energy[i], atomic_energy[i], 1.0e-12));
        }
    }

    #[test]
    fn scatter_inviscid_serial_matches_atomic_parallel() {
        if !cfg!(feature = "parallel-fvm") {
            return;
        }
        #[derive(Copy, Clone)]
        struct Geom {
            owner: usize,
            neighbor: usize,
            owner_scale: Real,
            neighbor_scale: Real,
        }
        #[derive(Copy, Clone)]
        struct Flux {
            mass: Real,
            momentum: [Real; 3],
            energy: Real,
        }
        let pairs = [(
            Geom {
                owner: 0,
                neighbor: 1,
                owner_scale: 1.0,
                neighbor_scale: -1.0,
            },
            Flux {
                mass: 0.1,
                momentum: [1.0, 2.0, 3.0],
                energy: 4.0,
            },
        )];
        let extract = |g: &Geom, f: &Flux| InviscidScatterOp {
            owner: g.owner,
            neighbor: g.neighbor,
            owner_scale: g.owner_scale,
            neighbor_scale: g.neighbor_scale,
            mass: f.mass,
            momentum: f.momentum,
            energy: f.energy,
        };

        let mut serial_density = vec![0.0; 2];
        let mut serial_mx = vec![0.0; 2];
        let mut serial_my = vec![0.0; 2];
        let mut serial_mz = vec![0.0; 2];
        let mut serial_energy = vec![0.0; 2];
        let serial_ctx = ExecutionContext::new(
            ExecConfig {
                scatter_mode: ScatterMode::Serial,
                ..ExecConfig::default()
            },
            MeshExecMetrics::new(1, 1, 1),
        );
        scatter_inviscid_pairs(
            InviscidPairScatter {
                ctx: &serial_ctx,
                bucket_len: 1,
                pairs: &pairs,
                residual: InviscidResidualMut {
                    density: &mut serial_density,
                    mx: &mut serial_mx,
                    my: &mut serial_my,
                    mz: &mut serial_mz,
                    energy: &mut serial_energy,
                },
            },
            extract,
        );

        let mut atomic_density = vec![0.0; 2];
        let mut atomic_mx = vec![0.0; 2];
        let mut atomic_my = vec![0.0; 2];
        let mut atomic_mz = vec![0.0; 2];
        let mut atomic_energy = vec![0.0; 2];
        scatter_inviscid_pairs(
            InviscidPairScatter {
                ctx: &atomic_test_context(1),
                bucket_len: 1,
                pairs: &pairs,
                residual: InviscidResidualMut {
                    density: &mut atomic_density,
                    mx: &mut atomic_mx,
                    my: &mut atomic_my,
                    mz: &mut atomic_mz,
                    energy: &mut atomic_energy,
                },
            },
            extract,
        );

        for i in 0..2 {
            assert!(approx_eq(serial_density[i], atomic_density[i], 1.0e-12));
            assert!(approx_eq(serial_mx[i], atomic_mx[i], 1.0e-12));
            assert!(approx_eq(serial_my[i], atomic_my[i], 1.0e-12));
            assert!(approx_eq(serial_mz[i], atomic_mz[i], 1.0e-12));
            assert!(approx_eq(serial_energy[i], atomic_energy[i], 1.0e-12));
        }
    }
}
