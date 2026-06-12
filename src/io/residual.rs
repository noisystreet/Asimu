//! 瞬态残差历史写出（CSV，默认 log10(RMS(ρ̇))）。

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::error::Result;
use crate::io::limits::validate_input_path;
use crate::solver::{CompressibleStepInfo, IncompressiblePressureVelocityStepInfo};

/// 写出残差时间历程 CSV。
pub fn write_residual_csv(path: &Path, history: &[CompressibleStepInfo]) -> Result<()> {
    validate_input_path(path)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = File::create(path)?;
    let mut out = BufWriter::new(file);
    writeln!(out, "step,t,dt,log10_residual")?;
    for info in history {
        writeln!(
            out,
            "{},{:.16e},{:.16e},{:.16e}",
            info.step, info.physical_time, info.dt, info.residual_log10
        )?;
    }
    out.flush()?;
    Ok(())
}

/// 写出不可压缩 pressure-velocity 残差时间历程 CSV。
pub fn write_incompressible_residual_csv(
    path: &Path,
    history: &[IncompressiblePressureVelocityStepInfo],
    physical_time_scale: f64,
) -> Result<()> {
    validate_input_path(path)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = File::create(path)?;
    let mut out = BufWriter::new(file);
    writeln!(
        out,
        "step,t,dt,log10_residual,continuity,face_flux_divergence,momentum_residual,velocity_delta_interior,pressure_equation_residual,pressure_iters,pressure_residual,pressure_converged,momentum_iters,momentum_residual_norm,momentum_converged,coupling_converged"
    )?;
    let mut previous_time = 0.0;
    for info in history {
        let time = info.nondimensional_time * physical_time_scale;
        let dt = time - previous_time;
        previous_time = time;
        writeln!(
            out,
            "{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{},{:.16e},{},{},{:.16e},{},{}",
            info.step,
            time,
            dt,
            log10_positive(info.continuity),
            info.continuity,
            info.face_flux_divergence,
            info.momentum_residual,
            info.velocity_delta_interior,
            info.pressure_equation_residual,
            info.pressure_solve_iterations,
            info.pressure_solve_residual,
            info.pressure_solve_converged,
            info.momentum_solve_iterations,
            info.momentum_solve_residual,
            info.momentum_solve_converged,
            info.converged
        )?;
    }
    out.flush()?;
    Ok(())
}

fn log10_positive(value: f64) -> f64 {
    value.max(f64::MIN_POSITIVE).log10()
}
