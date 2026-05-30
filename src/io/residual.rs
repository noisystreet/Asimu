//! 瞬态残差历史写出（CSV，默认 log10(RMS(ρ̇))）。

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::error::Result;
use crate::io::limits::validate_input_path;
use crate::solver::CompressibleStepInfo;

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
