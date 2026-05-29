//! asimu CLI 入口。
//!
//! SPDX-License-Identifier: Apache-2.0 OR MIT

use clap::Parser;

use asimu::app;
use asimu::config::Cli;

fn main() {
    if let Err(err) = run() {
        eprintln!("错误: {err}");
        std::process::exit(1);
    }
}

fn run() -> asimu::error::Result<()> {
    let cli = Cli::parse();
    app::run(cli)
}
