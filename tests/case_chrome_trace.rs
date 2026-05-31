//! Chrome trace 路径解析（case.toml + CLI）。

use std::path::Path;

use asimu::io::parse_case_str;

#[test]
fn cli_chrome_trace_overrides_observability() {
    let content = r#"
name = "trace_cli"
[mesh]
kind = "structured_1d"
cells = 4
length = 1.0
[physics]
diffusivity = 1.0
[output]
dir = "out"
[observability]
chrome_trace = "profiling/from_toml.json"
"#;
    let case = parse_case_str(content).expect("parse");
    let cwd = std::env::current_dir().expect("cwd");
    let from_cli = case
        .effective_chrome_trace_path(Some("cli/trace.json"))
        .expect("cli")
        .expect("some");
    assert_eq!(from_cli, cwd.join("cli/trace.json"));
    let from_flag = case
        .effective_chrome_trace_path(Some(""))
        .expect("flag")
        .expect("some");
    assert_eq!(from_flag, Path::new("out/profiling/trace.json"));
}

#[test]
fn resolves_chrome_trace_relative_to_output_dir() {
    let content = r#"
name = "trace_test"
[mesh]
kind = "structured_1d"
cells = 4
length = 1.0
[physics]
diffusivity = 1.0
[output]
dir = "out"
[observability]
chrome_trace = "profiling/trace.json"
"#;
    let case = parse_case_str(content).expect("parse");
    let path = case.resolved_chrome_trace_path().expect("resolve");
    let path = path.expect("some");
    // parse_case_str 无 case_dir，相对 output/ 落在 cwd/out/...
    assert!(path.ends_with("out/profiling/trace.json"));
}
