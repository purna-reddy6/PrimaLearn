/// main.rs — Subprocess entry point for the spelke-enumerator.
///
/// Protocol:
///   Input (stdin):  JSON with fields: library, task, max_cost, time_budget, schema_version
///   Output (stdout): JSON with fields: schema_version, result, programs, n_explored, elapsed_ms
///
/// Python calls: subprocess.run(["./spelke-enumerator"], input=json_payload, capture_output=True)

mod enumerator;
mod evaluator;
mod types;

use std::io::{self, Read};
use std::time::Duration;

use enumerator::{EnumeratorConfig, TypeDirectedEnumerator};
use serde::{Deserialize, Serialize};
use types::LibraryPrimitive;

#[derive(Deserialize)]
struct InputPayload {
    #[allow(dead_code)]
    schema_version: u32,
    library: Vec<LibraryPrimitive>,
    #[allow(dead_code)]
    task_id: Option<String>,
    max_cost: u8,
    time_budget: f64,
}

#[derive(Serialize)]
struct OutputPayload {
    schema_version: u32,
    /// "found" | "not_found" | "timeout" | "node_cap"
    result: String,
    /// S-expression programs at type "grid", sorted by cost
    programs: Vec<ProgramEntry>,
    n_explored: u64,
    elapsed_ms: u64,
}

#[derive(Serialize)]
struct ProgramEntry {
    cost: u8,
    sexp: String,
}

fn main() {
    // Read all of stdin
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).expect("Failed to read stdin");

    // Parse input JSON
    let payload: InputPayload = match serde_json::from_str(&input) {
        Ok(p) => p,
        Err(e) => {
            let error_output = serde_json::json!({
                "schema_version": 1,
                "result": "error",
                "error": format!("JSON parse error: {}", e),
                "programs": [],
                "n_explored": 0,
                "elapsed_ms": 0
            });
            println!("{}", serde_json::to_string(&error_output).unwrap());
            std::process::exit(1);
        }
    };

    // Configure enumerator
    let config = EnumeratorConfig {
        max_cost: payload.max_cost,
        time_budget: Duration::from_secs_f64(payload.time_budget),
        node_cap: 200_000,
    };

    let enumerator = TypeDirectedEnumerator::new(config);

    // Run enumeration
    let result = enumerator.enumerate(&payload.library);

    // Determine result status
    let status = if result.hit_timeout {
        "timeout"
    } else if result.hit_node_cap {
        "node_cap"
    } else if result.programs.is_empty() {
        "not_found"
    } else {
        "found"
    };

    // Collect all grid programs as S-expressions
    let program_entries: Vec<ProgramEntry> = result
        .programs
        .iter()
        .map(|(cost, prog)| ProgramEntry {
            cost: *cost,
            sexp: prog.to_sexp(),
        })
        .collect();

    let output = OutputPayload {
        schema_version: 1,
        result: status.to_string(),
        programs: program_entries,
        n_explored: result.n_explored,
        elapsed_ms: result.elapsed_ms,
    };

    println!("{}", serde_json::to_string(&output).unwrap());
}
