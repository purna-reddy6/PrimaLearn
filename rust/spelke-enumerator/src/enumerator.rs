/// enumerator.rs — TypeDirectedEnumerator ported from Python.
///
/// Bottom-up, cost-bounded enumeration over the Spelke DSL.
/// Algorithm: fills table[(type_repr, cost)] → Vec<ProgramSExpr>
/// for costs 0..=max_cost, then returns programs at type "grid" for evaluation.

use std::collections::HashMap;
use std::rc::Rc;
use std::time::{Duration, Instant};

use crate::types::{
    cap_for_type_repr, parse_type_repr, typevar_candidate_reprs, Cost, LibraryPrimitive,
    LitValue, ParsedType, ProgramSExpr, Table, TypeRepr,
};

pub struct EnumeratorConfig {
    pub max_cost: Cost,
    pub time_budget: Duration,
    pub node_cap: u64,
}

impl Default for EnumeratorConfig {
    fn default() -> Self {
        EnumeratorConfig {
            max_cost: 5,
            time_budget: Duration::from_secs(5),
            node_cap: 200_000,
        }
    }
}

pub struct EnumeratorResult {
    /// All programs found at type "grid" across all cost levels, in order
    pub programs: Vec<(Cost, Rc<ProgramSExpr>)>,
    pub n_explored: u64,
    pub elapsed_ms: u64,
    pub hit_timeout: bool,
    pub hit_node_cap: bool,
}

pub struct TypeDirectedEnumerator {
    pub config: EnumeratorConfig,
}

impl TypeDirectedEnumerator {
    pub fn new(config: EnumeratorConfig) -> Self {
        TypeDirectedEnumerator { config }
    }

    /// Build the enumeration table and return all programs at type "grid".
    /// This is the hot path — equivalent to Python's _build_table() + candidate scan.
    pub fn enumerate(&self, library: &[LibraryPrimitive]) -> EnumeratorResult {
        let start = Instant::now();
        let deadline = start + self.config.time_budget;

        // Parse and filter primitives (same logic as Python usable_prims)
        let mut prims: Vec<ParsedPrim> = library
            .iter()
            .filter_map(|p| {
                let parsed = parse_type_repr(&p.type_repr);
                // Must be an Arrow with a concrete result type
                if matches!(parsed, ParsedType::Arrow(_, _)) && parsed.has_concrete_result() {
                    Some(ParsedPrim {
                        name: p.name.clone(),
                        type_repr: p.type_repr.clone(),
                        parsed,
                        prior: p.prior,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort: highest prior first (matches Python)
        prims.sort_by(|a, b| b.prior.partial_cmp(&a.prior).unwrap_or(std::cmp::Ordering::Equal));

        let mut table: Table = HashMap::new();
        let mut n_explored: u64 = 0;
        let mut hit_timeout = false;
        let mut hit_node_cap = false;

        // Cost 0: VarNode("input") at type "grid"
        add_to_table(
            &mut table,
            "grid",
            0,
            Rc::new(ProgramSExpr::Var { name: "input".to_string() }),
        );

        // Cost 1: literal constants
        // Color literals 0..9 at type "color"
        for c in 0u8..10 {
            add_to_table(
                &mut table,
                "color",
                1,
                Rc::new(ProgramSExpr::Lit { value: LitValue::Int(c as i64) }),
            );
        }
        // Int literals 1..5 at type "int"
        for i in 1u8..=5 {
            add_to_table(
                &mut table,
                "int",
                1,
                Rc::new(ProgramSExpr::Lit { value: LitValue::Int(i as i64) }),
            );
        }

        // Costs 1..=max_cost: primitive applications
        'cost_loop: for target_cost in 1u8..=self.config.max_cost {
            if Instant::now() >= deadline {
                hit_timeout = true;
                break;
            }

            // Per-cost-level node counter (reset each level like Python)
            let mut level_nodes: u64 = 0;

            for prim in &prims {
                if Instant::now() >= deadline {
                    hit_timeout = true;
                    break 'cost_loop;
                }

                // Prim itself costs 1; remaining budget goes to arguments
                let arg_budget = if target_cost >= 1 { target_cost - 1 } else { continue };

                let steps = prim.parsed.arrow_steps();
                if steps.is_empty() {
                    continue;
                }

                let prim_node = Rc::new(ProgramSExpr::Prim { name: prim.name.clone() });

                self.try_apply(
                    prim_node,
                    &steps,
                    0,
                    arg_budget,
                    target_cost,
                    &mut table,
                    deadline,
                    &mut level_nodes,
                );

                if level_nodes > self.config.node_cap {
                    hit_node_cap = true;
                    break; // break prim loop, continue to next cost level
                }
            }

            n_explored += level_nodes;
        }

        let elapsed_ms = start.elapsed().as_millis() as u64;

        // Collect all programs at type "grid" across cost levels
        let mut programs: Vec<(Cost, Rc<ProgramSExpr>)> = Vec::new();
        for cost in 0u8..=self.config.max_cost {
            if let Some(bucket) = table.get(&("grid".to_string(), cost)) {
                for prog in bucket {
                    programs.push((cost, prog.clone()));
                }
            }
        }

        EnumeratorResult {
            programs,
            n_explored,
            elapsed_ms,
            hit_timeout,
            hit_node_cap,
        }
    }

    /// Recursive argument application — port of Python's _try_apply().
    ///
    /// Fills table[(result_type_repr, target_cost)] with AppNodes constructed
    /// by distributing `remaining` budget among argument slots.
    fn try_apply(
        &self,
        node: Rc<ProgramSExpr>,
        steps: &[(ParsedType, ParsedType)],
        step_idx: usize,
        remaining: Cost,
        target_cost: Cost,
        table: &mut Table,
        deadline: Instant,
        nodes_explored: &mut u64,
    ) {
        if step_idx >= steps.len() {
            return;
        }
        if Instant::now() >= deadline {
            return;
        }
        if *nodes_explored > self.config.node_cap {
            return;
        }

        let (ref arg_type, ref result_type) = steps[step_idx];
        let is_last = step_idx == steps.len() - 1;
        let arg_type_repr = arg_type.to_repr();

        for arg_cost in 0u8..=remaining {
            if Instant::now() >= deadline {
                return;
            }
            let left = remaining - arg_cost;

            // Gather candidates at (arg_type_repr, arg_cost)
            let candidates: Vec<Rc<ProgramSExpr>> = {
                let mut out = Vec::new();

                // Direct lookup
                if let Some(bucket) = table.get(&(arg_type_repr.clone(), arg_cost)) {
                    out.extend(bucket.iter().cloned());
                }

                // TypeVariable: also try curated candidate types
                if arg_type.is_variable() {
                    for &cand_repr in typevar_candidate_reprs() {
                        if cand_repr != arg_type_repr {
                            if let Some(bucket) = table.get(&(cand_repr.to_string(), arg_cost)) {
                                out.extend(bucket.iter().cloned());
                            }
                        }
                    }
                }

                out
            };

            for arg_node in candidates {
                if Instant::now() >= deadline {
                    return;
                }
                *nodes_explored += 1;
                if *nodes_explored > self.config.node_cap {
                    return;
                }

                let app = Rc::new(ProgramSExpr::App {
                    func: node.clone(),
                    arg: arg_node,
                });

                if is_last {
                    // Must consume the entire budget
                    if left == 0 {
                        let result_repr = result_type.to_repr();
                        add_to_table(table, &result_repr, target_cost, app);
                    }
                } else {
                    // Partial application — continue with remaining steps
                    // left >= 0 always (u8 subtraction can't underflow here since arg_cost <= remaining)
                    self.try_apply(
                        app,
                        steps,
                        step_idx + 1,
                        left,
                        target_cost,
                        table,
                        deadline,
                        nodes_explored,
                    );
                }
            }
        }
    }
}

/// Parsed primitive with its type decomposed
struct ParsedPrim {
    name: String,
    #[allow(dead_code)]
    type_repr: TypeRepr,
    parsed: ParsedType,
    prior: f64,
}

/// Add a node to the table bucket, respecting the per-type cap
fn add_to_table(table: &mut Table, type_repr: &str, cost: Cost, node: Rc<ProgramSExpr>) {
    let cap = cap_for_type_repr(type_repr);
    let key = (type_repr.to_string(), cost);
    let bucket = table.entry(key).or_insert_with(Vec::new);
    if bucket.len() < cap {
        bucket.push(node);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{LibraryPrimitive, LitValue, ProgramSExpr};

    fn make_prim(name: &str, type_repr: &str, prior: f64) -> LibraryPrimitive {
        LibraryPrimitive {
            name: name.to_string(),
            type_repr: type_repr.to_string(),
            prior,
        }
    }

    #[test]
    fn test_enumerate_empty_library() {
        let enumerator = TypeDirectedEnumerator::new(EnumeratorConfig {
            max_cost: 3,
            ..Default::default()
        });
        let result = enumerator.enumerate(&[]);
        // Only cost-0 VarNode("input") at grid should be present
        assert!(!result.programs.is_empty());
        let (cost, prog) = &result.programs[0];
        assert_eq!(*cost, 0);
        assert!(matches!(prog.as_ref(), ProgramSExpr::Var { name } if name == "input"));
    }

    #[test]
    fn test_enumerate_grid_to_grid_prim() {
        // A single grid→grid primitive should produce programs at cost 1
        let lib = vec![make_prim("identity", "grid \u{2192} grid", 0.5)];
        let enumerator = TypeDirectedEnumerator::new(EnumeratorConfig {
            max_cost: 2,
            ..Default::default()
        });
        let result = enumerator.enumerate(&lib);
        // Should have: cost=0 (VarNode), cost=1 (identity input)
        let cost1: Vec<_> = result.programs.iter().filter(|(c, _)| *c == 1).collect();
        assert!(!cost1.is_empty(), "Expected programs at cost 1");
        // The cost-1 program should be (identity input)
        let sexp = cost1[0].1.to_sexp();
        assert_eq!(sexp, "(identity input)", "Expected (identity input) but got {sexp}");
    }

    #[test]
    fn test_cost_model_app_is_additive() {
        // grid→grid→grid prim at cost 1, applied to cost-0 (input) and cost-0 (input)
        // → total cost 1 + 0 + 0 = 1
        // So should appear at target_cost=1
        let lib = vec![make_prim("overlay", "grid \u{2192} grid \u{2192} grid", 0.5)];
        let enumerator = TypeDirectedEnumerator::new(EnumeratorConfig {
            max_cost: 3,
            ..Default::default()
        });
        let result = enumerator.enumerate(&lib);
        let cost1: Vec<_> = result.programs.iter().filter(|(c, _)| *c == 1).collect();
        // (overlay input input) at cost 1
        assert!(!cost1.is_empty(), "overlay(input, input) should be at cost 1");
        let found = cost1.iter().any(|(_, p)| p.to_sexp() == "((overlay input) input)");
        assert!(found, "Expected ((overlay input) input) at cost 1");
    }

    #[test]
    fn test_timeout_respected() {
        // Library with expensive primitive, very short budget
        let lib = vec![
            make_prim("f", "grid \u{2192} grid", 0.5),
            make_prim("g", "grid \u{2192} grid", 0.4),
        ];
        let enumerator = TypeDirectedEnumerator::new(EnumeratorConfig {
            max_cost: 6,
            time_budget: Duration::from_millis(1), // 1ms — should timeout quickly
            node_cap: 200_000,
        });
        let result = enumerator.enumerate(&lib);
        // Should have hit timeout or finished quickly — just check it doesn't hang
        assert!(result.elapsed_ms < 5000, "Should not take more than 5s");
    }

    #[test]
    fn test_lit_nodes_in_table() {
        let enumerator = TypeDirectedEnumerator::new(EnumeratorConfig {
            max_cost: 1,
            ..Default::default()
        });
        let result = enumerator.enumerate(&[]);
        // Cost-0: input at grid
        // Cost-1: color literals 0..9, int literals 1..5
        // But result.programs only has type "grid"
        // We can't directly inspect the table from here — test via enumerate result
        // Just verify it completes
        assert!(!result.hit_node_cap);
    }
}
