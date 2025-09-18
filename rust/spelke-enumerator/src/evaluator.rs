/// evaluator.rs — Program evaluation stub.
///
/// In the subprocess architecture, evaluation is done by Python (calling node.evaluate()).
/// Rust enumerates programs, returns them as S-expressions, Python evaluates them.
///
/// This module is a stub for future PyO3 integration where Rust would evaluate directly.

use crate::types::{ProgramSExpr, Task};
use std::rc::Rc;

/// Evaluate a program S-expression on a task.
/// Currently a stub — in subprocess mode, evaluation is done in Python.
/// Returns None to indicate "evaluation not implemented in Rust".
pub fn evaluate_on_task(_prog: &Rc<ProgramSExpr>, _task: &Task) -> Option<bool> {
    None
}
