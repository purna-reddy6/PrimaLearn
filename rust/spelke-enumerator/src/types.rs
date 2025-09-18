/// types.rs — Core type representations for the Spelke DSL enumerator.
///
/// TypeRepr is the canonical string representation of a type, matching Python's repr(t).
/// This must match Python exactly so table keys compare correctly during verification.

/// Type representation string — must match Python's repr(t) exactly.
/// Examples:
///   "grid", "int", "color", "bool", "object", "list[object]"
///   "grid → grid", "grid → grid → grid"
///   "'a" (TypeVariable)
pub type TypeRepr = String;

/// Cost — u8 is sufficient (max_cost is typically 5–6)
pub type Cost = u8;

/// A single ARC training example: (input_grid, output_grid)
#[derive(Debug, Clone, serde::Deserialize)]
pub struct TrainingExample {
    pub input: Vec<Vec<u8>>,
    pub output: Vec<Vec<u8>>,
}

/// A task: a task_id plus training pairs
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Task {
    pub task_id: String,
    pub train: Vec<TrainingExample>,
}

/// A primitive from the library.
/// In Rust we cannot call arbitrary Python closures, so we store only the metadata
/// needed for enumeration (name, type_repr, prior). Evaluation is done by calling
/// back to Python via subprocess.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct LibraryPrimitive {
    /// Name as it appears in the Python library (e.g. "rotate_cw", "abs_0")
    pub name: String,
    /// Type signature repr string — must match Python repr(t)
    pub type_repr: TypeRepr,
    /// Prior probability (log-space or raw — matches Python priors dict)
    pub prior: f64,
}

/// Parsed type from a type_repr string.
/// We parse these lazily for the enumeration logic.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ParsedType {
    /// Bare constructor: "grid", "int", "color", "bool", "object"
    Constructor(String),
    /// Parameterized list: "list[object]"
    List(Box<ParsedType>),
    /// TypeVariable: "'a", "'b"
    Variable(String),
    /// Arrow: A → B (right-associative)
    Arrow(Box<ParsedType>, Box<ParsedType>),
}

impl ParsedType {
    /// Return the repr string — must match Python's repr(t)
    pub fn to_repr(&self) -> String {
        match self {
            ParsedType::Constructor(name) => name.clone(),
            ParsedType::List(elem) => format!("list[{}]", elem.to_repr()),
            ParsedType::Variable(name) => format!("'{}", name),
            ParsedType::Arrow(arg, result) => {
                let arg_str = match arg.as_ref() {
                    ParsedType::Arrow(_, _) => format!("({})", arg.to_repr()),
                    _ => arg.to_repr(),
                };
                format!("{} \u{2192} {}", arg_str, result.to_repr())
            }
        }
    }

    /// True if this type contains no TypeVariables
    pub fn is_concrete(&self) -> bool {
        match self {
            ParsedType::Variable(_) => false,
            ParsedType::Constructor(_) => true,
            ParsedType::List(elem) => elem.is_concrete(),
            ParsedType::Arrow(arg, result) => arg.is_concrete() && result.is_concrete(),
        }
    }

    /// True if the final return type (stripping all arrows) is concrete
    pub fn has_concrete_result(&self) -> bool {
        match self {
            ParsedType::Arrow(_, result) => result.has_concrete_result(),
            other => other.is_concrete(),
        }
    }

    /// True if this is a TypeVariable
    pub fn is_variable(&self) -> bool {
        matches!(self, ParsedType::Variable(_))
    }

    /// Decompose an Arrow into (arg, result) steps (left to right)
    /// E.g. "grid → int → grid" → [(grid, int→grid), (int, grid)]
    pub fn arrow_steps(&self) -> Vec<(ParsedType, ParsedType)> {
        let mut steps = Vec::new();
        let mut cur = self.clone();
        while let ParsedType::Arrow(arg, result) = cur {
            let result_clone = (*result).clone();
            steps.push((*arg, *result));
            cur = result_clone;
        }
        steps
    }

    /// Final return type (strip all arrows)
    pub fn return_type(&self) -> &ParsedType {
        match self {
            ParsedType::Arrow(_, result) => result.return_type(),
            other => other,
        }
    }
}

/// Parse a type repr string back into a ParsedType.
/// This must round-trip: parse(repr(t)).to_repr() == repr(t)
pub fn parse_type_repr(s: &str) -> ParsedType {
    let s = s.trim();
    parse_type_repr_inner(s)
}

fn parse_type_repr_inner(s: &str) -> ParsedType {
    // Try to find top-level " → " (arrow), right-associative
    // We need to find " → " that is not inside parentheses or brackets
    if let Some(idx) = find_top_level_arrow(s) {
        let left = &s[..idx];
        let right = &s[idx + " \u{2192} ".len()..];
        return ParsedType::Arrow(
            Box::new(parse_type_repr_inner(left.trim())),
            Box::new(parse_type_repr_inner(right.trim())),
        );
    }

    // Parenthesized: (A → B)
    if s.starts_with('(') && s.ends_with(')') {
        let inner = &s[1..s.len() - 1];
        // Only strip parens if they're balanced
        if is_balanced(inner) {
            return parse_type_repr_inner(inner.trim());
        }
    }

    // List type: "list[...]"
    if s.starts_with("list[") && s.ends_with(']') {
        let inner = &s[5..s.len() - 1];
        return ParsedType::List(Box::new(parse_type_repr_inner(inner)));
    }

    // TypeVariable: starts with '
    if s.starts_with('\'') {
        return ParsedType::Variable(s[1..].to_string());
    }

    // Constructor
    ParsedType::Constructor(s.to_string())
}

/// Find the index of the top-level " → " arrow (not inside parens/brackets)
fn find_top_level_arrow(s: &str) -> Option<usize> {
    // Python repr uses Unicode →  (U+2192, 3 bytes UTF-8: E2 86 92)
    let arrow = " \u{2192} ";
    let bytes = s.as_bytes();
    let arrow_bytes = arrow.as_bytes();

    let mut depth_paren = 0i32;
    let mut depth_bracket = 0i32;
    let mut i = 0;

    while i < bytes.len() {
        match bytes[i] {
            b'(' => depth_paren += 1,
            b')' => depth_paren -= 1,
            b'[' => depth_bracket += 1,
            b']' => depth_bracket -= 1,
            _ => {}
        }
        if depth_paren == 0 && depth_bracket == 0 {
            if s[i..].starts_with(arrow) {
                return Some(i);
            }
        }
        // advance by char boundary
        i += 1;
        while i < bytes.len() && (bytes[i] & 0xC0) == 0x80 {
            i += 1;
        }
    }
    None
}

fn is_balanced(s: &str) -> bool {
    let mut depth = 0i32;
    for c in s.chars() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            _ => {}
        }
        if depth < 0 {
            return false;
        }
    }
    depth == 0
}

/// Enumeration table: (type_repr, cost) → list of program S-expressions (reference-counted)
pub type Table = std::collections::HashMap<(TypeRepr, Cost), Vec<std::rc::Rc<ProgramSExpr>>>;

/// Program S-expression — a lightweight representation of a program node.
/// Used internally in the table. We use an Rc to avoid cloning entire trees.
#[derive(Debug, Clone)]
pub enum ProgramSExpr {
    Var { name: String },
    Lit { value: LitValue },
    Prim { name: String },
    App { func: std::rc::Rc<ProgramSExpr>, arg: std::rc::Rc<ProgramSExpr> },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum LitValue {
    Int(i64),
}

impl ProgramSExpr {
    /// Pretty-print as a compact S-expression string (for JSON output)
    pub fn to_sexp(&self) -> String {
        match self {
            ProgramSExpr::Var { name } => name.clone(),
            ProgramSExpr::Lit { value } => match value {
                LitValue::Int(i) => i.to_string(),
            },
            ProgramSExpr::Prim { name } => name.clone(),
            ProgramSExpr::App { func, arg } => {
                format!("({} {})", func.to_sexp(), arg.to_sexp())
            }
        }
    }
}

/// Per-type table caps — matches Python _TYPE_CAPS exactly
pub fn cap_for_type_repr(type_repr: &str) -> usize {
    // Checked in order — first match wins (substring match like Python)
    if type_repr.contains("grid") {
        500
    } else if type_repr.contains("list[object]") {
        100
    } else if type_repr.contains("object") {
        100
    } else if type_repr.contains("int") {
        50
    } else if type_repr.contains("color") {
        20
    } else if type_repr.contains("bool") {
        20
    } else {
        200
    }
}

/// TypeVariable candidate types (matching Python _TYPEVAR_CANDIDATE_TYPES)
pub fn typevar_candidate_reprs() -> &'static [&'static str] {
    &["grid", "int", "color", "object", "list[object]"]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_roundtrip_constructor() {
        let reprs = ["grid", "int", "color", "bool", "object"];
        for r in &reprs {
            let parsed = parse_type_repr(r);
            assert_eq!(parsed.to_repr(), *r, "roundtrip failed for {r}");
        }
    }

    #[test]
    fn test_parse_roundtrip_list() {
        let r = "list[object]";
        assert_eq!(parse_type_repr(r).to_repr(), r);
    }

    #[test]
    fn test_parse_roundtrip_arrow_simple() {
        let r = "grid \u{2192} grid";
        assert_eq!(parse_type_repr(r).to_repr(), r);
    }

    #[test]
    fn test_parse_roundtrip_arrow_curried() {
        // Python: Arrow(tgrid, Arrow(tgrid, tgrid)) → "grid → grid → grid"
        let r = "grid \u{2192} grid \u{2192} grid";
        assert_eq!(parse_type_repr(r).to_repr(), r);
    }

    #[test]
    fn test_parse_roundtrip_arrow_nested_arg() {
        // (grid → grid) → grid
        let r = "(grid \u{2192} grid) \u{2192} grid";
        assert_eq!(parse_type_repr(r).to_repr(), r);
    }

    #[test]
    fn test_parse_typevar() {
        let r = "'a";
        let parsed = parse_type_repr(r);
        assert_eq!(parsed.to_repr(), r);
        assert!(parsed.is_variable());
    }

    #[test]
    fn test_arrow_steps() {
        // grid → int → grid decomposes into [(grid, int→grid), (int, grid)]
        let t = parse_type_repr("grid \u{2192} int \u{2192} grid");
        let steps = t.arrow_steps();
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].0.to_repr(), "grid");
        assert_eq!(steps[1].0.to_repr(), "int");
        assert_eq!(steps[1].1.to_repr(), "grid");
    }

    #[test]
    fn test_has_concrete_result() {
        assert!(parse_type_repr("grid \u{2192} grid").has_concrete_result());
        assert!(parse_type_repr("grid").has_concrete_result());
        assert!(!parse_type_repr("'a").has_concrete_result());
        assert!(!parse_type_repr("grid \u{2192} 'a").has_concrete_result());
    }

    #[test]
    fn test_is_concrete() {
        assert!(parse_type_repr("grid").is_concrete());
        assert!(!parse_type_repr("'a").is_concrete());
        assert!(parse_type_repr("list[object]").is_concrete());
    }

    #[test]
    fn test_cap_for_type_repr() {
        assert_eq!(cap_for_type_repr("grid"), 500);
        assert_eq!(cap_for_type_repr("list[object]"), 100);
        assert_eq!(cap_for_type_repr("object"), 100);
        assert_eq!(cap_for_type_repr("int"), 50);
        assert_eq!(cap_for_type_repr("color"), 20);
        assert_eq!(cap_for_type_repr("bool"), 20);
        assert_eq!(cap_for_type_repr("unknown_type"), 200);
    }
}
