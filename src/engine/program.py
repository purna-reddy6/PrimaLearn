"""
program.py — Program representation, parsing, and execution.

Programs are compositions of typed primitives from the Spelke DSL.
Represented as expression trees that can be evaluated, serialized,
and analyzed for cross-system composition.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any, Optional
from src.spelke_dsl.base import (
    Primitive, PrimitiveRegistry, SpelkeSystem, Type, Arrow,
    UnificationError, unify, fresh_type_variables,
)


class ProgramNode:
    """Base class for program AST nodes."""

    def evaluate(self, env: dict[str, Any]) -> Any:
        raise NotImplementedError

    def to_str(self) -> str:
        raise NotImplementedError

    def size(self) -> int:
        """Number of nodes in the AST (description length proxy)."""
        raise NotImplementedError

    def primitives_used(self) -> set[str]:
        """Which primitives appear in this program."""
        raise NotImplementedError

    def systems_used(self, registry: PrimitiveRegistry) -> set[SpelkeSystem]:
        """Which Spelke systems are touched by this program."""
        systems = set()
        for name in self.primitives_used():
            if name in registry:
                systems.add(registry[name].system)
        return systems

    def is_cross_system(self, registry: PrimitiveRegistry) -> bool:
        """Does this program compose primitives from multiple Spelke systems?"""
        core_systems = {SpelkeSystem.OBJECTS, SpelkeSystem.FORMS, SpelkeSystem.NUMBER,
                        SpelkeSystem.AGENTS, SpelkeSystem.PERSONS, SpelkeSystem.PLACES}
        used = self.systems_used(registry) & core_systems
        return len(used) > 1

    def __repr__(self):
        return self.to_str()


class PrimNode(ProgramNode):
    """A primitive reference."""

    def __init__(self, name: str, primitive: Primitive):
        self.name = name
        self.primitive = primitive

    def evaluate(self, env: dict[str, Any]) -> Any:
        impl = self.primitive.implementation
        if self.primitive.arity() == 0:
            return impl()
        return impl

    def to_str(self) -> str:
        return self.name

    def size(self) -> int:
        return 1

    def primitives_used(self) -> set[str]:
        return {self.name}


class VarNode(ProgramNode):
    """A variable reference (lambda-bound or environment)."""

    def __init__(self, name: str):
        self.name = name

    def evaluate(self, env: dict[str, Any]) -> Any:
        if self.name not in env:
            raise RuntimeError(f"Unbound variable: {self.name}")
        return env[self.name]

    def to_str(self) -> str:
        return self.name

    def size(self) -> int:
        return 1

    def primitives_used(self) -> set[str]:
        return set()


class AppNode(ProgramNode):
    """Function application: (f x)."""

    def __init__(self, func: ProgramNode, arg: ProgramNode):
        self.func = func
        self.arg = arg

    def evaluate(self, env: dict[str, Any]) -> Any:
        f = self.func.evaluate(env)
        x = self.arg.evaluate(env)
        return f(x)

    def to_str(self) -> str:
        return f"({self.func.to_str()} {self.arg.to_str()})"

    def size(self) -> int:
        return 1 + self.func.size() + self.arg.size()

    def primitives_used(self) -> set[str]:
        return self.func.primitives_used() | self.arg.primitives_used()


class LamNode(ProgramNode):
    """Lambda abstraction: (λ var. body)."""

    def __init__(self, var_name: str, body: ProgramNode):
        self.var_name = var_name
        self.body = body

    def evaluate(self, env: dict[str, Any]) -> Any:
        def closure(val):
            new_env = dict(env)
            new_env[self.var_name] = val
            return self.body.evaluate(new_env)
        return closure

    def to_str(self) -> str:
        return f"(λ{self.var_name}. {self.body.to_str()})"

    def size(self) -> int:
        return 1 + self.body.size()

    def primitives_used(self) -> set[str]:
        return self.body.primitives_used()


class LetNode(ProgramNode):
    """Let binding: let var = expr in body."""

    def __init__(self, var_name: str, expr: ProgramNode, body: ProgramNode):
        self.var_name = var_name
        self.expr = expr
        self.body = body

    def evaluate(self, env: dict[str, Any]) -> Any:
        val = self.expr.evaluate(env)
        new_env = dict(env)
        new_env[self.var_name] = val
        return self.body.evaluate(new_env)

    def to_str(self) -> str:
        return f"(let {self.var_name} = {self.expr.to_str()} in {self.body.to_str()})"

    def size(self) -> int:
        return 1 + self.expr.size() + self.body.size()

    def primitives_used(self) -> set[str]:
        return self.expr.primitives_used() | self.body.primitives_used()


class LitNode(ProgramNode):
    """A literal value (integer, bool, etc.)."""

    def __init__(self, value: Any):
        self.value = value

    def evaluate(self, env: dict[str, Any]) -> Any:
        return self.value

    def to_str(self) -> str:
        return str(self.value)

    def size(self) -> int:
        return 1

    def primitives_used(self) -> set[str]:
        return set()


# ──────────────────────────────────────────────────────────────────────
# Program — wrapper with metadata
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Program:
    """
    A complete program: AST + type + metadata.

    Programs are the unit of synthesis, compression, and library growth.
    """
    root: ProgramNode
    inferred_type: Optional[Type] = None
    log_likelihood: float = 0.0  # log p(program | library)
    log_prior: float = 0.0      # log p(program) under PCFG
    task_id: Optional[str] = None
    source: str = "synthesis"    # "synthesis", "dream", "human"

    def evaluate(self, input_val: Any = None) -> Any:
        """Run the program on an input."""
        env = {}
        if input_val is not None:
            env["input"] = input_val
        result = self.root.evaluate(env)
        # If result is a function (from lambda), apply to input
        if callable(result) and input_val is not None:
            return result(input_val)
        return result

    @property
    def source_code(self) -> str:
        if self.root is None:
            return "<no-ast>"
        return self.root.to_str()

    @property
    def description_length(self) -> int:
        if self.root is None:
            return 0
        return self.root.size()

    def primitives_used(self) -> set[str]:
        if self.root is None:
            return set()
        return self.root.primitives_used()

    def systems_used(self, registry: PrimitiveRegistry) -> set[SpelkeSystem]:
        if self.root is None:
            return set()
        return self.root.systems_used(registry)

    def is_cross_system(self, registry: PrimitiveRegistry) -> bool:
        if self.root is None:
            return False
        return self.root.is_cross_system(registry)

    def to_dict(self) -> dict:
        return {
            "source_code": self.source_code,
            "description_length": self.description_length,
            "log_likelihood": self.log_likelihood,
            "log_prior": self.log_prior,
            "task_id": self.task_id,
            "source": self.source,
            "primitives": sorted(self.primitives_used()),
        }

    def __repr__(self):
        return f"Program({self.source_code})"


# ──────────────────────────────────────────────────────────────────────
# Parser — string → AST
# ──────────────────────────────────────────────────────────────────────

class ProgramParser:
    """Parse s-expression program strings into ASTs."""

    def __init__(self, registry: PrimitiveRegistry):
        self.registry = registry

    def parse(self, source: str) -> Program:
        tokens = self._tokenize(source)
        node, _ = self._parse_expr(tokens, 0)
        return Program(root=node)

    def _tokenize(self, source: str) -> list[str]:
        source = source.replace("(", " ( ").replace(")", " ) ")
        return [t for t in source.split() if t]

    def _parse_expr(self, tokens: list[str], pos: int) -> tuple[ProgramNode, int]:
        if pos >= len(tokens):
            raise ValueError("Unexpected end of program")

        token = tokens[pos]

        if token == "(":
            pos += 1
            # Check for lambda
            if pos < len(tokens) and tokens[pos].startswith("λ"):
                var_name = tokens[pos][1:] if len(tokens[pos]) > 1 else tokens[pos + 1]
                if len(tokens[pos]) > 1:
                    pos += 1
                else:
                    pos += 2
                # Skip the dot if present
                if pos < len(tokens) and tokens[pos] == ".":
                    pos += 1
                body, pos = self._parse_expr(tokens, pos)
                if pos < len(tokens) and tokens[pos] == ")":
                    pos += 1
                return LamNode(var_name, body), pos

            # Check for let
            if pos < len(tokens) and tokens[pos] == "let":
                pos += 1
                var_name = tokens[pos]; pos += 1
                if tokens[pos] == "=": pos += 1
                expr, pos = self._parse_expr(tokens, pos)
                if tokens[pos] == "in": pos += 1
                body, pos = self._parse_expr(tokens, pos)
                if pos < len(tokens) and tokens[pos] == ")":
                    pos += 1
                return LetNode(var_name, expr, body), pos

            # Application: (f x1 x2 ...)
            func, pos = self._parse_expr(tokens, pos)
            while pos < len(tokens) and tokens[pos] != ")":
                arg, pos = self._parse_expr(tokens, pos)
                func = AppNode(func, arg)
            if pos < len(tokens) and tokens[pos] == ")":
                pos += 1
            return func, pos

        # Primitive or variable
        if token in self.registry:
            return PrimNode(token, self.registry[token]), pos + 1

        # Try as integer literal
        try:
            val = int(token)
            return LitNode(val), pos + 1
        except ValueError:
            pass

        # Try as boolean literal
        if token.lower() in ("true", "#t"):
            return LitNode(True), pos + 1
        if token.lower() in ("false", "#f"):
            return LitNode(False), pos + 1

        # Variable
        return VarNode(token), pos + 1


# ──────────────────────────────────────────────────────────────────────
# Program builder helpers
# ──────────────────────────────────────────────────────────────────────

def make_program(registry: PrimitiveRegistry, prim_name: str, *args) -> Program:
    """Quick program construction from a primitive and literal args."""
    prim = registry[prim_name]
    node = PrimNode(prim_name, prim)
    for arg in args:
        if isinstance(arg, ProgramNode):
            arg_node = arg
        elif isinstance(arg, Program):
            arg_node = arg.root
        else:
            arg_node = LitNode(arg)
        node = AppNode(node, arg_node)
    return Program(root=node)
