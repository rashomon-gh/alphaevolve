"""
Program Validator Module for AlphaEvolve.

This module provides utilities to validate Python syntax and structure of programs.
It can be used to verify submitted programs before execution or evaluation.
"""

import ast
from pathlib import Path
from typing import Tuple, Optional, Dict, Any


class ProgramValidator:
    """Validator for Python programs."""

    def __init__(self):
        """Initialize the ProgramValidator."""
        self.errors = []

    def validate_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Python syntax of a code string.

        Args:
            code: Python code as a string

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            error_msg = f"Syntax Error: {e.msg} at line {e.lineno}"
            if e.offset:
                error_msg += f", column {e.offset}"
            return False, error_msg

    def validate_file(self, filepath: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate Python syntax of a file.

        Args:
            filepath: Path to the Python file

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                code = f.read()
            return self.validate_syntax(code)
        except FileNotFoundError:
            return False, f"File not found: {filepath}"
        except IOError as e:
            return False, f"IO Error reading file: {e}"

    def validate_structure(self, code: str) -> Dict[str, Any]:
        """
        Analyze the structure of a Python program.

        Args:
            code: Python code as a string

        Returns:
            Dictionary containing structural information about the program
        """
        try:
            tree = ast.parse(code)

            structure = {
                "valid": True,
                "classes": [],
                "functions": [],
                "imports": [],
                "global_statements": 0,
            }

            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    structure["classes"].append(
                        {
                            "name": node.name,
                            "methods": [
                                n.name
                                for n in node.body
                                if isinstance(n, ast.FunctionDef)
                            ],
                            "lineno": node.lineno,
                        }
                    )
                elif isinstance(node, ast.FunctionDef):
                    structure["functions"].append(
                        {"name": node.name, "lineno": node.lineno}
                    )
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        structure["imports"].extend(
                            [alias.name for alias in node.names]
                        )
                    else:
                        module = node.module or ""
                        structure["imports"].extend(
                            f"{module}.{alias.name}" for alias in node.names
                        )
                else:
                    structure["global_statements"] += 1

            return structure
        except SyntaxError as e:
            return {
                "valid": False,
                "error": f"Syntax Error: {e.msg} at line {e.lineno}",
            }

    def validate_program(
        self, code: str, filepath: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of a Python program.

        Args:
            code: Python code as a string
            filepath: Optional filepath for error reporting

        Returns:
            Dictionary containing validation results
        """
        result = {
            "valid": True,
            "syntax_valid": True,
            "structure": None,
            "errors": [],
            "warnings": [],
        }

        # Validate syntax
        syntax_valid, syntax_error = self.validate_syntax(code)
        result["syntax_valid"] = syntax_valid

        if not syntax_valid:
            result["valid"] = False
            if filepath:
                result["errors"].append(f"{filepath}: {syntax_error}")
            else:
                result["errors"].append(syntax_error)
            return result

        # Analyze structure
        structure = self.validate_structure(code)
        result["structure"] = structure

        if not structure["valid"]:
            result["valid"] = False
            result["errors"].append(structure["error"])

        return result


def validate_program(code: str) -> Tuple[bool, Optional[str]]:
    """
    Convenience function to validate a program's syntax.

    Args:
        code: Python code as a string

    Returns:
        Tuple of (is_valid, error_message)
    """
    validator = ProgramValidator()
    return validator.validate_syntax(code)


def validate_program_file(filepath: Path) -> Tuple[bool, Optional[str]]:
    """
    Convenience function to validate a program file's syntax.

    Args:
        filepath: Path to the Python file

    Returns:
        Tuple of (is_valid, error_message)
    """
    validator = ProgramValidator()
    return validator.validate_file(filepath)
