"""Unit tests for the program_validator module."""
import pytest
from pathlib import Path
import tempfile
from alphaevolve.program_validator import (
    ProgramValidator,
    validate_program,
    validate_program_file,
)


class TestProgramValidator:
    """Test suite for ProgramValidator class."""
    
    @pytest.fixture
    def validator(self):
        """Create a ProgramValidator instance for testing."""
        return ProgramValidator()
    
    # Test validate_syntax method
    def test_validate_syntax_valid_code(self, validator):
        """Test validation of syntactically valid Python code."""
        valid_code = """
def add(a, b):
    return a + b

result = add(1, 2)
"""
        is_valid, error = validator.validate_syntax(valid_code)
        assert is_valid is True
        assert error is None
    
    def test_validate_syntax_invalid_code(self, validator):
        """Test validation of syntactically invalid Python code."""
        invalid_code = """
def add(a, b)
    return a + b
"""
        is_valid, error = validator.validate_syntax(invalid_code)
        assert is_valid is False
        assert error is not None
        assert "Syntax Error" in error
    
    def test_validate_syntax_missing_parenthesis(self, validator):
        """Test validation detects missing parenthesis."""
        invalid_code = "print('hello'"
        is_valid, error = validator.validate_syntax(invalid_code)
        assert is_valid is False
        assert error is not None
    
    def test_validate_syntax_indentation_error(self, validator):
        """Test validation detects indentation errors."""
        invalid_code = """
def test():
x = 1
"""
        is_valid, error = validator.validate_syntax(invalid_code)
        assert is_valid is False
        assert error is not None
    
    def test_validate_syntax_empty_string(self, validator):
        """Test validation of empty code."""
        is_valid, error = validator.validate_syntax("")
        assert is_valid is True
        assert error is None
    
    # Test validate_file method
    def test_validate_file_existing_valid(self, validator, tmp_path):
        """Test validation of an existing valid file."""
        test_file = tmp_path / "valid.py"
        valid_code = """
def multiply(x, y):
    return x * y
"""
        test_file.write_text(valid_code)
        
        is_valid, error = validator.validate_file(test_file)
        assert is_valid is True
        assert error is None
    
    def test_validate_file_existing_invalid(self, validator, tmp_path):
        """Test validation of an existing invalid file."""
        test_file = tmp_path / "invalid.py"
        invalid_code = """
def test()
    pass
"""
        test_file.write_text(invalid_code)
        
        is_valid, error = validator.validate_file(test_file)
        assert is_valid is False
        assert error is not None
    
    def test_validate_file_not_found(self, validator):
        """Test validation of a non-existent file."""
        test_file = Path("/nonexistent/path/file.py")
        is_valid, error = validator.validate_file(test_file)
        assert is_valid is False
        assert "File not found" in error
    
    # Test validate_structure method
    def test_validate_structure_valid_code(self, validator):
        """Test structure analysis of valid code."""
        code = """
import os
from sys import path

class Calculator:
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b

def helper_function():
    pass

x = 10
"""
        structure = validator.validate_structure(code)
        assert structure['valid'] is True
        assert len(structure['classes']) == 1
        assert structure['classes'][0]['name'] == 'Calculator'
        assert len(structure['classes'][0]['methods']) == 2
        assert len(structure['functions']) == 1
        assert structure['functions'][0]['name'] == 'helper_function'
        assert len(structure['imports']) == 2
        assert 'os' in structure['imports']
        assert 'sys.path' in structure['imports']
    
    def test_validate_structure_no_classes_functions(self, validator):
        """Test structure analysis of code with no classes or functions."""
        code = """
x = 10
y = 20
"""
        structure = validator.validate_structure(code)
        assert structure['valid'] is True
        assert len(structure['classes']) == 0
        assert len(structure['functions']) == 0
        assert structure['global_statements'] == 2
    
    def test_validate_structure_invalid_code(self, validator):
        """Test structure analysis of invalid code."""
        invalid_code = "def broken("
        structure = validator.validate_structure(invalid_code)
        assert structure['valid'] is False
        assert 'error' in structure
    
    def test_validate_structure_multiple_classes(self, validator):
        """Test structure analysis with multiple classes."""
        code = """
class ClassA:
    def method_a(self):
        pass

class ClassB:
    def method_b(self):
        pass
"""
        structure = validator.validate_structure(code)
        assert structure['valid'] is True
        assert len(structure['classes']) == 2
    
    # Test validate_program method
    def test_validate_program_valid(self, validator):
        """Test comprehensive validation of a valid program."""
        valid_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        result = validator.validate_program(valid_code)
        assert result['valid'] is True
        assert result['syntax_valid'] is True
        assert len(result['errors']) == 0
        assert result['structure'] is not None
    
    def test_validate_program_invalid_syntax(self, validator):
        """Test comprehensive validation of a program with syntax errors."""
        invalid_code = "def broken("
        result = validator.validate_program(invalid_code)
        assert result['valid'] is False
        assert result['syntax_valid'] is False
        assert len(result['errors']) > 0
    
    def test_validate_program_with_filepath(self, validator, tmp_path):
        """Test program validation with filepath for error reporting."""
        invalid_code = "x ="
        filepath = tmp_path / "test.py"
        
        result = validator.validate_program(invalid_code, filepath)
        assert result['valid'] is False
        assert len(result['errors']) > 0
    
    def test_validate_program_structure_only(self, validator):
        """Test that structure analysis works on valid programs."""
        code = """
class TestClass:
    def test_method(self):
        return 42

def standalone_function():
    return "hello"
"""
        result = validator.validate_program(code)
        assert result['valid'] is True
        assert result['structure'] is not None
        assert len(result['structure']['classes']) == 1
        assert len(result['structure']['functions']) == 1


class TestConvenienceFunctions:
    """Test suite for convenience functions."""
    
    def test_validate_program_valid(self):
        """Test validate_program convenience function with valid code."""
        code = """
def greet(name):
    return f"Hello, {name}!"
"""
        is_valid, error = validate_program(code)
        assert is_valid is True
        assert error is None
    
    def test_validate_program_invalid(self):
        """Test validate_program convenience function with invalid code."""
        code = """
def broken(
    pass
"""
        is_valid, error = validate_program(code)
        assert is_valid is False
        assert error is not None
    
    def test_validate_program_file_valid(self, tmp_path):
        """Test validate_program_file convenience function with valid file."""
        test_file = tmp_path / "valid.py"
        test_file.write_text("""
def square(x):
    return x ** 2
""")
        is_valid, error = validate_program_file(test_file)
        assert is_valid is True
        assert error is None
    
    def test_validate_program_file_invalid(self, tmp_path):
        """Test validate_program_file convenience function with invalid file."""
        test_file = tmp_path / "invalid.py"
        test_file.write_text("""
def broken()
    pass
""")
        is_valid, error = validate_program_file(test_file)
        assert is_valid is False
        assert error is not None
    
    def test_validate_program_file_not_found(self):
        """Test validate_program_file with non-existent file."""
        test_file = Path("/nonexistent/file.py")
        is_valid, error = validate_program_file(test_file)
        assert is_valid is False
        assert "File not found" in error


class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_complex_code_with_imports(self):
        """Test validation of complex code with various imports."""
        code = """
import os
import sys
from typing import List, Dict, Optional
from pathlib import Path

class ComplexClass:
    def __init__(self, value: int):
        self.value = value
    
    def process(self, items: List[int]) -> Dict[str, int]:
        result = {}
        for item in items:
            result[str(item)] = item * 2
        return result
    
    @staticmethod
    def static_method():
        return "static"
    
    @classmethod
    def class_method(cls):
        return cls.__name__
"""
        validator = ProgramValidator()
        result = validator.validate_program(code)
        assert result['valid'] is True
        # Count each import name separately (List, Dict, Optional are separate imports)
        assert len(result['structure']['imports']) == 6
        assert len(result['structure']['classes']) == 1
    
    def test_code_with_decorators(self):
        """Test validation of code with decorators."""
        code = """
def decorator(func):
    return func

@decorator
def decorated_function():
    pass
"""
        validator = ProgramValidator()
        result = validator.validate_program(code)
        assert result['valid'] is True
    
    def test_code_with_nested_functions(self):
        """Test validation of code with nested functions."""
        code = """
def outer():
    def inner():
        return "inner"
    return inner()
"""
        validator = ProgramValidator()
        result = validator.validate_program(code)
        assert result['valid'] is True
    
    def test_code_with_lambda(self):
        """Test validation of code with lambda functions."""
        code = """
square = lambda x: x ** 2
apply = lambda f, x: f(x)
"""
        validator = ProgramValidator()
        result = validator.validate_program(code)
        assert result['valid'] is True
    
    def test_code_with_async_functions(self):
        """Test validation of code with async functions."""
        code = """
import asyncio

async def async_function():
    await asyncio.sleep(1)
    return "done"
"""
        validator = ProgramValidator()
        result = validator.validate_program(code)
        assert result['valid'] is True
    
    def test_code_with_docstrings(self):
        """Test validation of code with docstrings."""
        code = """
'''Module docstring.'''

def documented_function():
    '''Function docstring.'''
    pass

class DocumentedClass:
    '''Class docstring.'''
    
    def method(self):
        '''Method docstring.'''
        pass
"""
        validator = ProgramValidator()
        result = validator.validate_program(code)
        assert result['valid'] is True
    
    def test_code_with_type_hints(self):
        """Test validation of code with type hints."""
        code = """
from typing import List, Dict, Optional, Tuple

def typed_function(
    x: int,
    y: str,
    items: List[float],
    config: Optional[Dict[str, str]] = None
) -> Tuple[int, str]:
    return (x, y)
"""
        validator = ProgramValidator()
        result = validator.validate_program(code)
        assert result['valid'] is True
    
    def test_code_with_syntax_error_in_middle(self):
        """Test validation detects errors in the middle of code."""
        code = """
def first_function():
    return 1

def broken_function(
    pass

def last_function():
    return 3
"""
        validator = ProgramValidator()
        result = validator.validate_program(code)
        assert result['valid'] is False
        assert len(result['errors']) > 0
