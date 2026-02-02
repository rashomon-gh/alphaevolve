"""
Test script to verify Python syntax of all AlphaEvolve modules.
"""
import ast
import sys
from pathlib import Path


def test_syntax(filepath):
    """Test Python syntax of a file."""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)


def main():
    """Test syntax of all Python files in the project."""
    print("="*70)
    print("AlphaEvolve: Syntax Validation")
    print("="*70)
    
    # Find all Python files
    project_root = Path("/home/shawon/Projects/alphaevolve")
    python_files = list(project_root.glob("*.py")) + list(project_root.glob("**/*.py"))
    
    # Sort files
    python_files = sorted(python_files)
    
    print(f"\nFound {len(python_files)} Python files to check\n")
    
    results = []
    for filepath in python_files:
        rel_path = filepath.relative_to(project_root)
        success, error = test_syntax(filepath)
        results.append((rel_path, success, error))
        
        status = "✓" if success else "✗"
        print(f"{status} {rel_path}")
        
        if error:
            print(f"  Error: {error}")
    
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed
    
    print(f"Passed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    
    if failed > 0:
        print("\nFailed files:")
        for rel_path, success, error in results:
            if not success:
                print(f"  - {rel_path}: {error}")
        sys.exit(1)
    else:
        print("\nAll files have valid syntax! ✓")
        sys.exit(0)


if __name__ == "__main__":
    main()
