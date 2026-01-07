"""Utility functions for AlphaEvolve."""


def write_solution_to_file(code: str, output_path: str) -> None:
    """Write solution code to a file with error handling.

    Args:
        code: The solution code to write to the file.
        output_path: The path where the file should be created.

    Raises:
        IOError: If there's an error writing to the file.
    """
    try:
        with open(output_path, "w") as f:
            f.write(code)
    except IOError as e:
        raise IOError(f"Failed to write solution to {output_path}: {e}") from e
