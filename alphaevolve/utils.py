from pathlib import Path


"""Utility functions for AlphaEvolve."""


def write_solution_to_file(code: str, output_file_name: str) -> None:
    """Write solution code to a file with error handling.

    Args:
        code: The solution code to write to the file.
        output_file_name: The name of the output file to be created.

    Raises:
        IOError: If there's an error writing to the file.
    """
    try:
        output_path = Path("exported")
        output_path.joinpath(output_file_name)

        with open(output_path, "w") as f:
            f.write(code)
    except IOError as e:
        raise IOError(f"Failed to write solution to {output_file_name}: {e}") from e
