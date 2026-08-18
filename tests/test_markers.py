import pytest

from alphaevolve.task.markers import (
    EVOLVE,
    SKELETON,
    MarkerError,
    evolve_regions,
    parse_program,
    reassemble,
)

PROGRAM = """import math

# EVOLVE-BLOCK-START
def f(x):
    return x * 2
# EVOLVE-BLOCK-END

def main():
    print(f(3))
"""


def test_roundtrip_byte_exact():
    assert reassemble(parse_program(PROGRAM)) == PROGRAM


def test_roundtrip_no_trailing_newline():
    code = "# EVOLVE-BLOCK-START\nx = 1\n# EVOLVE-BLOCK-END"
    assert reassemble(parse_program(code)) == code


def test_segment_kinds_alternate():
    segments = parse_program(PROGRAM)
    kinds = [s.kind for s in segments]
    assert kinds == [SKELETON, EVOLVE, SKELETON]
    assert segments[1].text == "def f(x):\n    return x * 2\n"


def test_markers_belong_to_skeleton():
    segments = parse_program(PROGRAM)
    assert segments[0].text.endswith("# EVOLVE-BLOCK-START\n")
    assert segments[2].text.startswith("# EVOLVE-BLOCK-END\n")


def test_multiple_blocks():
    code = (
        "# EVOLVE-BLOCK-START\na = 1\n# EVOLVE-BLOCK-END\n"
        "mid\n"
        "# EVOLVE-BLOCK-START\nb = 2\n# EVOLVE-BLOCK-END\n"
    )
    segments = parse_program(code)
    assert [s.kind for s in segments].count(EVOLVE) == 2
    assert reassemble(segments) == code


def test_indented_markers():
    code = "def g():\n    # EVOLVE-BLOCK-START\n    y = 1\n    # EVOLVE-BLOCK-END\n    return y\n"
    segments = parse_program(code)
    assert segments[1].text == "    y = 1\n"


def test_nested_start_rejected():
    code = "# EVOLVE-BLOCK-START\n# EVOLVE-BLOCK-START\n# EVOLVE-BLOCK-END\n"
    with pytest.raises(MarkerError, match="nested"):
        parse_program(code)


def test_end_without_start_rejected():
    with pytest.raises(MarkerError, match="without matching"):
        parse_program("x = 1\n# EVOLVE-BLOCK-END\n")


def test_unterminated_rejected():
    with pytest.raises(MarkerError, match="unterminated"):
        parse_program("# EVOLVE-BLOCK-START\nx = 1\n")


def test_no_markers_is_all_skeleton():
    segments = parse_program("x = 1\n")
    assert [s.kind for s in segments] == [SKELETON]


def test_evolve_regions_offsets():
    regions = evolve_regions(PROGRAM)
    assert len(regions) == 1
    start, end = regions[0]
    assert PROGRAM[start:end] == "def f(x):\n    return x * 2\n"
