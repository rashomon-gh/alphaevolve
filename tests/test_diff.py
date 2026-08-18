import pytest

from alphaevolve.generation.diff import (
    DiffBlock,
    DiffError,
    apply_diff,
    apply_full_rewrite,
    extract_code_block,
    parse_diff,
)

PROGRAM = """import math

# EVOLVE-BLOCK-START
def f(x):
    return x * 2
# EVOLVE-BLOCK-END

def main():
    print(f(3))
"""


def make_diff(search: str, replace: str) -> str:
    return f"<<<<<<< SEARCH\n{search}=======\n{replace}>>>>>>> REPLACE"


# -- parsing ---------------------------------------------------------------


def test_parse_single_block():
    text = "Some explanation.\n" + make_diff("a\n", "b\n") + "\ntrailing prose"
    blocks = parse_diff(text)
    assert blocks == [DiffBlock(search="a\n", replace="b\n")]


def test_parse_multiple_blocks_ordered():
    text = make_diff("one\n", "1\n") + "\n\n" + make_diff("two\n", "2\n")
    blocks = parse_diff(text)
    assert [b.search for b in blocks] == ["one\n", "two\n"]


def test_parse_no_blocks_raises():
    with pytest.raises(DiffError) as exc:
        parse_diff("I think you should refactor the function.")
    assert exc.value.reason == "parse"


def test_parse_inside_code_fence():
    text = "```\n" + make_diff("a\n", "b\n") + "\n```"
    assert len(parse_diff(text)) == 1


# -- application -----------------------------------------------------------


def test_apply_inside_evolve_block():
    diff = [DiffBlock(search="    return x * 2\n", replace="    return x * 3\n")]
    result = apply_diff(PROGRAM, diff)
    assert "x * 3" in result
    assert result.count("EVOLVE-BLOCK") == 2


def test_apply_sequential_blocks_see_prior_edits():
    blocks = [
        DiffBlock(search="    return x * 2\n", replace="    y = x * 2\n    return y\n"),
        DiffBlock(search="    return y\n", replace="    return y + 1\n"),
    ]
    result = apply_diff(PROGRAM, blocks)
    assert "return y + 1" in result


def test_apply_not_found():
    with pytest.raises(DiffError) as exc:
        apply_diff(PROGRAM, [DiffBlock(search="nonexistent\n", replace="x\n")])
    assert exc.value.reason == "not_found"


def test_apply_ambiguous():
    code = "# EVOLVE-BLOCK-START\nx = 1\nx = 1\n# EVOLVE-BLOCK-END\n"
    with pytest.raises(DiffError) as exc:
        apply_diff(code, [DiffBlock(search="x = 1\n", replace="x = 2\n")])
    assert exc.value.reason == "ambiguous"


def test_apply_outside_evolve_block_rejected():
    with pytest.raises(DiffError) as exc:
        apply_diff(PROGRAM, [DiffBlock(search="    print(f(3))\n", replace="    pass\n")])
    assert exc.value.reason == "outside_evolve_block"


def test_apply_spanning_boundary_rejected():
    # Search text covers evolve content AND the END marker line -> outside.
    search = "    return x * 2\n# EVOLVE-BLOCK-END\n"
    with pytest.raises(DiffError) as exc:
        apply_diff(PROGRAM, [DiffBlock(search=search, replace="nope\n")])
    assert exc.value.reason == "outside_evolve_block"


def test_apply_marker_rewrite_rejected():
    with pytest.raises(DiffError) as exc:
        apply_diff(PROGRAM, [DiffBlock(search="# EVOLVE-BLOCK-END\n", replace="\n")])
    assert exc.value.reason == "outside_evolve_block"


def test_apply_empty_search_rejected():
    with pytest.raises(DiffError) as exc:
        apply_diff(PROGRAM, [DiffBlock(search="", replace="x\n")])
    assert exc.value.reason == "empty_search"


def test_skeleton_never_changes():
    diff = [DiffBlock(search="def f(x):\n    return x * 2\n", replace="def f(x):\n    return 0\n")]
    result = apply_diff(PROGRAM, diff)
    before, _, _ = PROGRAM.partition("# EVOLVE-BLOCK-START")
    after_before, _, _ = result.partition("# EVOLVE-BLOCK-START")
    assert before == after_before
    assert result.endswith("def main():\n    print(f(3))\n")


# -- full rewrite ----------------------------------------------------------


def test_full_rewrite_replaces_block_only():
    result = apply_full_rewrite(PROGRAM, "def f(x):\n    return x ** 2\n")
    assert "x ** 2" in result
    assert result.startswith("import math")
    assert result.endswith("def main():\n    print(f(3))\n")


def test_full_rewrite_adds_trailing_newline():
    result = apply_full_rewrite(PROGRAM, "def f(x):\n    return 1")
    assert "return 1\n# EVOLVE-BLOCK-END" in result


def test_full_rewrite_requires_single_block():
    code = (
        "# EVOLVE-BLOCK-START\na = 1\n# EVOLVE-BLOCK-END\n"
        "# EVOLVE-BLOCK-START\nb = 2\n# EVOLVE-BLOCK-END\n"
    )
    with pytest.raises(DiffError) as exc:
        apply_full_rewrite(code, "c = 3\n")
    assert exc.value.reason == "full_rewrite_ambiguous"


def test_replacement_injecting_marker_rejected():
    # An LLM that emits marker lines in the replacement would silently
    # reshape the evolve-block structure of every descendant program.
    diff = [
        DiffBlock(
            search="    return x * 2\n",
            replace="    return x * 3\n# EVOLVE-BLOCK-END\nevil = 1\n# EVOLVE-BLOCK-START\n",
        )
    ]
    with pytest.raises(DiffError) as exc:
        apply_diff(PROGRAM, diff)
    assert exc.value.reason == "marker_injection"


def test_full_rewrite_injecting_marker_rejected():
    with pytest.raises(DiffError) as exc:
        apply_full_rewrite(PROGRAM, "x = 1\n# EVOLVE-BLOCK-END\nevil = 2\n")
    assert exc.value.reason == "marker_injection"


def test_extract_code_block():
    assert extract_code_block("prose\n```python\nx = 1\n```\nmore") == "x = 1\n"
    assert extract_code_block("no fences here") == "no fences here"
