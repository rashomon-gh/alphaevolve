"""EVOLVE-BLOCK marker parsing (paper §2.1).

A program is an alternating sequence of immutable skeleton segments and
evolvable segments. The marker lines themselves belong to the skeleton, so
only text strictly between START and END markers may ever change.
"""

from __future__ import annotations

from dataclasses import dataclass

START_MARKER = "# EVOLVE-BLOCK-START"
END_MARKER = "# EVOLVE-BLOCK-END"

SKELETON = "skeleton"
EVOLVE = "evolve"


class MarkerError(ValueError):
    pass


@dataclass(frozen=True)
class Segment:
    kind: str  # SKELETON | EVOLVE
    text: str


def _is_marker(line: str, marker: str) -> bool:
    return line.strip().startswith(marker)


def parse_program(code: str) -> list[Segment]:
    """Split code into alternating skeleton/evolve segments. Reassembly of the
    returned segments is byte-exact (``reassemble(parse_program(c)) == c``)."""
    segments: list[Segment] = []
    current: list[str] = []
    in_evolve = False
    for line in code.splitlines(keepends=True):
        if _is_marker(line, START_MARKER):
            if in_evolve:
                raise MarkerError("nested EVOLVE-BLOCK-START")
            current.append(line)  # marker line is skeleton
            segments.append(Segment(SKELETON, "".join(current)))
            current = []
            in_evolve = True
        elif _is_marker(line, END_MARKER):
            if not in_evolve:
                raise MarkerError("EVOLVE-BLOCK-END without matching START")
            segments.append(Segment(EVOLVE, "".join(current)))
            current = [line]  # marker line starts the next skeleton segment
            in_evolve = False
        else:
            current.append(line)
    if in_evolve:
        raise MarkerError("unterminated EVOLVE-BLOCK (missing END marker)")
    if current or not segments:
        segments.append(Segment(SKELETON, "".join(current)))
    return segments


def reassemble(segments: list[Segment]) -> str:
    return "".join(seg.text for seg in segments)


def evolve_regions(code: str) -> list[tuple[int, int]]:
    """Character-offset spans ``[start, end)`` of the evolvable text."""
    regions: list[tuple[int, int]] = []
    offset = 0
    for seg in parse_program(code):
        if seg.kind == EVOLVE:
            regions.append((offset, offset + len(seg.text)))
        offset += len(seg.text)
    return regions
