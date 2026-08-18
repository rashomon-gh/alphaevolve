You are improving a SEARCH HEURISTIC that packs exactly n circles into the
unit square maximizing the SUM OF RADII (not the common equal-radius
variant). `improve(circles, n_circles, rng, budget_s)` receives the best
known valid packing (rows of x, y, r) and a wall-clock budget; return a valid
packing with an equal-or-larger radius sum. Best packings persist between
generations — continue refining where earlier generations stopped.

Reference point: for n=26 the paper (App. B.12) reports sum of radii 2.635,
improving the previous 2.634. Unequal radii matter: strong packings mix a few
large circles with small gap-fillers.

Useful ideas: physics-style relaxation (push overlapping circles apart, grow
radii to contact), Apollonius-style gap filling (place a new circle tangent
to two circles and a wall), pattern restarts (hexagonal cores, corner-anchored
big circles), simulated annealing over center positions with radii computed
greedily from contacts, gradient ascent on a soft objective then exact repair.

Validity is exact and strict: shape must be (n, 3), disks inside [0,1]², no
overlaps beyond 1e-9. An invalid return scores nothing — always re-validate
and fall back to the incoming packing if your improvement step failed.
