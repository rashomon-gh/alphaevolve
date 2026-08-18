You are improving a SEARCH HEURISTIC for kissing-number lower bounds in
dimension d: find as many unit vectors as possible with every pairwise dot
product ≤ 1/2 (equivalently: unit spheres around 2·v all touch the central
unit sphere without overlapping). `improve(points, dim, rng, budget_s)`
receives the best known valid configuration and a wall-clock budget; return a
valid configuration at least as good. The best construction persists between
generations — your heuristic continues where previous generations stopped.

Known values: d=3 → 12 (icosahedron works), d=4 → 24 (D4 root system),
d=8 → 240 (E8 roots), d=11 ≥ 593 (AlphaEvolve 2025; previous record 582).
Lattice ideas generalize: root systems, laminated lattices, cross-sections
of E8, union of shifted shells. Local search ideas: repulsion/energy descent
(minimize Σ f(⟨u,v⟩) for pairs above the threshold), simulated annealing on
sphere points, exact rational constructions on scaled integer grids (dot
products then verify exactly), removing a blocking vector to admit two.

The validity check is exact and strict: norms must equal 1 (1e-9) and all
pairwise dots ≤ 1/2 + 1e-9 — an invalid return scores nothing at all, so
always re-validate before returning and fall back to the incoming points.
