You are improving an optimizer that finds low-rank decompositions of the
matrix-multiplication tensor ⟨m,n,p⟩: the (m·n)×(n·p)×(p·m) binary tensor T
with T[i·n+j, j·p+k, k·m+i] = 1. A rank-R decomposition T = Σ_{r=1..R}
u_r ⊗ v_r ⊗ w_r is exactly an algorithm multiplying an m×n by an n×p matrix
with R scalar multiplications. Lower exact rank = faster algorithm.

Known landmarks: ⟨2,2,2⟩ has rank 7 (Strassen 1969); ⟨3,3,3⟩ best known 23;
⟨4,4,4⟩ had rank 49 (Strassen squared) until a rank-48 decomposition over the
complex half-integers was found (AlphaEvolve, 2025).

The evaluator rounds your factor entries to the nearest half-integer (real or
complex) and verifies T exactly in integer arithmetic — near-misses score
only through the auxiliary loss signal, so techniques that pull solutions
onto the half-integer grid matter: regularization toward discrete values,
rounding-aware penalties, cyclic-symmetry exploitation (⟨n,n,n⟩ is invariant
under cycling U→V→W→U), restarts from perturbed rounded solutions, adding
noise to escape plateaus, decreasing rank by warm-starting from a found
solution with one column removed.

Budget discipline: `iters` is the per-call optimization budget and the rank
descent loop shares it — wasted iterations on hopeless configurations cost
the chance to verify lower ranks.
