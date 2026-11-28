# BenchBot question

Run `benchmarks/pafi-swinburne-repro/pafi-swinburne-repro` from the repository
root on the full setting: the public example's 9 planes, 100 independent trajectories per plane,
2000 thermalization steps, and 5000 production steps at 1000 and 2000 K,
after the rgsaddle NEB converges.

Do all three W-vacancy barriers agree with the exact values printed in
`tomswinburne/pafi@1661a1d:doc/test_output.png` under the predeclared combined
two-standard-error plus half-display-unit criterion? Report the generated
CSV, NEB iteration count and final force, wall time, rejected basin
stretches, and any failed row. Do not tune the protocol or acceptance band
after seeing the result.
