# Direction-cover round 2 sanity measurements

Command:

```sh
cargo +1.88.0 test --release --test direction_cover quick_direction_cover_sanity -- --ignored --nocapture
```

The test uses 12 arms on a linear-cost analytic independent-well surface. “Leaving start” counts connected arms with a product; repeated hits on the same saddle count as one saddle. The soft-mode baseline displaces by the same 0.8 length and quenches.

| Atoms | Method | Arms leaving start | Saddles found | Force calls / saddle | ms / arm |
|---:|---|---:|---:|---:|---:|
| 75 | direction cover | 3 / 12 | 1 | 2681 | 0.888 |
| 75 | soft mode | 0 / 12 | 0 | n/a | 0.173 |
| 500 | direction cover | 3 / 12 | 1 | 2681 | 4.294 |
| 500 | soft mode | 0 / 12 | 0 | n/a | 0.884 |
| 4000 | direction cover | 3 / 12 | 1 | 2681 | 36.322 |
| 4000 | soft mode | 0 / 12 | 0 | n/a | 7.244 |

These are quick sanity observations from one run, not performance claims.
