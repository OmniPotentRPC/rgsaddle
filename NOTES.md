# Direction-cover sanity measurements

Command:

```sh
cargo +1.88.0 test --release --test direction_cover quick_direction_cover_sanity -- --ignored --nocapture
```

The test uses 12 arms on a linear-cost analytic independent-well surface. “Leaving start” counts connected arms with a product; repeated hits on the same saddle count as one saddle. The soft-mode baseline displaces by the same 0.8 length and quenches.

| Atoms | Method | Arms leaving start | Saddles found | Force calls / saddle | ms / arm |
|---:|---|---:|---:|---:|---:|
| 75 | direction cover | 4 / 12 | 1 | 2564 | 0.871 |
| 75 | soft mode | 0 / 12 | 0 | n/a | 0.195 |
| 500 | direction cover | 4 / 12 | 1 | 2564 | 4.387 |
| 500 | soft mode | 0 / 12 | 0 | n/a | 0.959 |
| 4000 | direction cover | 4 / 12 | 1 | 2564 | 37.939 |
| 4000 | soft mode | 0 / 12 | 0 | n/a | 7.929 |

These are quick sanity observations from one run, not performance claims.
