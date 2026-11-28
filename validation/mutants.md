# Mutation testing

`cargo mutants` 26.1.0, on `src/projection.rs` and `src/tangent.rs`.

```
cargo mutants -f src/projection.rs -f src/tangent.rs -j 4 --timeout 180 --baseline skip --copy-target true -- --features capi --lib --test formal_mirror climbing tangent projection improved
```

148 mutants tested. 106 caught. 42 unviable, because the replacement did not compile. 0 missed. There were no survivors and no timeouts.
