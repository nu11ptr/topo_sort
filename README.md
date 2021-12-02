# depends

A "cycle-safe" topological sort for a set of nodes with dependencies in Rust.
Basically, it allows sorting a list by its dependencies while checking for
cycles in the graph. If a cycle is detected, a `CycleError` is returned from the
iterator.

## Usage

```toml
[dependencies]
depends = "0.1"
```

A basic example:

```rust
let mut depends = Depends::with_capacity(5);
depends.insert("C", vec!["A", "B"]); // read: "C" depends on "A" and "B"
depends.insert("E", vec!["B", "C"]);
depends.insert("A", vec![]);
depends.insert("D", vec!["A", "C", "E"]);
depends.insert("B", vec!["A"]);

assert_eq!(
    vec!["A", "B", "C", "E", "D"],
    depends.to_owned_vec().unwrap()
);
```

Cycle detected:

```rust
let mut depends = Depends::with_capacity(3);
depends.insert(1, vec![2, 3]);
depends.insert(2, vec![3]);
depends.insert(3, vec![1]); // cycle

assert!(depends.to_vec().is_err())
```

## Algorithm

This is implemented
using [Kahn's algorithm](https://en.wikipedia.org/wiki/Topological_sorting).
While basic caution was taken not to do anything too egregious performance-wise,
the author's use cases are not performance sensitive, and it has not been
optimized in any way.

## Maintenance

The author will make basic changes to keep the crate updated to ensure it stays
compatible with future stable Rust, etc. but no further functionality
enhancements are likely. The crate meets the needs of the author and is unlikely
to get significant new features.
