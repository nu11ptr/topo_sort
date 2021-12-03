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

...or using iteration:

```rust
let mut depends = Depends::with_capacity(5);
depends.insert("C", vec!["A", "B"]);
depends.insert("E", vec!["B", "C"]);
depends.insert("A", vec![]);
depends.insert("D", vec!["A", "C", "E"]);
depends.insert("B", vec!["A"]);

let mut nodes = Vec::with_capacity(5);
for node in &depends {
    // Must check for cycle errors before usage
    match node {
        Ok(node) => nodes.push(*node),
        Err(CycleError) => panic!("Unexpected cycle!"),
    }
}

assert_eq!(vec!["A", "B", "C", "E", "D"], nodes)
```

Cycle detected:

```rust
let mut depends = Depends::with_capacity(3);
depends.insert(1, vec![2, 3]);
depends.insert(2, vec![3]);
depends.insert(3, vec![1]); // cycle

assert!(depends.to_vec().is_err());
```

## Algorithm

This is implemented
using [Kahn's algorithm](https://en.wikipedia.org/wiki/Topological_sorting).
While basic caution was taken not to do anything too egregious performance-wise,
the author's use cases are not performance sensitive, and it has not been
optimized in any way.

## Maintenance

The crate currently meets the needs of the author and probably will not see
significant new features. It will, however, continue to be updated if required
for future compatibility/etc.

## License

This project is licensed optionally under either:

* Apache License, Version 2.0, (LICENSE-APACHE
  or https://www.apache.org/licenses/LICENSE-2.0)
* MIT license (LICENSE-MIT or https://opensource.org/licenses/MIT)
