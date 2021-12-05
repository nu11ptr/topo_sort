# topo_sort

[![Crate](https://img.shields.io/crates/v/topo_sort?style=for-the-badge)](https://crates.io/crates/topo_sort)
[![Docs](https://img.shields.io/docsrs/topo_sort?style=for-the-badge)](https://docs.rs/topo_sort)

A "cycle-safe" topological sort for a set of nodes with dependencies in Rust.
Basically, it allows sorting a list by its dependencies while checking for
cycles in the graph. If a cycle is detected, a `CycleError` is returned from the
iterator (or `SortResults::Partial` is returned if using the `to/into_vec` APIs)
.

Topological sorts are used to sort the nodes of a unidirectional graph. Typical
applications would include anything that requires dependency based sorting. For
example, it could be used to find the correct order of compilation dependencies,
or it could be used to find module cycles in languages that don't allow cycles.
The applications are limitless.

## Examples

```toml
[dependencies]
topo_sort = "0.2"
```

A basic example:

```rust
use topo_sort::{SortResults, TopoSort};

fn main() {
    let mut topo_sort = TopoSort::with_capacity(5);
    // read as "C" depends on "A" and "B"
    topo_sort.insert("C", vec!["A", "B"]);
    topo_sort.insert("E", vec!["B", "C"]);
    topo_sort.insert("A", vec![]);
    topo_sort.insert("D", vec!["A", "C", "E"]);
    topo_sort.insert("B", vec!["A"]);

    match topo_sort.into_vec() {
        SortResults::Full(nodes) => assert_eq!(vec!["A", "B", "C", "E", "D"], nodes),
        SortResults::Partial(_) => panic!("unexpected cycle!"),
    }
}
```

...or using iteration:

```rust
use topo_sort::TopoSort;

fn main() {
    let mut topo_sort = TopoSort::with_capacity(5);
    topo_sort.insert("C", vec!["A", "B"]);
    topo_sort.insert("E", vec!["B", "C"]);
    topo_sort.insert("A", vec![]);
    topo_sort.insert("D", vec!["A", "C", "E"]);
    topo_sort.insert("B", vec!["A"]);

    let mut nodes = Vec::with_capacity(5);
    for node in &topo_sort {
        // We check for cycle errors before usage
        match node {
            Ok(node) => nodes.push(*node),
            Err(_) => panic!("Unexpected cycle!"),
        }
    }

    assert_eq!(vec!["A", "B", "C", "E", "D"], nodes);
}
```

Cycle detection:

```rust
use topo_sort::TopoSort;

fn main() {
    let mut topo_sort = TopoSort::with_capacity(3);
    topo_sort.insert(1, vec![2, 3]);
    topo_sort.insert(2, vec![3]);
    assert_eq!(vec![2, 1], topo_sort.try_owned_vec().unwrap());

    topo_sort.insert(3, vec![1]); // cycle
    assert!(topo_sort.try_vec().is_err());
}
```

## Features

* Cycle detection - impossible to get data without handling cycle error
    * Choose methods for retrieving "all or nothing" or partial data
* Inserted nodes are never copied/cloned (unless explicitly requested
  via `owned` methods)
* Only requires `Eq` and `Hash` implemented on nodes
    * There are a few optional `owned` methods that require `Clone`
* Choice of iteration or converting into `Vec`
* Lazy sorting - sorting is initiated on iteration only

## Usage

Using `TopoSort` is a basic two step process:

1. Add in your nodes and dependencies to `TopoSort`
2. Iterate over the results *OR* store them directly in a `Vec`

* For step 2, there are three general ways to consume:
    * Iteration - returns a `Result` so cycles can be detected every iteration
    * `to/into_vec` functions - returns a `SortResults` enum with a `Vec` of
      full (no cycle) or partial results (when cycle detected)
    * `try_[into]_vec` functions - returns a `Vec` wrapped in a `Result` (full
      or no results)

## Algorithm

This is implemented
using [Kahn's algorithm](https://en.wikipedia.org/wiki/Topological_sorting).
While basic caution was taken not to do anything too egregious performance-wise,
the author's use cases are not performance sensitive, and it has not been
optimized. Still, the author tried to make reasonable trade offs and make it
flexible for a variety of use cases, not just the author's.

## Safety

The crate uses two tiny `unsafe` blocks which use the addresses of `HashMap`
keys in a new `HashMap`. This was necessary to avoid cloning inserted data on
owned iteration by self referencing the struct. Since there is no removal in
regular iteration (`iter()` or `for` loop using `&`), this should be safe as
there is no chance of the data moving during borrowed iteration. During
owned/consuming iteration (`into_iter()` or `for` without `&`), we remove the
entries as we go. If Rust's `HashMap` were to change and shrink during removals,
this iterator could break. If this makes you uncomfortable, simply don't use
consuming iteration.

## License

This project is licensed optionally under either:

* Apache License, Version 2.0, (LICENSE-APACHE
  or https://www.apache.org/licenses/LICENSE-2.0)
* MIT license (LICENSE-MIT or https://opensource.org/licenses/MIT)
