#![warn(missing_docs)]

//! A "cycle-safe" topological sort for a set of nodes with dependencies in Rust.
//! Basically, it allows sorting a list by its dependencies while checking for
//! cycles in the graph. If a cycle is detected, a `CycleError` is returned from the
//! iterator.
//!
//! ## Usage
//!
//! ```toml
//! [dependencies]
//! topo_sort = "0.1"
//! ```
//!
//! A basic example:
//!
//! ```rust
//! let mut topo_sort = TopoSort::with_capacity(5);
//! topo_sort.insert("C", vec!["A", "B"]); // read: "C" depends on "A" and "B"
//! topo_sort.insert("E", vec!["B", "C"]);
//! topo_sort.insert("A", vec![]);
//! topo_sort.insert("D", vec!["A", "C", "E"]);
//! topo_sort.insert("B", vec!["A"]);
//!
//! assert_eq!(
//!     vec!["A", "B", "C", "E", "D"],
//!     topo_sort.to_owned_vec().unwrap()
//! );
//! ```
//!
//! ...or using iteration:
//!
//! ```rust
//! let mut topo_sort = TopoSort::with_capacity(5);
//! topo_sort.insert("C", vec!["A", "B"]);
//! topo_sort.insert("E", vec!["B", "C"]);
//! topo_sort.insert("A", vec![]);
//! topo_sort.insert("D", vec!["A", "C", "E"]);
//! topo_sort.insert("B", vec!["A"]);
//!
//! let mut nodes = Vec::with_capacity(5);
//! for node in &topo_sort {
//!     // We check for cycle errors before usage
//!     match node {
//!         Ok(node) => nodes.push(*node),
//!         Err(CycleError) => panic!("Unexpected cycle!"),
//!     }
//! }
//!
//! assert_eq!(vec!["A", "B", "C", "E", "D"], nodes);
//! ```

use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::{error, fmt};

/// An error type returned by the iterator when a cycle is detected in the dependency graph
#[derive(fmt::Debug)]
pub struct CycleError;

impl fmt::Display for CycleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl error::Error for CycleError {}

/// TopoSort is used as a collection to map nodes to their dependencies. The actual sort is "lazy" and is performed during iteration.
#[derive(Clone, Default)]
pub struct TopoSort<T> {
    // Dependent -> Dependencies
    node_depends: HashMap<T, HashSet<T>>,
}

impl<T> TopoSort<T>
where
    T: Eq + Hash,
{
    /// Initialize a new struct from a map. The key represents the node to be sorted and the set is its dependencies
    pub fn from_map(nodes: HashMap<T, HashSet<T>>) -> Self {
        TopoSort {
            node_depends: nodes,
        }
    }

    /// Initialize an empty struct with a given capacity
    pub fn with_capacity(capacity: usize) -> Self {
        TopoSort {
            node_depends: HashMap::with_capacity(capacity),
        }
    }

    /// Insert into this struct with the given node and a slice of its dependencies
    pub fn insert_from_slice(&mut self, node: T, slice: &[T])
    where
        T: Clone,
    {
        self.node_depends
            .insert(node, HashSet::from_iter(slice.to_vec()));
    }

    /// Insert into this struct with the given node and a set of its dependencies
    pub fn insert_from_set(&mut self, node: T, depends: HashSet<T>) {
        self.node_depends.insert(node, depends);
    }

    /// Insert into this struct with the given node and an iterator of its dependencies
    pub fn insert<I: IntoIterator<Item = T>>(&mut self, node: T, i: I) {
        self.node_depends.insert(node, i.into_iter().collect());
    }

    /// Start the sort process and return an iterator of the results
    pub fn iter(&self) -> TopoSortIter<'_, T> {
        TopoSortIter::new(&self.node_depends)
    }

    /// Sort and return a vector (with borrowed nodes) of the results
    pub fn to_vec(&self) -> Result<Vec<&T>, CycleError> {
        self.iter().collect()
    }

    /// Sort and return a vector (with owned/cloned nodes) of the results
    pub fn to_owned_vec(&self) -> Result<Vec<T>, CycleError>
    where
        T: Clone,
    {
        self.iter()
            .map(|result| result.map(|node| node.clone()))
            .collect()
    }
}

impl<'d, T> IntoIterator for &'d TopoSort<T>
where
    T: Eq + Hash,
{
    type Item = Result<&'d T, CycleError>;
    type IntoIter = TopoSortIter<'d, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterator over the final result of the topological sort
pub struct TopoSortIter<'d, T> {
    // Dependency -> (Dependents, Edge Count)
    nodes: HashMap<&'d T, (HashSet<&'d T>, u32)>,
    no_edges: Vec<&'d T>,
}

impl<'d, T> TopoSortIter<'d, T>
where
    T: Eq + Hash,
{
    fn new(node_depends: &'d HashMap<T, HashSet<T>>) -> Self {
        let len = node_depends.len(); // Avoids borrow issues in closure
        let mut nodes = HashMap::with_capacity(len);

        // Assume every dependency has every node as a dependent - likely wasteful, but avoids excess allocations
        let new_entry_fn = || (HashSet::with_capacity(len), 0);

        for (dependent, dependencies) in node_depends {
            for dependency in dependencies {
                // Filter nodes that are only dependencies or self dependencies - add others as edges
                if dependent != dependency && node_depends.contains_key(dependency) {
                    // Each dependent tracks the # of dependencies
                    let dependent_entry = nodes.entry(dependent).or_insert_with(new_entry_fn);
                    dependent_entry.1 += 1;

                    // Each dependency tracks all it's dependents
                    let dependency_entry = nodes.entry(dependency).or_insert_with(new_entry_fn);
                    dependency_entry.0.insert(dependent);
                }
            }
        }

        // Find first batch of ready nodes (TODO: move into loop so we can set capacity?)
        let no_edges: Vec<_> = nodes
            .iter()
            .filter(|(_, (_, edges))| *edges == 0)
            .map(|(&node, _)| node)
            .collect();

        TopoSortIter { nodes, no_edges }
    }
}

impl<'d, T> Iterator for TopoSortIter<'d, T>
where
    T: Eq + Hash,
{
    type Item = Result<&'d T, CycleError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.no_edges.pop() {
            Some(node) => {
                // NOTE: Unwrap() should be safe - we know it was in there from init
                // We are done with this node - remove entirely
                let (dependents, _) = &self.nodes.remove(node).unwrap();

                // Decrement the edge count of all nodes that depend on this one and add them
                // to no_edges when they hit zero
                for &dependent in dependents {
                    // NOTE: Unwrap() should be safe - we know it was in there from init
                    let (_, edges) = self.nodes.get_mut(dependent).unwrap();
                    *edges -= 1;
                    if *edges == 0 {
                        self.no_edges.push(dependent);
                    }
                }

                Some(Ok(node))
            }
            None if self.nodes.is_empty() => None,
            None => Some(Err(CycleError)),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.nodes.len();
        (len, Some(len))
    }
}

#[cfg(test)]
mod tests {
    use crate::TopoSort;

    #[test]
    fn test_direct_cycle() {
        let mut topo_sort = TopoSort::with_capacity(2);
        topo_sort.insert(1, vec![2]);
        topo_sort.insert(2, vec![1]); // cycle

        assert!(topo_sort.to_vec().is_err())
    }

    #[test]
    fn test_indirect_cycle() {
        let mut topo_sort = TopoSort::with_capacity(3);
        topo_sort.insert(1, vec![2, 3]);
        topo_sort.insert(2, vec![3]);
        topo_sort.insert(3, vec![1]); // cycle

        assert!(topo_sort.to_vec().is_err())
    }

    #[test]
    fn test_good() {
        let mut topo_sort = TopoSort::with_capacity(5);
        topo_sort.insert("C", vec!["A", "B"]);
        topo_sort.insert("E", vec!["B", "C"]);
        topo_sort.insert("A", vec![]);
        topo_sort.insert("D", vec!["A", "C", "E"]);
        topo_sort.insert("B", vec!["A"]);

        assert_eq!(
            vec!["A", "B", "C", "E", "D"],
            topo_sort.to_owned_vec().unwrap()
        );
    }

    #[test]
    fn test_good_with_excess_data() {
        let mut topo_sort = TopoSort::with_capacity(5);
        topo_sort.insert("C", vec!["F", "A", "B", "F"]); // There is no 'F' - two of them
        topo_sort.insert("E", vec!["C", "B", "C"]); // Double "C" dependency
        topo_sort.insert("A", vec!["A", "G"]); // Self dependency + there is no 'G'
        topo_sort.insert("D", vec!["A", "C", "E"]);
        topo_sort.insert("B", vec!["B", "A"]); // Self dependency

        assert_eq!(
            vec!["A", "B", "C", "E", "D"],
            topo_sort.to_owned_vec().unwrap()
        );
    }

    #[test]
    fn test_loop() {
        let mut topo_sort = TopoSort::with_capacity(5);
        topo_sort.insert("C", vec!["A", "B"]);
        topo_sort.insert("E", vec!["B", "C"]);
        topo_sort.insert("A", vec![]);
        topo_sort.insert("D", vec!["A", "C", "E"]);
        topo_sort.insert("B", vec!["A"]);

        let mut nodes = Vec::with_capacity(5);
        for node in &topo_sort {
            // Must check for cycle errors before usage
            match node {
                Ok(node) => nodes.push(*node),
                Err(CycleError) => panic!("Unexpected cycle!"),
            }
        }

        assert_eq!(vec!["A", "B", "C", "E", "D"], nodes);
    }
}
