use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::{error, fmt};

#[derive(fmt::Debug)]
pub struct CycleError;

impl fmt::Display for CycleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl error::Error for CycleError {}

#[derive(Clone, Default)]
pub struct Depends<T> {
    // Dependent -> Dependencies
    node_depends: HashMap<T, HashSet<T>>,
}

impl<T> Depends<T>
where
    T: Eq + Hash,
{
    pub fn from_map(nodes: HashMap<T, HashSet<T>>) -> Self {
        Depends {
            node_depends: nodes,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Depends {
            node_depends: HashMap::with_capacity(capacity),
        }
    }

    pub fn insert_from_slice(&mut self, node: T, slice: &[T])
    where
        T: Clone,
    {
        self.node_depends
            .insert(node, HashSet::from_iter(slice.to_vec()));
    }

    pub fn insert_from_set(&mut self, node: T, depends: HashSet<T>) {
        self.node_depends.insert(node, depends);
    }

    pub fn insert<I: IntoIterator<Item = T>>(&mut self, node: T, i: I) {
        self.node_depends.insert(node, i.into_iter().collect());
    }

    pub fn iter(&self) -> DependsIter<'_, T> {
        DependsIter::new(&self.node_depends)
    }

    pub fn to_vec(&self) -> Result<Vec<&T>, CycleError> {
        self.iter().collect()
    }

    pub fn to_owned_vec(&self) -> Result<Vec<T>, CycleError>
    where
        T: Clone,
    {
        self.iter()
            .map(|result| result.map(|node| node.clone()))
            .collect()
    }
}

impl<'d, T> IntoIterator for &'d Depends<T>
where
    T: Eq + Hash,
{
    type Item = Result<&'d T, CycleError>;
    type IntoIter = DependsIter<'d, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub struct DependsIter<'d, T> {
    // Dependency -> (Dependents, Edge Count)
    nodes: HashMap<&'d T, (HashSet<&'d T>, u32)>,
    no_edges: Vec<&'d T>,
}

impl<'d, T> DependsIter<'d, T>
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

        DependsIter { nodes, no_edges }
    }
}

impl<'d, T> Iterator for DependsIter<'d, T>
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
    use crate::Depends;

    #[test]
    fn test_direct_cycle() {
        let mut depends = Depends::with_capacity(2);
        depends.insert(1, vec![2]); // cycle
        depends.insert(2, vec![1]); // cycle

        assert!(depends.to_vec().is_err())
    }

    #[test]
    fn test_indirect_cycle() {
        let mut depends = Depends::with_capacity(3);
        depends.insert(1, vec![2, 3]);
        depends.insert(2, vec![3]);
        depends.insert(3, vec![1]); // cycle

        assert!(depends.to_vec().is_err())
    }

    #[test]
    fn test_good() {
        let mut depends = Depends::with_capacity(5);
        depends.insert("C", vec!["A", "B"]);
        depends.insert("E", vec!["B", "C"]);
        depends.insert("A", vec![]);
        depends.insert("D", vec!["A", "C", "E"]);
        depends.insert("B", vec!["A"]);

        assert_eq!(
            vec!["A", "B", "C", "E", "D"],
            depends.to_owned_vec().unwrap()
        )
    }

    #[test]
    fn test_good_with_excess_data() {
        let mut depends = Depends::with_capacity(5);
        depends.insert("C", vec!["F", "A", "B", "F"]); // There is no 'F' - two of them
        depends.insert("E", vec!["C", "B", "C"]); // Double "C" dependency
        depends.insert("A", vec!["A", "G"]); // Self dependency + there is no 'G'
        depends.insert("D", vec!["A", "C", "E"]);
        depends.insert("B", vec!["B", "A"]); // Self dependency

        assert_eq!(
            vec!["A", "B", "C", "E", "D"],
            depends.to_owned_vec().unwrap()
        )
    }

    #[test]
    fn test_loop() {
        let mut depends = Depends::with_capacity(5);
        depends.insert("C", vec!["A", "B"]);
        depends.insert("E", vec!["B", "C"]);
        depends.insert("A", vec![]);
        depends.insert("D", vec!["A", "C", "E"]);
        depends.insert("B", vec!["A"]);

        let mut actual = Vec::with_capacity(5);
        for node in &depends {
            actual.push(*node.unwrap());
        }

        assert_eq!(vec!["A", "B", "C", "E", "D"], actual)
    }
}
