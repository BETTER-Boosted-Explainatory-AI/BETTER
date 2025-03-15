class UnionFind:
    def __init__(self, elements, heap_type):
        self.max_weight = 0
        self.merge_distances = []  # Track merge distances
        self.merge_indices = []    # Track merge indices
        self.element_to_index = {el: i for i, el in enumerate(elements)}
        self.index_to_element = {i: el for el, i in self.element_to_index.items()}
        self.parent = {el: el for el in elements}  # Initializing the parent of each element as itself
        self.rank = {el: 0 for el in elements}  # For path compression optimization
        self.components = {el: [el] for el in elements}  # Each element is initially its own component
        self.cluster_sizes = {el: 1 for el in elements}  # Initialize cluster sizes to 1
        self.next_cluster_index = len(elements)
        self.heap_type = heap_type
        
    def find(self, element):
        # Path compression optimization
        if self.parent[element] != element:
            self.parent[element] = self.find(self.parent[element])  # Recursively find the root
        return self.parent[element]
    
    def union(self, element1, element2, weight):
        root1 = self.find(element1)
        root2 = self.find(element2)
        if root1 != root2:
            # Get indices for the roots
            idx1 = self.element_to_index[root1]
            idx2 = self.element_to_index[root2]
            # Assign a new cluster index, starting from n
            new_index = self.next_cluster_index  # Assign new index
            self.next_cluster_index += 1  # Increment next available index
            # Store the merge
            self.merge_indices.append([idx1, idx2])
            self.merge_distances.append(weight)
            # Union by rank to keep the tree flat
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
                self.components[root1].extend(self.components[root2])
                self.cluster_sizes[new_index] = self.cluster_sizes.pop(root1, 1) + self.cluster_sizes.pop(root2, 1)
                merged = (root1, root2)
            else:
                self.parent[root1] = root2
                self.components[root2].extend(self.components[root1])
                self.cluster_sizes[new_index] = self.cluster_sizes.pop(root2, 1) + self.cluster_sizes.pop(root1, 1)
                merged = (root2, root1)
            # Properly map the new cluster index
            self.element_to_index[root1] = new_index
            self.element_to_index[root2] = new_index
            self.index_to_element[new_index] = f"Cluster-{new_index}"
            return merged
        return False

    def normalize_distances(self):
        """ Normalize merge distances to the range [0, 100]. """
        if not self.merge_distances:
            raise ValueError("No merges have been performed yet.")
        min_dist = min(self.merge_distances)
        max_dist = max(self.merge_distances)
        if max_dist == min_dist:
            # Avoid division by zero: all distances are identical, normalize to 50
            self.merge_distances = [50] * len(self.merge_distances)
        else:
            self.merge_distances = [
                (d - min_dist) / (max_dist - min_dist) * 100
                for d in self.merge_distances
            ]