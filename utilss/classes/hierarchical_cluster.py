import numpy as np
import json
import os
import copy
from scipy.cluster.hierarchy import to_tree

class HierarchicalCluster:
    def __init__(self, labels_dict=None):
        self.Z = []  # Track merge distances
        self.labels_dict = labels_dict or {}
        self.new_labels = None
        self.Z_tree_format = None
        
    def create_dendrogram_data(self, UnionFind, labels, max_weight):
        """
        Create dendrogram data (Z matrix) without plotting and return it
        """
        if not UnionFind.merge_indices or not UnionFind.merge_distances:
            raise ValueError("No merges have been performed yet")
                
        cluster_sizes = {}  # Track cluster sizes dynamically
        current_cluster_index = len(UnionFind.element_to_index)  # Start from n
        Z = []
        used_clusters = set()  # Track used clusters to prevent duplicates
            
        inverted_distances = [dist for dist in UnionFind.merge_distances]
        
        for i, ((idx1, idx2), dist) in enumerate(zip(UnionFind.merge_indices, inverted_distances)):
                if idx1 in used_clusters or idx2 in used_clusters:
                    raise ValueError(f"Cluster index {idx1} or {idx2} is being reused before merging.")
                
                size1 = cluster_sizes.get(idx1, 1)
                size2 = cluster_sizes.get(idx2, 1)
                new_cluster_size = size1 + size2  # Cumulative size
                # Use the inverted normalized distance directly
                Z.append([idx1, idx2, dist, new_cluster_size])
                
                # Mark these clusters as used
                used_clusters.add(idx1)
                used_clusters.add(idx2)
                
                # Assign a new cluster index for tracking
                cluster_sizes[current_cluster_index] = new_cluster_size
                current_cluster_index += 1
                            
        self.Z = np.array(Z)
        
        if not self.labels_dict:
            self.labels_dict = {name: i for i, name in enumerate(labels)}
            
        self.new_labels = sorted(labels)
            
        return self.Z
    
    
    def _build_tree_format(self, node, labels):
        if node.is_leaf():
            return {
                "id": node.id,
                "name": labels[node.id],
                }
        else:
            return {
                "id": node.id,
                "name": f"Cluster {node.id}",
                "children": [self._build_tree_format(node.get_left(), labels), self._build_tree_format(node.get_right(), labels)],
                "value": node.dist
            }

    def _build_tree_hierarchy(self, linkage_matrix, labels):
        tree, nodes = to_tree(linkage_matrix, rd=True)
        return self._build_tree_format(tree, labels)
    
        
    def save_dendrogram_as_json(self, dendrogram_filename, labels):
        """
        Create dendrogram data and save it as JSON
        
        Parameters:
        - labels: List of labels for the dendrogram leaves
        - max_weight: Maximum weight/distance value
        - UnionFind: UnionFind data structure with merge history
        - output_path: Path to save the JSON file
        
        Returns:
        - Path to the saved JSON file
        """
        # Create a dictionary with the dendrogram data
        
        DENDROGRAMS_PATH = os.getenv("DENDROGRAMS_PATH")
        dendrogram_path = f'{DENDROGRAMS_PATH}/{dendrogram_filename}.json'
        
        self.Z = np.array(self.Z, dtype=np.float64)
        self.Z_tree_format = self._build_tree_hierarchy(self.Z, labels)
        
        dendrogram_data = {
            'Z': self.Z.tolist(),  # Convert numpy array to list for JSON serialization
            'new_labels': self.new_labels,
            'Z_tree_format': self.Z_tree_format
        }
        
        # Save the data as JSON
        with open(dendrogram_path, 'w') as f:
            json.dump(dendrogram_data, f, indent=2)
            
        print(f"Dendrogram data saved to {dendrogram_path}")
        return dendrogram_path
    
    def load_dendrogram_from_json(self, dendrogram_filename):
        """
        Load dendrogram data from a JSON file
        
        Parameters:
        - json_path: Path to the JSON file containing dendrogram data
        
        Returns:
        - self with loaded data
        """
        DENDROGRAMS_PATH = os.getenv("DENDROGRAMS_PATH")
        dendrogram_path = f'{DENDROGRAMS_PATH}/{dendrogram_filename}.json'
        
        with open(dendrogram_path, 'r') as f:
            data = json.load(f)
            
        self.Z = np.array(data['Z'])
        self.new_labels = data.get('new_labels')
        self.Z_tree_format = data.get('Z_tree_format')
        
        print(f"Dendrogram data loaded from {dendrogram_path}")
        return self
        


    # sub heirarchical clustering
    
    def _validate_labels(self, labels, selected_labels):
        """
        Validate that selected labels exist in the original dataset.
        
        Parameters:
        - labels (list): All class names corresponding to the original leaf indices.
        - selected_labels (list): The subset of class labels to include in the sub-dendrogram.
        
        Returns:
        - list: Indices of selected labels in the original dataset.
        """
        original_label_to_idx = {name: i for i, name in enumerate(labels)}
        
        # Check that all selected labels exist in original data
        for label in selected_labels:
            if label not in original_label_to_idx:
                raise ValueError(f"Label '{label}' not found in original class names")
        
        # Get original indices of selected labels
        return [original_label_to_idx[label] for label in selected_labels]
    
    def _build_dendrogram_structure(self, labels, selected_indices):
        """
        Build the full dendrogram structure with parent-child relationships.
        
        Parameters:
        - labels (list): All class names.
        - selected_indices (list): Indices of selected labels.
        
        Returns:
        - tuple: (n_samples, parent, children_left, children_right, node_height)
        """
        n_samples = len(labels)
        n_nodes = 2 * n_samples - 1  # Total number of nodes in the dendrogram
        
        # Initialize parent and children maps
        parent = np.zeros(n_nodes, dtype=np.int64) - 1  # -1 means no parent
        children_left = np.zeros(n_nodes, dtype=np.int64) - 1
        children_right = np.zeros(n_nodes, dtype=np.int64) - 1
        node_height = np.zeros(n_nodes, dtype=np.float64)
        
        # Fill in the structure from Z
        for i, (left, right, height, _) in enumerate(self.Z):
            left = int(left)
            right = int(right)
            node_id = n_samples + i
            
            children_left[node_id] = left
            children_right[node_id] = right
            parent[left] = node_id
            parent[right] = node_id
            node_height[node_id] = height
            
        return n_samples, parent, children_left, children_right, node_height
    
    def _find_path_down(self, node, targets, children_left, children_right, path):
        """
        Find a path from a node down to any target in targets.
        
        Parameters:
        - node: The starting node.
        - targets: Set of target nodes to reach.
        - children_left, children_right: Arrays of left and right children.
        - path: List to store the path (modified in-place).
        
        Returns:
        - bool: True if a path was found, False otherwise.
        """
        path.append(node)
        
        if node in targets:
            return True
        
        left = children_left[node]
        right = children_right[node]
        
        if left != -1:
            if self._find_path_down(left, targets, children_left, children_right, path):
                return True
            path.pop()  # Remove the failed path
        
        if right != -1:
            if self._find_path_down(right, targets, children_left, children_right, path):
                return True
            path.pop()  # Remove the failed path
        
        return False
    
    def _find_common_ancestors(self, selected_indices, parent, node_height):
        """
        Find lowest common ancestors for all pairs of selected indices.
        
        Parameters:
        - selected_indices (list): Indices of selected labels.
        - parent (np.ndarray): Array of parent indices.
        - node_height (np.ndarray): Array of node heights.
        
        Returns:
        - dict: Dictionary mapping (idx1, idx2) to their lowest common ancestor.
        """
        # Helper function to find all ancestors of a node
        def get_ancestors(node_id):
            ancestors = []
            current = node_id
            while parent[current] != -1:
                ancestors.append(parent[current])
                current = parent[current]
            return ancestors
        
        # Get all ancestors for each selected leaf
        all_ancestors = {}
        for idx in selected_indices:
            all_ancestors[idx] = get_ancestors(idx)
        
        # Find lowest common ancestors for all pairs of selected indices
        common_ancestors = {}
        for i, idx1 in enumerate(selected_indices):
            for idx2 in selected_indices[i+1:]:
                # Find the lowest common ancestor
                ancestors1 = set(all_ancestors[idx1])
                for ancestor in all_ancestors[idx2]:
                    if ancestor in ancestors1:
                        # This is the lowest common ancestor
                        common_ancestors[(idx1, idx2)] = ancestor
                        break
        
        return common_ancestors
    
    def _process_ancestor_relationships(self, common_ancestors, parent, node_height, selected_indices, 
                                       children_left, children_right):
        """
        Process ancestor relationships to build the sub-dendrogram.
        
        Parameters:
        - common_ancestors (dict): Dictionary of common ancestors.
        - parent (np.ndarray): Array of parent indices.
        - node_height (np.ndarray): Array of node heights.
        - selected_indices (list): Indices of selected labels.
        - children_left, children_right: Arrays of left and right children.
        
        Returns:
        - list: Relationships between nodes in the sub-dendrogram.
        """
        # Map original indices to new indices
        node_map = {idx: i for i, idx in enumerate(selected_indices)}
        next_node_id = len(selected_indices)
        relationships = []
        
        # Process all pairs to find relationships
        for (idx1, idx2), ancestor in sorted(common_ancestors.items(), key=lambda x: node_height[x[1]]):
            # Find paths from each index to the ancestor
            path1 = []
            current = idx1
            while current != ancestor:
                current = parent[current]
                path1.append(current)
            
            path2 = []
            current = idx2
            while current != ancestor:
                current = parent[current]
                path2.append(current)
            
            # We need the direct children of the ancestor that are on paths to idx1 and idx2
            if path1 and path2:
                child1 = path1[-1] if len(path1) > 1 else idx1
                child2 = path2[-1] if len(path2) > 1 else idx2
                
                # Create the relationship based on the direct children
                if child1 in node_map and child2 in node_map:
                    relationships.append((node_map[child1], node_map[child2], node_height[ancestor]))
                    
                    # Create a new node for this merger
                    node_map[ancestor] = next_node_id
                    next_node_id += 1
                else:
                    # Handle multi-level hierarchy cases
                    if child1 not in node_map and child1 not in selected_indices:
                        self._process_intermediate_child(child1, selected_indices, children_left, 
                                                      children_right, node_map, node_height, 
                                                      relationships, next_node_id)
                    
                    if child2 not in node_map and child2 not in selected_indices:
                        self._process_intermediate_child(child2, selected_indices, children_left, 
                                                      children_right, node_map, node_height, 
                                                      relationships, next_node_id)
        
        return relationships
    
    def _process_intermediate_child(self, child, selected_indices, children_left, children_right, 
                                   node_map, node_height, relationships, next_node_id):
        """
        Process an intermediate child node not directly in the selected indices.
        
        Parameters:
        - child: The child node to process.
        - selected_indices (list): Indices of selected labels.
        - children_left, children_right: Arrays of left and right children.
        - node_map (dict): Map from original indices to new indices.
        - node_height (np.ndarray): Array of node heights.
        - relationships (list): List of relationships to update.
        - next_node_id (int): Next available node ID.
        """
        descend_path = []
        if self._find_path_down(child, selected_indices, children_left, children_right, descend_path):
            # Use the first path found
            for i in range(len(descend_path)-1):
                if descend_path[i] in node_map and descend_path[i+1] not in node_map:
                    relationships.append((node_map[descend_path[i]], next_node_id, 
                                        node_height[descend_path[i+1]]))
                    node_map[descend_path[i+1]] = next_node_id
                    next_node_id += 1
    
    def _build_sub_z_matrix(self, labels, selected_indices):
        """
        Build the sub-dendrogram Z matrix directly from the original Z matrix.
        
        Parameters:
        - labels (list): All class names.
        - selected_indices (list): Indices of selected labels.
        
        Returns:
        - np.ndarray: The sub-dendrogram linkage matrix.
        """
        n_samples = len(labels)
        Z_sub = []
        cluster_size = {i: 1 for i in range(len(selected_indices))}
        
        # Create a mapping of original indices to their new positions
        new_positions = {original: new for new, original in enumerate(selected_indices)}
        
        # Clone the selected indices to track merges
        active_nodes = selected_indices.copy()
        next_id = len(selected_indices)
        
        # Process the original Z matrix in order
        for i, (left, right, height, _) in enumerate(self.Z):
            left, right = int(left), int(right)
            left_in_active = left in active_nodes
            right_in_active = right in active_nodes
            
            # If both nodes are in our active set, merge them
            if left_in_active and right_in_active:
                new_left = new_positions[left]
                new_right = new_positions[right]
                
                # Calculate the size of the new cluster
                new_size = cluster_size.get(new_left, 1) + cluster_size.get(new_right, 1)
                
                # Add to Z_sub (ensuring new_left < new_right as scipy expects)
                if new_left > new_right:
                    new_left, new_right = new_right, new_left
                
                Z_sub.append([new_left, new_right, height, new_size])
                
                # Update tracking
                active_nodes.remove(left)
                active_nodes.remove(right)
                active_nodes.append(n_samples + i)  # Add the new node
                new_positions[n_samples + i] = next_id
                cluster_size[next_id] = new_size
                next_id += 1
            
            # If one node is in our active set, replace it with the parent
            elif left_in_active:
                active_nodes.remove(left)
                active_nodes.append(n_samples + i)
                new_positions[n_samples + i] = new_positions[left]
            
            elif right_in_active:
                active_nodes.remove(right)
                active_nodes.append(n_samples + i)
                new_positions[n_samples + i] = new_positions[right]
        
        # Convert to numpy array and validate indices
        return self._finalize_sub_z(Z_sub, selected_indices)
    
    def _finalize_sub_z(self, Z_sub, selected_indices):
        """
        Finalize the sub-dendrogram Z matrix.
        
        Parameters:
        - Z_sub (list): List of sub-dendrogram merges.
        - selected_indices (list): Indices of selected labels.
        
        Returns:
        - np.ndarray: The finalized sub-dendrogram linkage matrix.
        """
        if Z_sub:
            Z_sub_array = np.array(Z_sub)
            
            # Ensure the first two columns only reference valid indices
            max_idx = len(selected_indices) - 1
            for i, row in enumerate(Z_sub_array):
                if row[0] > max_idx:
                    Z_sub_array[i, 0] = max_idx
                if row[1] > max_idx:
                    Z_sub_array[i, 1] = max_idx
                max_idx += 1
            
            return Z_sub_array
        else:
            # If no relationships were found, create an empty array with the right shape
            return np.empty((0, 4))
    
    def extract_sub_dendrogram(self, labels, selected_labels):
        """
        Extract a sub-dendrogram from the full Z matrix for only the selected labels.
        
        Parameters:
        - labels (list): All class names corresponding to the original leaf indices.
        - selected_labels (list): The subset of class labels to include in the sub-dendrogram.
        
        Returns:
        - np.ndarray: The sub-dendrogram linkage matrix.
        - list: The selected labels in the order they appear in the sub-dendrogram.
        """
        # Step 1: Validate labels and get indices
        selected_indices = self._validate_labels(labels, selected_labels)
        
        # Step 2: Build the full dendrogram structure
        n_samples, parent, children_left, children_right, node_height = self._build_dendrogram_structure(
            labels, selected_indices)
        
        # Step 3: Find common ancestors (optional for complex hierarchies)
        common_ancestors = self._find_common_ancestors(selected_indices, parent, node_height)
        
        # Step 4: Process ancestor relationships (optional, used for complex hierarchies)
        relationships = self._process_ancestor_relationships(
            common_ancestors, parent, node_height, selected_indices, children_left, children_right)
        
        # Step 5: Build the sub-Z matrix (main approach)
        Z_sub = self._build_sub_z_matrix(labels, selected_indices)
        
        return Z_sub, selected_labels
    
    def filter_dendrogram_by_labels(self, full_data, target_labels):
        """
        Create a minimal dendrogram containing only the specified labels while preserving hierarchy.
        
        Args:
            full_data (dict): The full dendrogram data structure
            target_labels (list): List of label names to keep
            
        Returns:
            dict: A minimal tree containing only paths to the specified labels
        """
        
        # Check if a node or its descendants contain any of the target labels
        def contains_target_label(node):
            # If this is a leaf node (no children)
            if 'children' not in node:
                return node.get('name') in target_labels
            
            # Check all children
            for child in node.get('children', []):
                if contains_target_label(child):
                    return True
            
            return False
        
        # Create a filtered copy of the tree
        def filter_tree(node):
            # If this node doesn't contain any target labels in its subtree, skip it
            if not contains_target_label(node):
                return None
            
            # Create a new node with the same ID and name
            new_node = {
                'id': node.get('id'),
                'name': node.get('name')
            }
            
            # Copy the value if it exists
            if 'value' in node:
                new_node['value'] = node.get('value')
            
            # If it's a leaf node, we're done
            if 'children' not in node:
                return new_node
            
            # Process children
            filtered_children = []
            for child in node.get('children', []):
                filtered_child = filter_tree(child)
                if filtered_child:
                    filtered_children.append(filtered_child)
            
            # Add filtered children if any exist
            if filtered_children:
                new_node['children'] = filtered_children
            
            return new_node
        
        # Start filtering from the root
        return filter_tree(full_data)

    def merge_clusters(self, node):
        if "children" not in node:
            return node
        
        merged_children = []
        for child in node["children"]:
            merged_child = self.merge_clusters(child)
            if merged_child:
                merged_children.append(merged_child)
        
        # If all children have value 100, merge them into the parent
        if all(c.get("value", 0) == 100 for c in merged_children):
            node["children"] = [grandchild for child in merged_children for grandchild in child.get("children", [])]
        else:
            node["children"] = merged_children
        
        # If the cluster has only one direct child, remove itself
        if len(node["children"]) == 1:
            return node["children"][0]
        
        return node
    
    def get_sub_dendrogram_formatted(self, selected_labels):
        filtered_tree = self.filter_dendrogram_by_labels(self.Z_tree_format, selected_labels)
        filtered_tree = self.merge_clusters(filtered_tree)
        filtered_tree_json = json.dumps(filtered_tree, indent=2)
        return filtered_tree_json


