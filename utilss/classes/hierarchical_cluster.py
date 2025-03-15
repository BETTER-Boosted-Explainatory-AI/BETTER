import numpy as np
import json

class HierarchicalCluster:
    def __init__(self, labels_dict=None):
        self.Z = []  # Track merge distances
        self.labels_dict = labels_dict or {}
        self.label_mapping = None
        
    def create_dendrogram_data(self, UnionFind, labels, max_weight):
        """
        Create dendrogram data (Z matrix) without plotting and return it
        """
        if not UnionFind.merge_indices or not UnionFind.merge_distances:
            raise ValueError("No merges have been performed yet")
            
        self.label_mapping = {i: label for i, label in enumerate(labels)}    
        cluster_sizes = {}  # Track cluster sizes dynamically
        current_cluster_index = len(UnionFind.element_to_index)  # Start from n
        
        Z = []
        used_clusters = set()  # Track used clusters to prevent duplicates
        for (idx1, idx2), dist in zip(UnionFind.merge_indices, UnionFind.merge_distances):
            if idx1 in used_clusters or idx2 in used_clusters:
                raise ValueError(f"Cluster index {idx1} or {idx2} is being reused before merging.")
                
            size1 = cluster_sizes.get(idx1, 1)
            size2 = cluster_sizes.get(idx2, 1)
            new_cluster_size = size1 + size2  # Cumulative size
            
            # Adjust the distance by subtracting from max_weight
            adjusted_dist = dist
            if UnionFind.heap_type == "max":
                adjusted_dist = max_weight + dist  
                
            Z.append([idx1, idx2, adjusted_dist, new_cluster_size])
            # Mark these clusters as used
            used_clusters.add(idx1)
            used_clusters.add(idx2)
            # Assign a new cluster index for tracking
            cluster_sizes[current_cluster_index] = new_cluster_size
            current_cluster_index += 1
            
        self.Z = np.array(Z)
        if not self.labels_dict:
            self.labels_dict = {name: i for i, name in enumerate(labels)}
            
        return self.Z
    
    def save_dendrogram_as_json(self, labels, output_path):
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
        dendrogram_data = {
            'Z': self.Z.tolist(),  # Convert numpy array to list for JSON serialization
            'labels': labels,
            'labels_dict': self.labels_dict,
            'label_mapping': self.label_mapping
        }
        
        # Save the data as JSON
        with open(output_path, 'w') as f:
            json.dump(dendrogram_data, f, indent=2)
            
        print(f"Dendrogram data saved to {output_path}")
        return output_path
    
    def load_dendrogram_from_json(self, json_path):
        """
        Load dendrogram data from a JSON file
        
        Parameters:
        - json_path: Path to the JSON file containing dendrogram data
        
        Returns:
        - self with loaded data
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        self.Z = np.array(data['Z'])
        self.labels_dict = data['labels_dict']
        self.label_mapping = data.get('label_mapping')
        
        print(f"Dendrogram data loaded from {json_path}")
        return self
        
    def find_lca(self, label1, label2, n_original_elements, cluster_elements=None):
        """
        Find the lowest common ancestor (LCA) of two labels in the dendrogram.
        
        Parameters:
        - label1, label2: The labels to find the LCA for
        - n_original_elements: Number of original elements (leaves)
        - cluster_elements: Optional dict to track elements in each cluster
        
        Returns:
        - tuple: (label1, label2, distance, cluster_id)
        """
        if cluster_elements is None:
            cluster_elements = {}
            for i in range(n_original_elements):
                cluster_elements[i] = {i}
                
        if label1 == label2:
            raise ValueError("Labels must be different to find a common ancestor")
        
        # Convert labels to indices if they're not already
        idx1 = label1 if isinstance(label1, int) else self.labels_dict.get(label1)
        idx2 = label2 if isinstance(label2, int) else self.labels_dict.get(label2)
        
        if idx1 is None or idx2 is None:
            raise ValueError(f"Label not found in dendrogram: {label1 if idx1 is None else label2}")
    
        for i, (left, right, dist, size) in enumerate(self.Z):
            left, right = int(left), int(right)
            cluster_id = n_original_elements + i
    
            # Merge the sets of elements
            cluster_elements[cluster_id] = cluster_elements.get(left, {left}).union(
                cluster_elements.get(right, {right}))
    
            # Check if both indices are found in this merged cluster
            if idx1 in cluster_elements[cluster_id] and idx2 in cluster_elements[cluster_id]:
                # Determine which label is in which cluster
                if idx1 in cluster_elements.get(left, {left}) and idx2 in cluster_elements.get(right, {right}):
                    # If idx1 is in left and idx2 is in right
                    return idx1, idx2, dist, cluster_id
                elif idx2 in cluster_elements.get(left, {left}) and idx1 in cluster_elements.get(right, {right}):
                    # If idx2 is in left and idx1 is in right
                    return idx2, idx1, dist, cluster_id
                else:
                    # Either both are in left or both are in right, so we need to go deeper
                    continue

        # If no LCA is found
        raise ValueError("No common ancestor found for the given labels")