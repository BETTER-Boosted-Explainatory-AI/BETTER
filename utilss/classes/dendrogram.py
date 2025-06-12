from scipy.cluster.hierarchy import to_tree
from utilss.wordnet_utils import process_hierarchy
import json
import os
import numpy as np
import pickle


class Dendrogram:
    def __init__(self, dendrogram_filename, Z=None):
        self.Z = Z
        self.Z_tree_format = None
        self.dendrogram_filename = dendrogram_filename

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
        self.Z_tree_format = self._build_tree_format(tree, labels)
        self.Z_tree_format = process_hierarchy(self.Z_tree_format)
        return self.Z_tree_format  
    
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

    def save_dendrogram(self, linkage_matrix=None):
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
        
        # DENDROGRAMS_PATH = os.getenv("DENDROGRAMS_PATH")
        # dendrogram_path = f'{DENDROGRAMS_PATH}/{self.dendrogram_filename}.json'
        
        # Load existing data if the file exists
        directory = os.path.dirname(self.dendrogram_filename)
        
        # Create directories if they don't exist
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        # Define file paths
        pickle_path = f'{self.dendrogram_filename}.pkl'
        json_path = f'{self.dendrogram_filename}.json'
        
        # Save linkage matrix as pickle if provided
        if linkage_matrix is not None:
            with open(pickle_path, 'wb') as f:
                pickle.dump(linkage_matrix, f)
            print(f"Linkage matrix saved to {pickle_path}")
        
        # Save tree format as JSON
        if self.Z_tree_format is not None:
            with open(json_path, 'w') as f:
                json.dump(self.Z_tree_format, f, indent=2)
            print(f"Tree format saved to {json_path}")
        
        return pickle_path, json_path
        
    
    def load_dendrogram(self):
        """
        Load dendrogram data from a JSON file
        
        Parameters:
        - json_path: Path to the JSON file containing dendrogram data
        
        Returns:
        - self with loaded data
        """
        pickle_path = f'{self.dendrogram_filename}.pkl'
        json_path = f'{self.dendrogram_filename}.json'
        
        # Load linkage matrix from pickle
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                self.Z = pickle.load(f)
            print(f"Linkage matrix loaded from {pickle_path}")
        else:
            print(f"Pickle file not found: {pickle_path}")
            self.Z = None
        
        # Load tree format from JSON
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                self.Z_tree_format = json.load(f)
            print(f"Tree format loaded from {json_path}")
        else:
            print(f"JSON file not found: {json_path}")
            self.Z_tree_format = None
        
        return self
    
    def find_name_hierarchy(self, node, target_name):
        """
        Recursively search for a target name in the hierarchical structure.
        
        Args:
            node (dict): The current node in the hierarchical structure
            target_name (str): The name to search for
        
        Returns:
            list: A list of names representing the hierarchy from target to root,
                or None if the target is not found
        """
        # Check if the current node's name matches the target
        if node.get('name') == target_name:
            return [target_name]
        
        # If the node has children, recursively search through them
        if 'children' in node:
            for child in node['children']:
                # Recursively search in each child
                result = self.find_name_hierarchy(child, target_name)
                
                # If a path is found, prepend the current node's name
                if result is not None:
                    # Only prepend the current node's name if it's a cluster
                    if node.get('name'):
                        result.append(node['name'])
                    return result
        
        # If no match is found in this branch
        return None
        
    def rename_cluster(self, cluster_id, new_name):
        """
        Rename a cluster in the dendrogram hierarchy, only if the new name is unique and the node is not a leaf.

        Args:
            cluster_id (int): The ID of the cluster to rename
            new_name (str): The new name for the cluster

        Returns:
            dict: The updated dendrogram hierarchy, or None if name exists or node is a leaf
        """
        print(f"Renaming cluster {cluster_id} to {new_name}")

        # Collect all existing names in the dendrogram
        def collect_names(node, names):
            names.add(node.get('name'))
            for child in node.get('children', []):
                collect_names(child, names)

        existing_names = set()
        collect_names(self.Z_tree_format, existing_names)

        # Only rename if the new name does not exist
        if new_name in existing_names:
            print(f"Name '{new_name}' already exists. Rename aborted.")
            return None

        # Find the node by cluster_id
        def find_node(node):
            if node.get('id') == cluster_id:
                return node
            for child in node.get('children', []):
                result = find_node(child)
                if result:
                    return result
            return None

        target_node = find_node(self.Z_tree_format)
        if target_node is None:
            print(f"Cluster ID {cluster_id} not found.")
            return None

        # If the node is a leaf (no children), do not rename
        if 'children' not in target_node or not target_node['children']:
            print(f"Cluster ID {cluster_id} is a leaf. Rename aborted.")
            return None

        # Rename the node
        target_node['name'] = new_name
<<<<<<< Updated upstream
        return self.Z_tree_format
=======
        return self.Z_tree_format
    
    def get_common_ancestor_subtree(self, labels, max_leaf_nodes=35):
        if self.Z_tree_format is None:
            raise ValueError("Dendrogram not loaded or empty")
        
        if not 2 <= len(labels) <= 4:
            raise ValueError("Number of labels must be between 2 and 4")
        
        # Find paths to each label
        paths = []
        for label in labels:
            path = self.find_name_hierarchy(self.Z_tree_format, label)
            if path is None:
                raise ValueError(f"Label '{label}' not found in dendrogram")
            paths.append(path)
        
        common_ancestor = None
        for i in range(len(paths[0])):
            current = paths[0][i]
            if all(current in path for path in paths):
                common_ancestor = current
                break
        
        if common_ancestor is None:
            raise ValueError("No common ancestor found for the specified labels")
        
        # Get subtree from common ancestor
        def find_subtree(node, target_name):
            if node.get('name') == target_name:
                return node
            if 'children' in node:
                for child in node['children']:
                    result = find_subtree(child, target_name)
                    if result:
                        return result
            return None
        
        subtree = find_subtree(self.Z_tree_format, common_ancestor)
        
        # Count leaf nodes and collect leaf labels
        def count_leaf_nodes_and_labels(node):
            if 'children' not in node:
                return 1, [node['name']]
            count = 0
            labels = []
            for child in node['children']:
                child_count, child_labels = count_leaf_nodes_and_labels(child)
                count += child_count
                labels.extend(child_labels)
            return count, labels
        
        leaf_count, leaf_labels = count_leaf_nodes_and_labels(subtree)
        
        if leaf_count > max_leaf_nodes:
            raise ValueError(f"Subtree contains {leaf_count} leaf nodes, which exceeds the maximum of {max_leaf_nodes}")
        
        return subtree, leaf_labels

def s3_file_exists(bucket_name: str, s3_key: str) -> bool:
    """Check if a file exists in S3"""
    s3_client = get_users_s3_client() 
    print(f"Checking if file exists in S3: {bucket_name}/{s3_key}")
    try:
        s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        print(f"File found: {bucket_name}/{s3_key}")
        return True
    except ClientError as e:
        print(f"File not found: {bucket_name}/{s3_key}, Error: {str(e)}")
        return False


def upload_json_to_s3(data: dict, bucket_name: str, s3_key: str):
    """Upload JSON data directly to S3"""
    s3_client = get_users_s3_client() 
    try:
        json_string = json.dumps(data, indent=2)
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=json_string.encode('utf-8'),
            ContentType='application/json'
        )
        logger.info(f"JSON data uploaded to s3://{bucket_name}/{s3_key}")
    except ClientError as e:
        logger.error(f"Error uploading JSON to S3: {e}")
        raise

def download_json_from_s3(bucket_name: str, s3_key: str) -> dict:
    """Download and parse JSON data from S3"""
    s3_client = get_users_s3_client() 
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        content = response['Body'].read().decode('utf-8')
        return json.loads(content)
    except ClientError as e:
        logger.error(f"Error downloading JSON from S3: {e}")
        raise

def upload_pickle_to_s3(data, bucket_name: str, s3_key: str):
    """Upload pickle data directly to S3"""
    s3_client = get_users_s3_client() 
    try:
        # Serialize to bytes
        pickle_bytes = pickle.dumps(data)
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=pickle_bytes,
            ContentType='application/octet-stream'
        )
        logger.info(f"Pickle data uploaded to s3://{bucket_name}/{s3_key}")
    except ClientError as e:
        logger.error(f"Error uploading pickle to S3: {e}")
        raise

def download_pickle_from_s3(bucket_name: str, s3_key: str):
    """Download and deserialize pickle data from S3"""
    s3_client = get_users_s3_client() 
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        pickle_bytes = response['Body'].read()
        return pickle.loads(pickle_bytes)
    except ClientError as e:
        logger.error(f"Error downloading pickle from S3: {e}")
        raise
>>>>>>> Stashed changes
