from nltk.corpus import wordnet as wn
import os

def convert_folder_name_to_label(folder_name):
    synsets = wn.synset_from_pos_and_offset('n', int(folder_name[1:]))
    return synsets.lemma_names()[0]

def convert_folder_names_to_readable_labels(dataset_path):
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    class_names = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d)) and d.startswith('n') and d[1:].isdigit()
    ])
    
    if not class_names:
        raise ValueError(f"No subdirectories found in {dataset_path}")
    
    special_case_mapping = {
        "n02012849": "crane_bird",       # Crane bird
        "n03126707": "crane_machine",    # Crane machine
        "n03710637": "maillot",          # Maillot (swimsuit)
        "n03710721": "tank_suit"         # Tank suit (different type of swimsuit)
    }
    
    # Create mapping from folder names to readable labels
    folder_to_label = {}
    for folder_name in class_names:
        if folder_name in special_case_mapping:
            readable_label = special_case_mapping[folder_name]
        elif folder_name.startswith('n'):
            readable_label = convert_folder_name_to_label(folder_name)
        else:
            readable_label = folder_name
            
        folder_to_label[folder_name] = readable_label
    
    readable_labels = [folder_to_label[folder_name] for folder_name in class_names]
    
    return readable_labels, folder_to_label


def folder_name_to_number(folder_name):
    synsets = wn.synsets(folder_name)
    if synsets:
        offset = synsets[0].offset()        
        folder_number = 'n{:08d}'.format(offset)
        return folder_number
    
def common_group(groups):
    common_hypernyms = []
    hierarchy = {}
    
    for group in groups:
        hierarchy[group] = []
        synsets = wn.synsets(group)
        if synsets:
            hypernyms = synsets[0].hypernym_paths()
            for path in hypernyms:
                hierarchy[group].extend([node.name().split('.')[0] for node in path])
                
    if len(hierarchy) == 1:
        common_hypernyms = list(hierarchy.values())[0]
    else:
        for hypernym in hierarchy[groups.pop()]:
            if all(hypernym in hypernyms for hypernyms in hierarchy.values()):
                common_hypernyms.append(hypernym)
    
    return common_hypernyms[::-1]


def get_all_leaf_names(node):
    """Extract all leaf node names from a cluster hierarchy."""
    if "children" not in node:
        # Only return actual object names, not cluster names
        if "Cluster" not in node["name"]:
            return [node["name"]]
        return []
    
    names = []
    for child in node["children"]:
        names.extend(get_all_leaf_names(child))
    return names

def find_common_hypernyms(words):
    """Find common hypernyms for a list of words using multiple strategies."""
    # Clean words
    clean_words = [word.replace('_', ' ').lower() for word in words if "Cluster" not in word]
    
    if len(clean_words) < 2:
        return None
    
    # Strategy 1: Try direct WordNet hypernyms
    synsets_list = []
    for word in clean_words:
        synsets = wn.synsets(word, pos=wn.NOUN)
        if synsets:
            synsets_list.append(synsets[0])  # Take first synset (most common)
    
    # Find common hypernyms if we have at least 2 synsets
    if len(synsets_list) >= 2:
        all_hypernyms = []
        for i in range(len(synsets_list) - 1):
            for j in range(i + 1, len(synsets_list)):
                lchs = synsets_list[i].lowest_common_hypernyms(synsets_list[j])
                for lch in lchs:
                    if lch.name() != 'entity.n.01':  # Skip extremely general hypernyms
                        all_hypernyms.append((lch, lch.min_depth()))
        
        if all_hypernyms:
            # Sort by depth (more specific hypernyms have greater depth)
            all_hypernyms.sort(key=lambda x: x[1], reverse=True)
            return all_hypernyms[0][0].name().split('.')[0].replace('_', ' ')

def improved_rename_clusters(node, depth=0):
    """Rename clusters based on their contents, with names becoming more general at higher levels."""
    # Process children first (bottom-up approach)
    if "children" in node:
        for i, child in enumerate(node["children"]):
            node["children"][i] = improved_rename_clusters(child, depth + 1)
    
    # Only rename nodes that have "Cluster" in their name
    if "Cluster" in node["name"]:
        # Collect all leaf names under this cluster
        leaf_names = get_all_leaf_names(node)
        
        # If no leaf names (rare case), try using child cluster names
        if not leaf_names and "children" in node:
            child_names = [child["name"] for child in node["children"] if "Cluster" not in child["name"]]
            if child_names:
                leaf_names = child_names
        
        # Try to find a common hypernym or category
        new_name = None
        
        # Adjust naming strategy based on depth
        if leaf_names:
            new_name = find_common_hypernyms(leaf_names) 
        
        # Apply the new name if found
        if new_name:
            node["name"] = new_name.capitalize()
    
    return node

def process_hierarchy(hierarchy_data):
    """Process the entire hierarchy, renaming clusters while preserving structure."""
    return improved_rename_clusters(hierarchy_data)
