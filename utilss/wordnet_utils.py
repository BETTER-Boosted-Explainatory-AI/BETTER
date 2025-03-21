from nltk.corpus import wordnet as wn

def convert_folder_name_to_label(folder_name):
    synsets = wn.synset_from_pos_and_offset('n', int(folder_name[1:]))
    return synsets.lemma_names()[0]

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


def get_lowest_common_hypernyms(words):
    """Find the lowest common hypernym (category) for a list of words using WordNet."""
    # Get all synsets for each word
    synsets_list = []
    for word in words:
        # Clean up the word (remove underscores, etc.)
        clean_word = word.replace('_', ' ').lower()
        synsets = wn.synsets(clean_word)
        if synsets:
            synsets_list.append(synsets[0])  # Take the first synset as most common meaning
    
    if len(synsets_list) < 2:
        return None
    
    # Find lowest common hypernyms
    common_hypernyms = []
    for i in range(len(synsets_list) - 1):
        for j in range(i + 1, len(synsets_list)):
            hypernyms = synsets_list[i].lowest_common_hypernyms(synsets_list[j])
            if hypernyms and hypernyms[0].name() != 'entity.n.01':  # Skip very generic hypernyms
                common_hypernyms.extend(hypernyms)
    
    # Return most specific hypernym (lowest in hierarchy)
    if common_hypernyms:
        # Sort by information content (more specific = higher value)
        common_hypernyms.sort(key=lambda h: h.min_depth(), reverse=True)
        return common_hypernyms[0].name().split('.')[0].replace('_', ' ')
    
    return None

def get_leaf_names(node):
    """Extract all leaf node names from a cluster."""
    if "children" not in node:
        return [node["name"]]
    
    names = []
    for child in node["children"]:
        names.extend(get_leaf_names(child))
    return names

def rename_clusters(node):
    """Recursively rename clusters based on their children."""
    # Skip leaf nodes
    if "children" not in node:
        return node
    
    # First, recursively process all children
    for i, child in enumerate(node["children"]):
        node["children"][i] = rename_clusters(child)
    
    # If this is a cluster node (has "Cluster" in name)
    if "Cluster" in node["name"]:
        # Get direct child names for immediate categorization
        direct_child_names = [child["name"] for child in node["children"] 
                             if "Cluster" not in child["name"]]
        
        # If direct children are not clusters themselves
        if direct_child_names:
            new_name = get_lowest_common_hypernym(direct_child_names)
            if new_name:
                node["name"] = new_name.capitalize()
                # node["original_cluster_id"] = node["name"]  # Preserve original ID
        else:
            # Get all leaf node names from all children
            all_leaf_names = get_leaf_names(node)
            new_name = get_lowest_common_hypernym(all_leaf_names)
            if new_name:
                node["name"] = new_name.capitalize()
                # node["original_cluster_id"] = node["name"]  # Preserve original ID
    
    return node

def get_lowest_common_hypernym(words):
    """Find a suitable category name for a list of words."""
    # Filter out any cluster names
    real_words = [w for w in words if "Cluster" not in w]
    
    if len(real_words) < 2:
        return None
    
    # Try WordNet's lowest common hypernym
    hypernym = get_lowest_common_hypernyms(real_words)
    
    # Custom handling for specific categories
    # word_set = set(w.lower().replace('_', ' ') for w in real_words)
    
    # # Custom rules for specific categories
    # if all(w in ['persian cat', 'tabby', 'egyptian cat'] for w in word_set):
    #     return "domestic cats"
    # elif all(w in ['chimpanzee', 'gorilla', 'spider monkey'] for w in word_set):
    #     return "primates"
    # elif all(w in ['wok', 'frying pan', 'caldron', 'teapot', 'coffeepot', 'crock pot'] for w in word_set):
    #     return "cookware"
    # elif all(w in ['broccoli', 'cauliflower', 'head cabbage', 'zucchini'] for w in word_set):
    #     return "vegetables"
    # elif all(w in ['orange', 'lemon', 'fig', 'granny smith'] for w in word_set):
    #     return "fruits"
    # elif all(w in ['airliner', 'warplane', 'space shuttle'] for w in word_set):
    #     return "aircraft"
    # elif all(w in ['minivan', 'police van', 'limousine', 'jeep', 'sports car'] for w in word_set):
    #     return "vehicles"
    # elif all(w in ['american coot', 'black swan', 'white stork', 'flamingo'] for w in word_set):
    #     return "birds"
    # elif all(w in ['catamaran', 'trimaran', 'container ship', 'fireboat'] for w in word_set):
    #     return "watercraft"
    
    return hypernym
