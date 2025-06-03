"""
Standalone test script to test WordNet improvements on your actual S3 dendrogram.
This script downloads the dendrogram, tests improvements locally, and shows results.
Does NOT modify the original file in S3.
"""

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

import os
import json
import copy
from botocore.exceptions import ClientError

def download_dendrogram_from_s3():
    """Download the specific dendrogram from S3"""
    try:
        from utilss.s3_utils import get_users_s3_client
        
        s3_client = get_users_s3_client()
        bucket = "better-xai-users"
        key = "4225e444-c031-7077-1026-cb470f0a8a98/496105c6-8406-47eb-b2ff-00f0fd532d38/dissimilarity/dendrogram.json"
        
        print(f"📥 Downloading dendrogram from S3...")
        print(f"   Bucket: {bucket}")
        print(f"   Key: {key}")
        
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        dendrogram_data = json.loads(content)
        
        print(f"✅ Successfully downloaded dendrogram")
        print(f"   Root node name: {dendrogram_data.get('name', 'Unknown')}")
        print(f"   Root node ID: {dendrogram_data.get('id', 'Unknown')}")
        
        return dendrogram_data
        
    except ClientError as e:
        print(f"❌ Error downloading from S3: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def analyze_dendrogram_structure(data, max_depth=3):
    """Analyze and show the structure of the dendrogram"""
    print(f"\n📊 DENDROGRAM STRUCTURE ANALYSIS")
    print("=" * 50)
    
    def count_nodes(node, depth=0):
        """Count total nodes, clusters, and leaves"""
        if "children" not in node:
            return {"total": 1, "leaves": 1, "clusters": 0}
        
        stats = {"total": 1, "leaves": 0, "clusters": 1}
        for child in node["children"]:
            child_stats = count_nodes(child, depth + 1)
            stats["total"] += child_stats["total"]
            stats["leaves"] += child_stats["leaves"]  
            stats["clusters"] += child_stats["clusters"]
        
        return stats
    
    def show_structure(node, depth=0, max_depth=3):
        """Show tree structure up to max_depth"""
        indent = "  " * depth
        name = node.get("name", "Unknown")
        node_id = node.get("id", "N/A")
        value = node.get("value", "N/A")
        
        if "children" not in node:
            print(f"{indent}🍃 {name} (ID: {node_id})")
        else:
            print(f"{indent}📁 {name} (ID: {node_id}, Value: {value}) - {len(node['children'])} children")
            if depth < max_depth:
                for child in node["children"]:
                    show_structure(child, depth + 1, max_depth)
            elif depth == max_depth and node["children"]:
                print(f"{indent}  ... ({len(node['children'])} more children)")
    
    # Get statistics
    stats = count_nodes(data)
    print(f"📈 Statistics:")
    print(f"   Total nodes: {stats['total']}")
    print(f"   Leaf nodes: {stats['leaves']}")
    print(f"   Cluster nodes: {stats['clusters']}")
    
    # Show structure
    print(f"\n🌳 Tree Structure (showing up to depth {max_depth}):")
    show_structure(data, max_depth=max_depth)

def find_all_clusters(node):
    """Find all cluster nodes and their leaf names"""
    clusters = []
    
    def traverse(node):
        if "Cluster" in node.get("name", ""):
            from utilss.wordnet_utils import get_all_leaf_names
            leaf_names = get_all_leaf_names(node)
            clusters.append({
                "id": node.get("id"),
                "name": node.get("name"),
                "leaf_names": leaf_names,
                "leaf_count": len(leaf_names)
            })
        
        if "children" in node:
            for child in node["children"]:
                traverse(child)
    
    traverse(node)
    return clusters

def test_wordnet_on_dendrogram(original_data):
    """Test WordNet improvements on the dendrogram"""
    print(f"\n🧪 TESTING WORDNET IMPROVEMENTS")
    print("=" * 50)
    
    # Make a deep copy so we don't modify the original
    test_data = copy.deepcopy(original_data)
    
    # Find original clusters
    print(f"📋 BEFORE WordNet Processing:")
    original_clusters = find_all_clusters(test_data)
    print(f"   Found {len(original_clusters)} clusters with 'Cluster' names")
    
    for i, cluster in enumerate(original_clusters[:10]):  # Show first 10
        print(f"   {i+1}. {cluster['name']} - {cluster['leaf_count']} leaves: {cluster['leaf_names'][:5]}{'...' if len(cluster['leaf_names']) > 5 else ''}")
    
    if len(original_clusters) > 10:
        print(f"   ... and {len(original_clusters) - 10} more clusters")
    
    # Apply WordNet processing
    print(f"\n🔄 Applying WordNet Processing...")
    try:
        from utilss.wordnet_utils import process_hierarchy
        improved_data = process_hierarchy(test_data, debug=True)
        
        # Find new clusters
        print(f"\n✅ AFTER WordNet Processing:")
        new_clusters = find_all_clusters(improved_data)
        remaining_generic = [c for c in new_clusters if "Cluster" in c["name"]]
        renamed_clusters = len(original_clusters) - len(remaining_generic)
        
        print(f"   🎯 Successfully renamed: {renamed_clusters} clusters")
        print(f"   ⚠️  Still generic: {len(remaining_generic)} clusters")
        
        # Show some examples of renamed clusters
        if renamed_clusters > 0:
            print(f"\n🏆 Examples of Successfully Renamed Clusters:")
            
            # Find clusters that were renamed by comparing with original
            for orig_cluster in original_clusters[:5]:
                # Find the corresponding node in improved data
                def find_node_by_id(node, target_id):
                    if node.get("id") == target_id:
                        return node
                    if "children" in node:
                        for child in node["children"]:
                            result = find_node_by_id(child, target_id)
                            if result:
                                return result
                    return None
                
                improved_node = find_node_by_id(improved_data, orig_cluster["id"])
                if improved_node and improved_node.get("name") != orig_cluster["name"]:
                    print(f"   🔄 {orig_cluster['name']} → {improved_node.get('name')}")
                    print(f"      Leaves: {orig_cluster['leaf_names'][:3]}{'...' if len(orig_cluster['leaf_names']) > 3 else ''}")
        
        # Show remaining generic clusters
        if remaining_generic:
            print(f"\n⚠️  Clusters Still Using Generic Names:")
            for cluster in remaining_generic[:5]:
                print(f"   {cluster['name']} - {cluster['leaf_count']} leaves: {cluster['leaf_names'][:3]}{'...' if len(cluster['leaf_names']) > 3 else ''}")
        
        return improved_data
        
    except Exception as e:
        print(f"❌ Error during WordNet processing: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_specific_clusters(original_data):
    """Test WordNet processing on specific interesting clusters"""
    print(f"\n🎯 TESTING SPECIFIC CLUSTER TYPES")
    print("=" * 50)
    
    clusters = find_all_clusters(original_data)
    
    # Look for clusters with interesting leaf combinations
    interesting_clusters = []
    
    for cluster in clusters:
        leaves = cluster["leaf_names"]
        if len(leaves) >= 2:  # Only clusters with multiple leaves
            # Check for common categories
            trees = [l for l in leaves if any(word in l.lower() for word in ['tree', 'oak', 'pine', 'palm', 'maple', 'birch'])]
            animals = [l for l in leaves if any(word in l.lower() for word in ['dog', 'cat', 'lion', 'tiger', 'bear', 'wolf', 'fox'])]
            fruits = [l for l in leaves if any(word in l.lower() for word in ['apple', 'banana', 'orange', 'grape', 'berry'])]
            vehicles = [l for l in leaves if any(word in l.lower() for word in ['car', 'truck', 'bus', 'motorcycle', 'vehicle'])]
            
            if len(trees) >= 2:
                interesting_clusters.append({"cluster": cluster, "type": "Trees", "matches": trees})
            elif len(animals) >= 2:
                interesting_clusters.append({"cluster": cluster, "type": "Animals", "matches": animals})
            elif len(fruits) >= 2:
                interesting_clusters.append({"cluster": cluster, "type": "Fruits", "matches": fruits})
            elif len(vehicles) >= 2:
                interesting_clusters.append({"cluster": cluster, "type": "Vehicles", "matches": vehicles})
    
    if interesting_clusters:
        print(f"🔍 Found {len(interesting_clusters)} clusters with recognizable categories:")
        for item in interesting_clusters[:5]:
            cluster = item["cluster"]
            print(f"\n   📂 {cluster['name']} - Potential {item['type']}:")
            print(f"      All leaves: {cluster['leaf_names']}")
            print(f"      {item['type']} found: {item['matches']}")
            
            # Test WordNet on just this cluster's leaves
            print(f"      Testing WordNet on these leaves...")
            try:
                from utilss.wordnet_utils import find_common_hypernyms_improved
                result = find_common_hypernyms_improved(cluster['leaf_names'], 0, debug=False)
                print(f"      💡 Suggested name: {result if result else 'No suggestion found'}")
            except Exception as e:
                print(f"      ❌ Error: {e}")
    else:
        print("🤔 No obvious category clusters found. Testing with sample clusters...")
        
        # Test a few random clusters
        for cluster in clusters[:3]:
            if len(cluster['leaf_names']) >= 2:
                print(f"\n   📂 {cluster['name']}:")
                print(f"      Leaves: {cluster['leaf_names']}")
                try:
                    from utilss.wordnet_utils import find_common_hypernyms_improved
                    result = find_common_hypernyms_improved(cluster['leaf_names'], 0, debug=False)
                    print(f"      💡 Suggested name: {result if result else 'No suggestion found'}")
                except Exception as e:
                    print(f"      ❌ Error: {e}")

def save_results_locally(original_data, improved_data):
    """Save the test results locally for comparison"""
    try:
        # Save original
        with open("original_dendrogram.json", "w") as f:
            json.dump(original_data, f, indent=2)
        
        # Save improved (if it exists)
        if improved_data:
            with open("improved_dendrogram.json", "w") as f:
                json.dump(improved_data, f, indent=2)
        
        print(f"\n💾 Results saved locally:")
        print(f"   📄 original_dendrogram.json - Original from S3")
        if improved_data:
            print(f"   📄 improved_dendrogram.json - With WordNet improvements")
        print(f"   You can compare these files to see the differences")
        
    except Exception as e:
        print(f"❌ Error saving files: {e}")

def main():
    """Run the standalone test"""
    print("🚀 STANDALONE WORDNET TEST ON S3 DENDROGRAM")
    print("=" * 60)
    print("Testing on: s3://better-xai-users/.../dendrogram.json")
    print("This will NOT modify the original file in S3")
    print("=" * 60)
    
    # Download the dendrogram
    original_data = download_dendrogram_from_s3()
    if not original_data:
        print("❌ Could not download dendrogram. Exiting.")
        return
    
    # Analyze structure
    analyze_dendrogram_structure(original_data)
    
    # Test specific clusters
    test_specific_clusters(original_data)
    
    # Test WordNet improvements
    improved_data = test_wordnet_on_dendrogram(original_data)
    
    # Save results locally
    save_results_locally(original_data, improved_data)
    
    print(f"\n" + "=" * 60)
    print("✅ STANDALONE TEST COMPLETED")
    print("=" * 60)
    print("📊 Summary:")
    print("   - Downloaded dendrogram from S3 ✓")
    print("   - Analyzed structure ✓")  
    print("   - Tested WordNet improvements ✓")
    print("   - Saved results locally ✓")
    print("   - Original S3 file unchanged ✓")

if __name__ == "__main__":
    main()