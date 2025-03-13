class GraphBuilder:
    def __init__(self, graph_type, threshold, infinity):
        self.graph_type = graph_type
        self.threshold = threshold
        self.infinity = infinity
    
    def get_infinity(self):
        return self.infinity

    def get_threshold(self):
        return self.threshold
    
    def get_edge_weight(self, pred_prob):
        if self.graph_type == "dissimilarity":
            return 1 - pred_prob
        return pred_prob
    
    def should_edge_be_added(self, pred_label, target_label, pred_prob):
        return ((pred_label != target_label) and  (pred_prob > self.threshold))
    
    def update_graph(self, graph, pred_label, target_label, probability, image_id):
        weight = self.get_edge_weight(probability)
        
        if graph.are_adjacent(pred_label, target_label):
            edge_id = graph.get_eid(pred_label, target_label)
            graph.es[edge_id]["weight"] += weight
        else:
            graph.add_edge(pred_label, target_label, weight=weight)
            
        edge_data = {
            "image_id": image_id,
            "source": pred_label,
            "target": target_label,
            "target_probability": probability,
        }
        
        return edge_data
    
    def update_graph_dissimilarity(self, graph, shows_labels, label, target_label):
        if label not in shows_labels:
            if graph.are_adjacent(target_label, label):
                edge_id = graph.get_eid(target_label, label)
                graph.es[edge_id]["weight"] += self.infinity
            else:
                graph.add_edge(target_label, label, weight=self.infinity) 