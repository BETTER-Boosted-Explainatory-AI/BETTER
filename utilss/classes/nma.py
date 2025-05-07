import tensorflow as tf
import pandas as pd
from igraph import Graph
from utilss.enums.heap_types import HeapType
from .preprocessing.batch_predictor import BatchPredictor
from .preprocessing.heap_processor import HeapProcessor
from .preprocessing.graphs.graph_builder import GraphBuilder
from .preprocessing.hierarchical_clustering_builder import HierarchicalClusteringBuilder
from .preprocessing.z_builder import ZBuilder


class NMA:
    def __init__(
        self,
        model,
        X,
        y,
        labels,
        graph_type="similarity",
        top_k=4,
        graph_threshold=1e-6,
        infinity=10,
        min_confidence=0.8,
        save_connections=True,
        batch_size=32,
    ):
        """
        X: images array,
        y: labels,
        model: keras model,
        graph_type: similarity/dissimilarity
        k: top predictions
        t: top prediction thresholds to take into account
        Preprossing the nma
        """
        if not isinstance(model, tf.keras.Model):
            raise TypeError("model must be an instance of tf.keras.Model")

        self.model = model
        self.save_connections = save_connections
        self.graph_type = graph_type
        self.top_k = top_k
        self.graph_threshold = graph_threshold
        self.infinity = infinity
        self.min_confidence = min_confidence
        self.labels = labels
        self.edges_df = None
        self.Z = None

        if graph_type == "dissimilarity":
            self.heap_type = HeapType.MINIMUM.value
        elif graph_type == "similarity" or graph_type == "count":
            self.heap_type = HeapType.MAXIMUM.value

        self.TBD_graph = None

        self._preprocessing(X, y, batch_size)

    def _get_image_probabilities_by_id(self, image_id):
        probabilities_df = self.edges_df[self.edges_df["image_id"] == image_id]
        return probabilities_df

    def _get_dataframe_by_count(self):
        return (
            self.edges_df.groupby("source")["target"]
            .value_counts()
            .reset_index(name="count")
        )

    def _preprocessing(self, X, y, batch_size):
        try:
            graph = Graph(directed=False)
            graph.add_vertices(self.labels)

            edges_data = []
            batch_images = []
            batch_labels = []

            predictor = BatchPredictor(self.model)
            builder = GraphBuilder(self.graph_type, self.infinity)
            for i, image in enumerate(X):
                source_label = y[i]

                batch_images.append(image)
                batch_labels.append(source_label)

                if len(batch_images) == predictor.batch_size or i == len(X) - 1:
                    top_predictions_batch = predictor.get_top_predictions(
                        batch_images, self.labels, self.top_k, self.graph_threshold
                    )

                    added_labels = []
                    for j, top_predictions in enumerate(top_predictions_batch):
                        current_label = batch_labels[j]

                        if len(top_predictions) == 0:
                            print(top_predictions)
                            continue

                        if len(top_predictions[0]) < 2:
                            continue

                        if top_predictions[0][2] > self.min_confidence:
                            filtered_predictions = top_predictions

                            for _, pred_label, pred_prob in filtered_predictions:
                                if pred_label not in self.labels:
                                    raise ValueError(
                                        f"Prediction label '{pred_label}' not in graph labels."
                                    )

                                if current_label != pred_label:
                                    edge_data = builder.update_graph(
                                        graph, current_label, pred_label, pred_prob, i
                                    )
                                    # Only append edge_data if it's not None (not a self-loop)
                                    if edge_data is not None:
                                        edges_data.append(edge_data)
                                        added_labels.append(pred_label)

                        if self.graph_type == "dissimilarity":
                            # for label in unique_labels_in_y:
                            for label in self.labels:
                                if label != current_label:
                                    builder.add_infinity_edges(
                                        graph, added_labels, label, current_label
                                    )

                    batch_images = []
                    batch_labels = []

            if self.save_connections:
                self.edges_df = pd.DataFrame(edges_data)

            self.TBD_graph = graph
            heap_processor = HeapProcessor(self.TBD_graph, self.graph_type, self.labels)
            self.heap_processor = heap_processor

            clustering = HierarchicalClusteringBuilder(heap_processor, self.labels)
            self.Z = ZBuilder.create_z_matrix_from_tree(clustering, self.labels)

        except Exception as e:
            print(f"Error while preprocessing model: {str(e)}")

    def get_neighbors_by_label_name(self, label_name):
        neighbors = self.TBD_graph.neighbors(label_name)
        neighbors_names = [self.TBD_graph.vs[n]["name"] for n in neighbors]

        data = [
            {
                "Neighbor": neighbor,
                "Weight": self.TBD_graph.es[
                    self.TBD_graph.get_eid(label_name, neighbor)
                ]["weight"],
            }
            for neighbor in neighbors_names
        ]

        return pd.DataFrame(data).sort_values(by="Weight", ascending=False)
