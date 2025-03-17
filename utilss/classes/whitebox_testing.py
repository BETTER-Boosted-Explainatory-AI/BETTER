import pandas as pd

class WhiteBoxTesting:
    def __init__(self, model_name):
        self.model_name = model_name
        self.problematic_imgs = None

    def find_problematic_images(self, source_labels, target_labels, edges_data):
        # Filter where source is in source_labels and target is in target_labels
        filtered_edges_ds = edges_data[
            (edges_data['source'].isin(source_labels)) & 
            (edges_data['target'].isin(target_labels))
        ]

        # Filter where source is in target_labels and target is in source_labels
        filtered_edges_ds_switched = edges_data[
            (edges_data['source'].isin(target_labels)) & 
            (edges_data['target'].isin(source_labels))
        ]

        # Combine both filtered datasets
        combined_filtered_edges_ds = pd.concat([filtered_edges_ds, filtered_edges_ds_switched])

        print(combined_filtered_edges_ds)
        image_counts = combined_filtered_edges_ds['image_id'].value_counts()
        self.problematic_imgs = image_counts.index.tolist()
        print("Counts of each image ID:", image_counts.to_dict())

        return image_counts.to_dict()

