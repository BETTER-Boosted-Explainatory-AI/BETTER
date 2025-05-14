import pandas as pd
from utilss.enums.datasets_enum import DatasetsEnum
from data.datasets.imagenet_info import IMAGENET_INFO

class WhiteBoxTesting:
    def __init__(self, model_name):
        self.model_name = model_name
        self.problematic_imgs = None

    def find_problematic_images(self, source_labels, target_labels, edges_df, dataset_str):
        mapped_sources = source_labels
        mapped_targets = target_labels
        
        if dataset_str == DatasetsEnum.IMAGENET.value:
            # For ImageNet, we need to map directory labels to readable labels
            readable_to_dir = IMAGENET_INFO["readable_to_directory"]
            mapped_sources = [readable_to_dir[label] for label in source_labels]
            mapped_targets = [readable_to_dir[label] for label in target_labels]
            print("Mapped sources:", mapped_sources)
            print("Mapped targets:", mapped_targets)
             

        # Filter where source is in source_labels and target is in target_labels
        filtered_edges_ds = edges_df[
            (edges_df['source'].isin(mapped_sources)) & 
            (edges_df['target'].isin(mapped_targets))
        ]

        # Filter where source is in target_labels and target is in source_labels
        filtered_edges_ds_switched = edges_df[
            (edges_df['source'].isin(mapped_sources)) & 
            (edges_df['target'].isin(mapped_targets))
        ]
        print(filtered_edges_ds_switched.head())
        

        # Combine both filtered datasets
        combined_filtered_edges_ds = pd.concat([filtered_edges_ds, filtered_edges_ds_switched])
        print("Combined filtered edges dataset:")
        print(combined_filtered_edges_ds)
        if dataset_str == DatasetsEnum.IMAGENET.value:
            dir_to_readable = IMAGENET_INFO["directory_to_readable"]
            combined_filtered_edges_ds['source'] = combined_filtered_edges_ds['source'].map(dir_to_readable)
            combined_filtered_edges_ds['target'] = combined_filtered_edges_ds['target'].map(dir_to_readable)


        print(combined_filtered_edges_ds)
        image_counts = combined_filtered_edges_ds['image_id'].value_counts()
        self.problematic_imgs = image_counts.index.tolist()
        print("Counts of each image ID:", image_counts.to_dict())

        return image_counts.to_dict()
