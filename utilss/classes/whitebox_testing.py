class WhiteBoxTesting:
    def __init__(self, model_name):
        self.model_name = model_name
        self.problematic_imgs = None

    def find_problematic_images(self, source_labels, target_labels, edges_data):
        filtered_edges_ds = edges_data[
            (edges_data['source'].isin(source_labels)) & 
            (edges_data['target'].isin(target_labels))
        ]  

        print(filtered_edges_ds)
        self.problematic_imgs = filtered_edges_ds['image_id'].tolist()
        print("Array of image ID's:", self.problematic_imgs)

        return self.problematic_imgs

