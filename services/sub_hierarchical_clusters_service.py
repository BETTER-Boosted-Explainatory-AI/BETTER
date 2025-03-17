import numpy as np

def _get_sub_heirarchical_clustering(dataset_str, labels):

    labels = sorted(labels)  # Ensure sorted order
    class_index = {cls: i for i, cls in enumerate(labels)}  # Map class names to indices

    # Step 1: Initialize a zero matrix
    confusion_matrix = np.zeros((len(labels), len(labels)))

    # Step 3: Populate only top 5 misclassifications per source
    for source, group in count_per_target.groupby('source'):
        top_5 = group.nlargest(5, 'count')  # Get the top 5 misclassifications for this source

        for _, row in top_5.iterrows():
            target, count = row['target'], row['count']
            if source in class_index and target in class_index:  # Ensure both exist in class_names
                i, j = class_index[source], class_index[target]
                confusion_matrix[i, j] = count

    return confusion_matrix