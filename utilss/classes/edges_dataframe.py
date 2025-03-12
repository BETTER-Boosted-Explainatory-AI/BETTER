import pandas as pd
import os

class EdgesDataframe:
    def __init__(self, model_filename, df_filename, edges_df=None):
        self.model_filename = model_filename
        self.edges_df = pd.DataFrame() if edges_df is None else edges_df
        self.df_filename = df_filename

    def get_dataframe(self):
        return self.edges_df

    def save_dataframe(self):
        self.edges_df.to_csv(self.df_filename, index=False)
        print(f'Edges dataframe has been saved: {self.df_filename}')

    def load_dataframe(self):
        if os.path.exists(self.df_filename):
            self.edges_df = pd.read_csv(self.df_filename)
            print(f'Edges dataframe has been loaded: {self.df_filename}')
        else:
            print(f'File not found: {self.df_filename}')

    def get_image_probabilities_by_id(self, image_id):
        probabilities_df = self.edges_df[self.edges_df['image_id'] == image_id]
        return probabilities_df

    def get_dataframe_by_count(self):
        return self.edges_df.groupby('source')['target'].value_counts().reset_index(name='count')