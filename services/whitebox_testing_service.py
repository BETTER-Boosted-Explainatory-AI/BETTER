from utilss.classes.whitebox_testing import WhiteBoxTesting
from utilss.classes.edges_dataframe import EdgesDataframe

def get_white_box_analysis(model_name, source_labels, target_labels, edges_data_filename):
    edges_data = EdgesDataframe(model_name, edges_data_filename)
    edges_data.load_dataframe()
    
    whitebox_testing = WhiteBoxTesting(model_name)
    problematic_imgs = whitebox_testing.find_problematic_images(source_labels, target_labels, edges_data.edges_df)
    
    return problematic_imgs
