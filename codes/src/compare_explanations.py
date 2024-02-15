import pandas as pd
import torch.nn.functional as F


def compare_explanations(explanations_list: list[dict], comparison_function):
    """
        in: list of dictionaries with keys 'explanations', 'paths', 'labels', 'model_name', 'explanation_name'
        out: dataframe models x models with comparison_function applied to each pair of model explanations
    """
    models = [data['model_name'] for data in explanations_list]
    similarity_df = pd.DataFrame(index=models, columns=models)

    for i in range(len(explanations_list)):
        for j in range(i, len(explanations_list)):
            if i == j:
                similarity_ij = 1
            else:
                similarity_ij = comparison_function(explanations_list[i]['explanations'], explanations_list[j]['explanations'])
            model_i = explanations_list[i]['model_name']
            model_j = explanations_list[j]['model_name']
            similarity_df.loc[model_i, model_j] = similarity_ij
            similarity_df.loc[model_j, model_i] = similarity_ij
    
    return similarity_df

def cosine_similarity(explanations_1, explanations_2):
    """
        in: explanations_1, explanations_2 - torch.tensors of same dimensions, each row is an explanation
        out: cosine similarity of explanations_1 and explanations_2
    """
    explanations_1_flattened = explanations_1.flatten(start_dim=1)
    explanations_2_flattened = explanations_2.flatten(start_dim=1)
    cosine_similarities = F.cosine_similarity(explanations_1_flattened, explanations_2_flattened, dim=1)
    return cosine_similarities.mean().item()