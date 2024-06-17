import pandas as pd
import torch.nn.functional as F
from torch import exp
from math import sqrt


def compare_explanations(explanations_list: list[dict], comparison_function, compare_only_explanations_with_same_predictions=False):
    """
        in: list of dictionaries with keys 'explanations', 'paths', 'labels', 'model_name', 'explanation_name', 'predictions'
        out: dataframe models x models with comparison_function applied to each pair of model explanations
    """
    models = [data['model_name'] for data in explanations_list]
    similarity_df = pd.DataFrame(index=models, columns=models)

    for i in range(len(explanations_list)):
        for j in range(i, len(explanations_list)):
            if compare_only_explanations_with_same_predictions and sum(explanations_list[i]['predictions'] != explanations_list[j]['predictions']) != 0:
                indices_of_matching_prediction = explanations_list[i]['predictions'] == explanations_list[j]['predictions']
                explanations_i = explanations_list[i]['explanations'][indices_of_matching_prediction]
                explanations_j = explanations_list[j]['explanations'][indices_of_matching_prediction]
            else:
                explanations_i = explanations_list[i]['explanations']
                explanations_j = explanations_list[j]['explanations']
            
            if len(explanations_i) == 0 or len(explanations_j) == 0:
                similarity_ij = 0
            else:
                similarity_ij = comparison_function(explanations_i, explanations_j)
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
    cosine_similarities = F.cosine_similarity(explanations_1.flatten(start_dim=1), explanations_2.flatten(start_dim=1), dim=1)
    return cosine_similarities.mean().item()

def radial_basis_function(explanations_1, explanations_2, sigma=10):
    """
        in: explanations_1, explanations_2 - torch.tensors of same dimensions, each row is an explanation
        out: radial basis function similarity of explanations_1 and explanations_2
    """
    squared_distances = (explanations_1.flatten(start_dim=1) - explanations_2.flatten(start_dim=1)).pow(2).sum(dim=1)
    rbf_similarities = exp(-0.5 * squared_distances / sigma**2)
    return rbf_similarities.mean().item()

def cosine_similarity_distance_with_stdev_and_mean(explanations_1, explanations_2):
    """
        in: explanations_1, explanations_2 - torch.tensors of same dimensions, each row is an explanation
        out: distance from (0,0) to (1-mean, stdev) of cosine similarities
    """
    cosine_similarities = F.cosine_similarity(explanations_1.flatten(start_dim=1), explanations_2.flatten(start_dim=1), dim=1)    
    distance = sqrt((1 - cosine_similarities.mean().item())**2 + cosine_similarities.std().item()**2)
    return distance

def radial_basis_function_distance_with_stdev_and_mean(explanations_1, explanations_2, sigma=10):
    """
        in: explanations_1, explanations_2 - torch.tensors of same dimensions, each row is an explanation
        out: distance from (0,0) to (1-mean, stdev) of RBF similarities
    """
    squared_distances = (explanations_1.flatten(start_dim=1) - explanations_2.flatten(start_dim=1)).pow(2).sum(dim=1)
    rbf_similarities = exp(-0.5 * squared_distances / sigma**2)
    distance = sqrt((1 - rbf_similarities.mean().item())**2 + rbf_similarities.std().item()**2)
    return distance

def count_same_predictions(explanations_list: list[dict]):
    """
        in: list of dictionaries with keys 'explanations', 'paths', 'labels', 'model_name', 'explanation_name', 'predictions'
        out: dataframe models x models with count of same predictions
    """
    models = [data['model_name'] for data in explanations_list]
    count_of_same_predictions_df = pd.DataFrame(index=models, columns=models)

    for i in range(len(explanations_list)):
        for j in range(i, len(explanations_list)):
            count_of_same_predictions = sum(explanations_list[i]['predictions'] == explanations_list[j]['predictions']).item()
            model_i = explanations_list[i]['model_name']
            model_j = explanations_list[j]['model_name']
            count_of_same_predictions_df.loc[model_i, model_j] = count_of_same_predictions
            count_of_same_predictions_df.loc[model_j, model_i] = count_of_same_predictions
    
    return count_of_same_predictions_df