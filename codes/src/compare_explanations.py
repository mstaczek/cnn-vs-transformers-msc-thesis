from sklearn.metrics import cohen_kappa_score
import pandas as pd
import torch.nn.functional as F
from torch import exp
import torch
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
    flattening_dim = 1 if len(explanations_1.shape) == 3 else 0
    cosine_similarities = F.cosine_similarity(explanations_1.flatten(start_dim=flattening_dim), explanations_2.flatten(start_dim=flattening_dim), dim=flattening_dim)
    return cosine_similarities.mean().item()

def radial_basis_function(explanations_1, explanations_2, sigma=10):
    """
        in: explanations_1, explanations_2 - torch.tensors of same dimensions, each row is an explanation
        out: radial basis function similarity of explanations_1 and explanations_2
    """
    flattening_dim = 1 if len(explanations_1.shape) == 3 else 0
    squared_distances = (explanations_1.flatten(start_dim=flattening_dim) - explanations_2.flatten(start_dim=flattening_dim)).pow(2).sum(dim=flattening_dim)
    rbf_similarities = exp(-0.5 * squared_distances / sigma**2)
    return rbf_similarities.mean().item()

def cosine_similarity_distance_with_stdev_and_mean(explanations_1, explanations_2):
    """
        in: explanations_1, explanations_2 - torch.tensors of same dimensions, each row is an explanation
        out: distance from (0,0) to (1-mean, stdev) of cosine similarities
    """
    flattening_dim = 1 if len(explanations_1.shape) == 3 else 0
    cosine_similarities = F.cosine_similarity(explanations_1.flatten(start_dim=flattening_dim), explanations_2.flatten(start_dim=flattening_dim), dim=flattening_dim)    
    distance = sqrt((1 - cosine_similarities.mean().item())**2 + cosine_similarities.std().item()**2)
    return distance

def radial_basis_function_distance_with_stdev_and_mean(explanations_1, explanations_2, sigma=80):
    """
        in: explanations_1, explanations_2 - torch.tensors of same dimensions, each row is an explanation
        out: distance from (0,0) to (1-mean, stdev) of RBF similarities
    """
    flattening_dim = 1 if len(explanations_1.shape) == 3 else 0
    squared_distances = (explanations_1.flatten(start_dim=flattening_dim) - explanations_2.flatten(start_dim=flattening_dim)).pow(2).sum(dim=flattening_dim)
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

def _count_greater_how_many_thresholds(vector, thresholds):
    counter_vector = torch.zeros_like(vector)
    for t in thresholds:
        counter_vector += (vector > t).float()
    return counter_vector

def cohens_kappa_metric(explanations_1, explanations_2, thresholds=[0.5]):
    """
        in: explanations_1, explanations_2 - torch.tensors of same dimensions, each row is an explanation
        out: cohen_kappa_score of explanations_1 and explanations_2
    """
    binary_flattened_1 = _count_greater_how_many_thresholds(explanations_1.flatten(), thresholds)
    binary_flattened_2 = _count_greater_how_many_thresholds(explanations_2.flatten(), thresholds)
    result = cohen_kappa_score(binary_flattened_1, binary_flattened_2)
    return result
