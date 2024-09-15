import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# TODO: add custom metric to NN model
# def my_metric(point1, point2):
#     x1 = point1[:-1]
#     x2 = point2[:-1]
#     y1 = point1[-1]
#     y2 = point2[-1]
#     x_dist = np.linalg.norm(x1 - x2)
#     y_dist = np.abs(y1 - y2)
#     if y_dist < 1:
#         return x_dist
#     else:
#         return np.inf

def get_neighbors(X, index_list, k, num_of_samples):
    if index_list is None and num_of_samples is None:
        index_list = np.arange(len(X))
    elif index_list is None:
        num_of_samples = int(num_of_samples)
        index_list = np.random.choice(np.arange(len(X)), num_of_samples, replace=True)

    xs = X.iloc[index_list, :]
    neighbors_model = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X)

    # Getting neighbors for all indices in index_list
    _, neighbors_indices = neighbors_model.kneighbors(xs)
    neighbors_indices = neighbors_indices[:, 1:]

    return neighbors_indices, index_list


def choose_neighbors(neighbors_indices, m):
    # For each x, choose m neighbors without replacement
    chosen_neighbors_indices = np.array(
        [np.random.choice(neighbors_indices[i], m, replace=False) for i in
         range(len(neighbors_indices))])
    return chosen_neighbors_indices


def collect_neighbors_data(X, Y, chosen_neighbors_indices):
    # Collect corresponding neighbors' X and Y
    chosen_neighbors_xs = np.array(
        [X.iloc[chosen_neighbors_indices[i]] for i in range(len(chosen_neighbors_indices))])
    chosen_neighbors_ys = np.array(
        [Y.iloc[chosen_neighbors_indices[i]] for i in range(len(chosen_neighbors_indices))])
    return chosen_neighbors_xs, chosen_neighbors_ys


def generate_random_weights(alpha_lim, length, m):
    alphas = np.random.random(size=length) * alpha_lim
    random_weights = np.array([np.random.rand(m) for _ in range(length)])
    random_weights /= np.array([random_weights[i].sum() / alphas[i] for i in range(length)])[:,
                      None]
    return random_weights, alphas


def generate_new_samples(X, Y, index_list=None, k=5, m=5, alpha_lim=0.1, num_of_samples=None):
    neighbors_indices, index_list = get_neighbors(X, index_list, k, num_of_samples)
    chosen_neighbors_indices = choose_neighbors(neighbors_indices, m)
    chosen_neighbors_xs, chosen_neighbors_ys = collect_neighbors_data(X, Y,
                                                                      chosen_neighbors_indices)
    random_weights, alphas = generate_random_weights(alpha_lim, len(index_list), m)

    # Adjusted einsum operation
    diff_xs = np.einsum('ij,ijk->ik', random_weights, chosen_neighbors_xs)
    diff_ys = np.einsum('ij,ij->i', random_weights, chosen_neighbors_ys)

    new_samples_x = X.iloc[index_list] * (1 - alphas[:, None]) + diff_xs
    new_samples_y = Y.iloc[index_list] * (1 - alphas) + diff_ys

    new_samples_x = pd.DataFrame(new_samples_x, columns=X.columns).reset_index(drop=True)
    new_samples_y = pd.Series(new_samples_y, name=Y.name).reset_index(drop=True)

    return new_samples_x, new_samples_y
