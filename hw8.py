import numpy as np
import pandas as pd

RI3 = 0.58
RI4 = 0.90

def get_max_eig(mat):
    assert mat.shape[0] == mat.shape[1] and mat.shape[0] > 1
    e_values, e_vecs = np.linalg.eig(mat)
    max_idx = np.argmax(e_values)
    e_value = e_values[max_idx].real
    e_vec = np.abs(e_vecs[:, max_idx].real)
    sum = np.sum(e_vec)
    e_vec = e_vec / sum
    CR = (e_value - mat.shape[0]) / (mat.shape[0] - 1)
    return CR, e_vec

if __name__ == '__main__':
    weight_mat1 = np.array([[1, 2, 1 / 5],
                            [1 / 2, 1, 1 / 9],
                            [5, 9, 1]])
    weight_mat2 = np.array([[1, 1 / 2, 1 / 5],
                            [2, 1, 1 / 3],
                            [5, 3, 1]])
    factor_mat1 = np.array([[1, 1 / 2, 1 / 2, 1 / 7],
                            [2, 1, 1, 1 / 4],
                            [2, 1, 1, 1 / 4],
                            [7, 4, 4, 1]])
    factor_mat2 = np.array([[1, 1 / 2, 1 / 2, 2],
                            [2, 1, 1, 4],
                            [2, 1, 1, 3],
                            [1 / 2, 1 / 4, 1 / 3, 1]])
    factor_mat3 = np.array([[1, 1, 1, 2],
                            [1, 1, 1, 2],
                            [1, 1, 1, 2],
                            [1 / 2, 1 / 2, 1 / 2, 1]])
    factor_mat4 = np.array([[1, 2, 1, 4],
                            [1 / 2, 1, 1 / 2, 2],
                            [1, 2, 1, 3],
                            [1 / 4, 1 / 2, 1 / 3, 1]])
    factor_mat5 = np.array([[1, 3, 1, 5],
                            [1 / 3, 1, 1 / 3, 2],
                            [1, 3, 1, 6],
                            [1 / 5, 1 / 2, 1 / 6, 1]])

    weight1_value, weight1_vec = get_max_eig(weight_mat1)
    weight2_value, weight2_vec = get_max_eig(weight_mat2)
    weight2_vec = weight2_vec * weight1_vec[2]
    weight = np.zeros(5)
    weight[0: 2] = weight1_vec[0: 2]
    weight[2:] = weight2_vec

    factor1_value, factor1_vec = get_max_eig(factor_mat1)
    factor2_value, factor2_vec = get_max_eig(factor_mat2)
    factor3_value, factor3_vec = get_max_eig(factor_mat3)
    factor4_value, factor4_vec = get_max_eig(factor_mat4)
    factor5_value, factor5_vec = get_max_eig(factor_mat5)

    factor_mat = np.zeros((4, 5))
    factor_mat[:, 0] = factor1_vec
    factor_mat[:, 1] = factor2_vec
    factor_mat[:, 2] = factor3_vec
    factor_mat[:, 3] = factor4_vec
    factor_mat[:, 4] = factor5_vec

    score = np.sum(factor_mat * weight, axis=1)
    rank = np.empty(score.shape[0], dtype=np.int64)
    rank[np.argsort(score)[::-1]] = (np.arange(len(score)) + 1)

    print(weight1_vec)
    print(weight2_vec)
    print('weight:')
    print(weight)
    print('factor_mat:')
    print(factor_mat)
    print('score:')
    print(score)
    print('rank:')
    print(rank)
    print('C.R. = C.I./R.I.:')
    print([weight1_value / RI3, weight2_value / RI3, factor1_value / RI4,
           factor2_value / RI4, factor3_value / RI4, factor4_value / RI4,
           factor5_value / RI4])

    '''pd.DataFrame(weight_mat1).to_csv('weight_mat1.csv')
    pd.DataFrame(weight_mat2).to_csv('weight_mat2.csv')
    pd.DataFrame(factor_mat1).to_csv('factor_mat1.csv')
    pd.DataFrame(factor_mat2).to_csv('factor_mat2.csv')
    pd.DataFrame(factor_mat3).to_csv('factor_mat3.csv')
    pd.DataFrame(factor_mat4).to_csv('factor_mat4.csv')
    pd.DataFrame(factor_mat5).to_csv('factor_mat5.csv')
    pd.DataFrame(factor_mat).to_csv('factor_mat.csv')
    pd.DataFrame(np.array([weight1_value / RI3, weight2_value / RI3, factor1_value / RI4,
           factor2_value / RI4, factor3_value / RI4, factor4_value / RI4,
           factor5_value / RI4])).to_csv('cr.csv')'''

