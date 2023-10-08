import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn.base import clone


def plot_pred_vs_actual(predicted: pd.DataFrame, train_y_df: pd.DataFrame, estimator_name: str = ''):
    print('\nPlot prediction versus actual target values:\n')

    relative_rmse = mean_squared_error(train_y_df, predicted, squared=False) / train_y_df.mean()

    try:
        relative_rmse = np.round(relative_rmse, 5)[0]
    except IndexError:
        relative_rmse = np.round(relative_rmse, 5)

    plt.scatter(train_y_df, predicted)
    slope = 1.0
    intercept = 0

    try:
        line_values = [slope * x_value + intercept for x_value in train_y_df.values]
    except AttributeError:
        line_values = [slope * x_value + intercept for x_value in train_y_df]

    plt.plot(train_y_df, line_values, 'b')
    plt.grid()
    plt.xlabel('actual')
    plt.ylabel('predicted')
    plt.title(f'{estimator_name}\nrel_rmse: {relative_rmse}')
    plt.show()


def check_pred_performance(predicted: pd.DataFrame, train_y_df: pd.DataFrame,
                           model_selection_stage='', estimator_name='',
                           data_set_type=''):

    print(f'{model_selection_stage} {estimator_name} estimator prediction performance on the {data_set_type} data set: ')

    plot_pred_vs_actual(predicted, train_y_df, estimator_name)

    eval_dict = {'r_squared': r2_score(train_y_df, predicted)}
    print('\nr_squared:', eval_dict['r_squared'])

    eval_dict['rmse'] = mean_squared_error(train_y_df, predicted, squared=False)
    print('rmse:', eval_dict['rmse'])

    eval_dict['frac_rmse'] = eval_dict['rmse'] / train_y_df.values.mean()
    print('frac_rmse:', eval_dict['frac_rmse'])

    return eval_dict


def score_trained_estimator(a_trained_estimator, a_cap_x_df, a_y_df):
    mse = mean_squared_error(a_y_df, a_trained_estimator.predict(a_cap_x_df))
    rmse = np.sqrt(mse)
    relative_rmse = rmse / a_y_df.mean()
    return mse, rmse, relative_rmse


def model_assess_with_bootstrapping(a_best_model, a_num_bs_samples, a_train_cap_x_df, a_train_y_df, a_test_cap_x_df,
                                    a_test_y_df, an_estimator_name, target_attr):
    import warnings
    warnings.filterwarnings("ignore", message='')

    print('\n', '*' * 80, sep='')
    print('model assessment with bootstrapping:', an_estimator_name)

    # out of loop initialization
    rmse_df_row_dict_list = []
    rel_rmse_df_row_dict_list = []
    for bs_sample_index in range(a_num_bs_samples):
        # in loop initialization
        rmse_df_row_dict = {}
        rel_rmse_df_row_dict = {}

        # get a bootstrap data frame
        bs_cap_x_df = a_train_cap_x_df.sample(frac=1, replace=True, random_state=bs_sample_index)
        bs_y_df = a_train_y_df.loc[bs_cap_x_df.index]

        # clone the best model for fitting bootstrapped data frame
        bs_best_model = clone(a_best_model)  # Construct a new unfitted estimator with the same hyper parameters.

        # fit the bootstrap model
        bs_best_model.fit(bs_cap_x_df, bs_y_df[target_attr].array)

        # document bootstrap model performance
        mse, rmse, relative_rmse = score_trained_estimator(bs_best_model, a_test_cap_x_df, a_test_y_df)

        rmse_df_row_dict['estimator'] = an_estimator_name
        rmse_df_row_dict['rmse'] = rmse

        rel_rmse_df_row_dict['estimator'] = an_estimator_name
        rel_rmse_df_row_dict['relative_rmse'] = relative_rmse

        # accumulate bootstrap model performance documentation
        rmse_df_row_dict_list.append(rmse_df_row_dict.copy())
        rel_rmse_df_row_dict_list.append(rel_rmse_df_row_dict.copy())

    # convert accumulated bootstrap model performance into a data frame
    rmse_bs_results_df = pd.DataFrame(rmse_df_row_dict_list)
    rel_rmse_bs_results_df = pd.DataFrame(rel_rmse_df_row_dict_list)

    return rmse_bs_results_df, rel_rmse_bs_results_df
