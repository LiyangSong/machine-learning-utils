import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn.base import clone
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import time
import math
import collections


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


def flexibility_plot(a_gs_cv_results, an_estimator_name):
    # convert mse to rmse
    a_gs_cv_results.mean_train_score = np.sqrt(-1 * a_gs_cv_results.mean_train_score)
    a_gs_cv_results.mean_test_score = np.sqrt(-1 * a_gs_cv_results.mean_test_score)
    # sort by train score and label with index for plotting
    a_gs_cv_results = a_gs_cv_results.sort_values('mean_train_score', ascending=False).reset_index(drop=True). \
        reset_index()
    a_gs_cv_results = a_gs_cv_results[['index', 'rank_test_score', 'mean_train_score', 'mean_test_score']]

    # plot train and test rmse
    sns.scatterplot(x='index', y='mean_train_score', data=a_gs_cv_results, label='mean_train_score')
    sns.scatterplot(x='index', y='mean_test_score', data=a_gs_cv_results, label='mean_test_score')
    best_index = a_gs_cv_results.loc[a_gs_cv_results['rank_test_score'] == 1, 'index'].values[0]
    plt.axvline(x=best_index)
    plt.title(f'{an_estimator_name} flexibility plot')
    plt.xlabel('flexibility')
    plt.ylabel('rmse')
    plt.legend()

    # make index an integer on plot
    new_list = range(math.floor(min(a_gs_cv_results.index)), math.ceil(max(a_gs_cv_results.index)) + 1)

    skip = 1
    if 10 <= len(new_list) < 100:
        skip = 10
    elif 100 <= len(new_list) < 1000:
        skip = 100
    else:
        skip = 500
    plt.xticks(np.arange(min(new_list), max(new_list) + 1, skip))

    plt.grid()
    plt.show()
    return a_gs_cv_results


def grid_search_bs(a_train_cap_x_df, a_train_y_df, target_attr, estimators, experiment_dict, preprocessor):
    print('\nImplement grid search over hyper parameters to select best model:\n')

    i = -1
    a_df_row_dict_list = []
    for estimator, param_grid in experiment_dict.items():
        print('\n', '*' * 80)
        i += 1
        print(estimators[i])

        # build the composite estimator
        composite_estimator = Pipeline(steps=[('preprocessor', preprocessor), ('estimator', estimator)])

        # instantiate the grid search cv
        grid_search_cross_val = GridSearchCV(
            estimator=composite_estimator,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            refit=True,
            cv=5,
            verbose=1,
            pre_dispatch='2*n_jobs',
            error_score=np.nan,
            return_train_score=True
        )

        # fit the grid search cv
        grid_search_cross_val.fit(a_train_cap_x_df, a_train_y_df[target_attr].values.ravel())
        time.sleep(5)

        # plot the flexilibilty plot
        gs_cv_results = pd.DataFrame(grid_search_cross_val.cv_results_).sort_values('rank_test_score')

        gs_cv_results = flexibility_plot(gs_cv_results, estimators[i])
        print('\n', gs_cv_results[gs_cv_results['rank_test_score'] == 1], sep='')

        print('\nbest_model_hyperparameters:\n', grid_search_cross_val.best_params_)

        # collect results
        a_df_row_dict = collections.OrderedDict()  # used to store results
        best_estimator = grid_search_cross_val.best_estimator_
        a_df_row_dict['iteration'] = i
        a_df_row_dict['estimator'] = estimators[i]
        a_df_row_dict['r_squared'] = r2_score(a_train_y_df, best_estimator.predict(a_train_cap_x_df))
        a_df_row_dict['train_rmse'] = np.sqrt(mean_squared_error(a_train_y_df, best_estimator.predict(a_train_cap_x_df)))
        a_df_row_dict['train_relative_rmse'] = a_df_row_dict['train_rmse'] / a_train_y_df[target_attr].mean()
        a_df_row_dict['best_estimator'] = grid_search_cross_val.best_estimator_
        a_df_row_dict['best_estimator_hyperparameters'] = grid_search_cross_val.best_params_

        a_df_row_dict_list.append(a_df_row_dict.copy())

    return pd.DataFrame(a_df_row_dict_list)