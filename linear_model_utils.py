import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
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
    print('Plot prediction versus actual target values:\n')

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
    print('=' * 60)
    print(f'Check {model_selection_stage} {estimator_name} estimator prediction performance on the {data_set_type} data set: ')
    print('=' * 60)

    plot_pred_vs_actual(predicted, train_y_df, estimator_name)

    print('\nr_squared:', r2_score(train_y_df, predicted))
    print('rmse:', mean_squared_error(train_y_df, predicted, squared=False))
    print('frac_rmse:', mean_squared_error(train_y_df, predicted, squared=False) / train_y_df.values.mean())


def score_trained_estimator(a_trained_estimator, a_cap_x_df, a_y_df):
    mse = mean_squared_error(a_y_df, a_trained_estimator.predict(a_cap_x_df))
    rmse = np.sqrt(mse)
    relative_rmse = rmse / a_y_df.mean()
    return mse, rmse, relative_rmse


def model_assess_with_bootstrapping(results_df, bs_train_cap_x_df, bs_train_y_df, bs_test_cap_x_df,
                                    bs_test_y_df, target_attr, estimators, num_bs_samples=10):
    import warnings
    warnings.filterwarnings("ignore", message='')

    print('=' * 60)
    print('Model assessment with bootstrapping:')
    print('=' * 60)

    for i, estimator in enumerate(estimators):
        best_estimator = results_df[results_df.estimator == estimator].best_estimator[i]
        print('\n', '*' * 40)
        estimator_name = 'tuned_elastic_net'
        print('model assessment with bootstrapping on estimator:', estimator)

        # out of loop initialization
        rmse_df_row_dict_list = []
        rel_rmse_df_row_dict_list = []
        for bs_sample_index in range(num_bs_samples):

            # in loop initialization
            rmse_df_row_dict = {}
            rel_rmse_df_row_dict = {}

            # get a bootstrap data frame
            bs_cap_x_df = bs_train_cap_x_df.sample(frac=1, replace=True, random_state=bs_sample_index)
            bs_y_df = bs_train_y_df.loc[bs_cap_x_df.index]

            # clone the best model for fitting bootstrapped data frame
            bs_best_model = clone(best_estimator)  # Construct a new unfitted estimator with the same hyper parameters.

            # fit the bootstrap model
            bs_best_model.fit(bs_cap_x_df, bs_y_df[target_attr].values.ravel())

            # document bootstrap model performance
            mse, rmse, relative_rmse = score_trained_estimator(bs_best_model, bs_test_cap_x_df, bs_test_y_df)

            rmse_df_row_dict['estimator'] = estimator
            rmse_df_row_dict['rmse'] = rmse

            rel_rmse_df_row_dict['estimator'] = estimator
            rel_rmse_df_row_dict['relative_rmse'] = relative_rmse

            # accumulate bootstrap model performance documentation
            rmse_df_row_dict_list.append(rmse_df_row_dict)
            rel_rmse_df_row_dict_list.append(rel_rmse_df_row_dict)

    # convert accumulated bootstrap model performance into a data frame
    rmse_bs_results_df = pd.DataFrame(rmse_df_row_dict_list)
    rel_rmse_bs_results_df = pd.DataFrame(rel_rmse_df_row_dict_list)

    plot_bootstrapping('rmse', rmse_bs_results_df)
    plot_bootstrapping('relative_rmse', rel_rmse_bs_results_df)

    return rmse_bs_results_df, rel_rmse_bs_results_df


def plot_bootstrapping(y_label, results_df):
    sns.catplot(kind='box', x='estimator', y=y_label, data=results_df)
    plt.title('assess model performance on the train set using bootstrapping')
    plt.xticks(rotation=90)
    plt.grid()
    plt.show()


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
    print('=' * 60)
    print('Implement grid search over hyper parameters to select best model:')
    print('=' * 60)

    i = -1
    a_df_row_dict_list = []
    for estimator, param_grid in experiment_dict.items():
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
        print('\n', gs_cv_results[gs_cv_results['rank_test_score'] == 1])

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


def check_out_permutation_importance(results_df, train_cap_x_df, train_y_df, estimators):
    print('=' * 60)
    print('Check out permutation importance of features in the best estimator:')
    print('=' * 60)

    perm_imp_dict = {}

    for i, estimator in enumerate(estimators):

        perm_imp_dict[estimator] = []

        best_estimator = results_df[results_df.estimator == estimator].best_estimator[i]
        print('\n', 40 * '*')
        print('\nestimator: ', estimator)
        r_multi = permutation_importance(best_estimator, train_cap_x_df, train_y_df, n_repeats=10, random_state=0,
                                         scoring=['neg_mean_squared_error'])
        for metric in r_multi:
            temp_metric = metric
            if metric == 'neg_mean_squared_error':
                temp_metric = 'sqrt_' + metric
            print(f"\nmetric: {temp_metric}")
            r = r_multi[metric]
            for i in r.importances_mean.argsort()[::-1]:
                if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
                    feature_name = train_cap_x_df.columns[i]
                    mean = r.importances_mean[i]
                    std_dev = r.importances_std[i]
                    if metric == 'neg_mean_squared_error':
                        mean = np.sqrt(mean)
                        std_dev = np.sqrt(std_dev)
                        perm_imp_dict[estimator].append(feature_name)
                    print(
                        f"    {feature_name:<8}"
                        f" {mean:.3f}"
                        f" +/- {std_dev:.3f}"
                    )
    print('\nPermutation importance:')
    print(perm_imp_dict)