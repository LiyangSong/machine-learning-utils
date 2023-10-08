import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy as sp

from general_eda_utils import drop_obs_with_nans


def get_flattened_corr_matrix(corr_df: pd.DataFrame, corr_threshold: float = 0.75) -> pd.DataFrame:
    print(f'Get flattened correlation matrix from DataFrame with threshold {corr_threshold}:')

    corr_df = corr_df.where(np.tril(np.ones(corr_df.shape)).astype(np.bool_))
    flat_attr_correlations_df = corr_df.stack().reset_index()
    flat_attr_correlations_df = flat_attr_correlations_df.rename(
        columns={'level_0': 'attribute_x', 'level_1': 'attribute_y', 0: 'correlation'})
    flat_attr_correlations_df = (flat_attr_correlations_df[
                                     (flat_attr_correlations_df.correlation != 1) &
                                     (flat_attr_correlations_df.correlation.abs() > corr_threshold)]
                                 .sort_values('correlation')
                                 .reset_index(drop=True))

    print(f'\n{flat_attr_correlations_df.shape[0]} pairs of attributes with correlations row > {corr_threshold}')
    print(flat_attr_correlations_df)

    return flat_attr_correlations_df


def get_corr_data_frame(a_df: pd.DataFrame, a_num_attr_list: list, method: str = 'pearson',
                        corr_threshold: float = 0.75) -> pd.DataFrame:
    print(f'\nGet correlation DataFrame using {method} method with threshold {corr_threshold}:')

    attr_correlations_df = a_df[a_num_attr_list].corr(method=method)
    flat_attr_correlations_df = get_flattened_corr_matrix(attr_correlations_df, corr_threshold=corr_threshold)

    return flat_attr_correlations_df


def print_corr_of_num_attrs(a_df: pd.DataFrame, a_num_attr_list: list, method: str = 'pearson',
                            corr_threshold: float = 0.50) -> None:

    print('=' * 60)
    print('Check out correlation of numerical attributes:')
    print('=' * 60)

    print('\nHeatmap of design matrix attribute correlations:\n')

    if a_df[a_num_attr_list].shape[1] < 20:
        fig, ax = plt.subplots(figsize=(15, 15))
        sns.heatmap(a_df[a_num_attr_list].corr(method=method), annot=True, ax=ax)
        plt.show()
        _ = get_corr_data_frame(a_df, a_num_attr_list, method=method, corr_threshold=corr_threshold)
    else:
        print(f'\nSkip correlation heat map - {a_df[a_num_attr_list].shape[1]} attributes is too many for a useful '
              f'visual output.')
        _ = get_corr_data_frame(a_df, a_num_attr_list, method=method, corr_threshold=corr_threshold)


def print_pair_plot(a_df: pd.DataFrame, a_num_attr_list: list) -> None:
    print('\nInvestigate multi co-linearity: pair plots of the numerical attributes:\n')

    if a_df[a_num_attr_list].shape[1] < 20:
        sns.pairplot(a_df[a_num_attr_list], height=1)
        plt.tight_layout()
        plt.show()
    else:
        print(f'\nSkip pair plots - {a_df[a_num_attr_list].shape[1]} attributes is too many for a useful visual '
              f'output.')


def prep_data_for_vif_calc(a_df: pd.DataFrame, a_num_attr_list: list) -> (pd.DataFrame, str):
    print('\nPrepare DataFrame for vif calculation:')

    # drop observations with nans
    a_df = drop_obs_with_nans(a_df[a_num_attr_list])

    # prepare the data - make sure you perform the analysis on the design matrix
    design_matrix = None
    bias_attr = None
    for attr in a_df[a_num_attr_list]:
        if a_df[attr].nunique() == 1 and a_df[attr].iloc[0] == 1:  # found the bias attribute
            design_matrix = a_df[a_num_attr_list]
            bias_attr = attr
            print('Found the bias term - no need to add one')
            break

    if design_matrix is None:
        design_matrix = sm.add_constant(a_df[a_num_attr_list])
        bias_attr = 'const'
        a_num_attr_list = [bias_attr] + a_num_attr_list
        print('\nAdded a bias term to the data frame to construct the design matrix for assessment of vifs.')

    # if numerical attributes in the data frame are not scaled then scale them - don't scale the bias term
    a_num_attr_list.remove(bias_attr)
    if not (a_df[a_num_attr_list].mean() <= 1e-10).all():
        print('Scale the attributes - but not the bias term')
        design_matrix[a_num_attr_list] = StandardScaler().fit_transform(design_matrix[a_num_attr_list])

    return design_matrix, bias_attr


def print_vifs(a_df: pd.DataFrame, a_num_attr_list: list) -> None:
    print('=' * 60)
    print('Investigate multi co-linearity: calculate variance inflation factors (VIF):')
    print('=' * 60)

    design_matrix, bias_attr = prep_data_for_vif_calc(a_df, a_num_attr_list)

    # calculate the vifs
    vif_df = pd.DataFrame()
    vif_df['attribute'] = design_matrix.columns.tolist()
    vif_df['vif'] = [variance_inflation_factor(design_matrix.values, i) for i in range(design_matrix.shape[1])]
    vif_df['vif'] = vif_df['vif'].round(2)

    print('\n', vif_df)


def print_hist_of_num_attrs(a_df: pd.DataFrame, a_num_attr_list: list) -> None:
    print('=' * 60)
    print('Histograms of the numerical attributes:')
    print('=' * 60)
    a_df[a_num_attr_list].hist(figsize=(10, 10))
    plt.tight_layout()
    plt.show()


def print_boxplots_of_num_attrs(a_df, a_num_attr_list, n_cols=4):
    print('=' * 60)
    print('Boxplots of the numerical attributes:')
    print('=' * 60)

    n_attrs = len(a_num_attr_list)
    n_rows = (n_attrs // n_cols) + (n_attrs % n_cols > 0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))

    for i, attr in enumerate(a_num_attr_list):
        row = i // n_cols
        col = i % n_cols
        a_df.boxplot(column=attr, ax=axes[row, col])
        axes[row, col].set_title(attr)

    plt.tight_layout()
    plt.show()


def tukeys_method(a_df: pd.DataFrame, variable: str) -> (list, list):

    print(f'\nImplement Tukey\'s fences to identify outliers in attribute {variable} based on the Inter Quartile Range (IQR) method:')

    q1 = a_df[variable].quantile(0.25)
    q3 = a_df[variable].quantile(0.75)
    iqr = q3 - q1
    inner_fence = 1.5 * iqr
    outer_fence = 3 * iqr

    # inner fence lower and upper end
    inner_fence_le = q1 - inner_fence
    inner_fence_ue = q3 + inner_fence

    # outer fence lower and upper end
    outer_fence_le = q1 - outer_fence
    outer_fence_ue = q3 + outer_fence

    outliers_prob = []
    outliers_poss = []
    for index, x in zip(a_df.index, a_df[variable]):
        if x <= outer_fence_le or x >= outer_fence_ue:
            outliers_prob.append(index)
        if x <= inner_fence_le or x >= inner_fence_ue:
            outliers_poss.append(index)

    return outliers_prob, outliers_poss


def use_tukeys_method(a_df: pd.DataFrame, a_num_attr_list: list) -> (dict, dict):
    print('\nUse Tukey\'s method to identify outliers:')

    tukey_univariate_poss_outlier_dict = {}
    tukey_univariate_prob_outlier_dict = {}
    for attr in a_num_attr_list:
        print('\n', '*' * 40, '\n')
        print('\n', attr)
        outliers_prob, outliers_poss = tukeys_method(a_df, attr)
        print('tukey\'s method - outliers_prob indices: ', outliers_prob)
        tukey_univariate_prob_outlier_dict[attr] = outliers_prob
        print('tukey\'s method - outliers_poss indices: ', outliers_poss)
        tukey_univariate_poss_outlier_dict[attr] = outliers_poss

    return tukey_univariate_poss_outlier_dict, tukey_univariate_prob_outlier_dict


def check_out_univariate_outliers_in_cap_x(a_df, a_num_attr_list, show_outliers=False):
    print('=' * 60)
    print('Check out univariate outliers in cap_x:')
    print('=' * 60)

    tukey_univariate_poss_outlier_dict, tukey_univariate_prob_outlier_dict = use_tukeys_method(a_df, a_num_attr_list)

    if show_outliers:
        print('\ntukey_univariate_prob_outlier_dict:')
    attrs_with_tukey_prob_outliers_list = []
    univariate_outlier_list = []
    for attr, outliers_prob in tukey_univariate_prob_outlier_dict.items():
        if show_outliers:
            print(f'\nattr: {attr}; outliers_prob: {outliers_prob}')
        if len(outliers_prob) > 0:
            attrs_with_tukey_prob_outliers_list.append(attr)
            univariate_outlier_list.extend(outliers_prob)

    print('\n', 40 * '*', '\n')
    print('Univariate outlier summary:')
    print(f'\ncount of attributes with probable tukey univariate outliers:\n{len(attrs_with_tukey_prob_outliers_list)}')
    print(f'\nlist of attributes with probable tukey univariate outliers:\n{attrs_with_tukey_prob_outliers_list}')
    print(f'\ncount of unique probable tukey univariate outliers across all attributes:\n'
          f'{len(set(univariate_outlier_list))}')
    if show_outliers:
        print(f'\nlist of observations with probable tukey univariate outliers:\n{set(univariate_outlier_list)}')


def mahalanobis_method(a_df: pd.DataFrame) -> (list, np.sqrt):
    print('\nCompute Mahalanobis distance for observations to detect outliers:')

    # drop observations with nans
    a_df = drop_obs_with_nans(a_df)

    # calculate the mahalanobis distance
    x_minus_mu = a_df - np.mean(a_df)
    cov = np.cov(a_df.values.T)  # Covariance
    inv_covmat = sp.linalg.inv(cov)  # Inverse covariance
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    md = np.sqrt(mahal.diagonal())

    # calculate threshold
    threshold = np.sqrt(chi2.ppf((1 - 0.001), df=a_df.shape[1]))  # degrees of freedom = number of variables

    # collect outliers
    outlier = []
    for index, value in enumerate(md):
        if value > threshold:
            outlier.append(index)
        else:
            continue

    return outlier, md


def check_out_multivariate_outliers_in_cap_x(a_df, a_num_attr_list, show_outliers=False):
    print('=' * 60)
    print('Check out multivariate outliers in cap_x:')
    print('=' * 60)

    outlier, _ = mahalanobis_method(a_df[a_num_attr_list])

    print('\nMultivariate outlier summary:')
    print(f'Count of multivariate outliers using mahalanobis method: {len(outlier)}')
    if show_outliers:
        print('Multivariate outliers using mahalanobis method:', outlier)
