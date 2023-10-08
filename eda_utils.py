import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import scipy.stats as stats
import numpy as np
import seaborn as sns
import time
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scikit_posthocs import posthoc_dunn
import scipy as sp

import common_utils


# Common methods

def split_nominal_and_numerical(a_df: pd.DataFrame, a_target_attr_list: list) -> (list, list):
    a_df = a_df.drop(columns=a_target_attr_list)
    a_numerical_attr_list = a_df.select_dtypes(include=['number']).columns.tolist()
    a_nominal_attr_list = a_df.select_dtypes(exclude=['number']).columns.tolist()

    print("\ntarget_attr_list: \n", a_target_attr_list)
    print("\nnumerical_attr_list: \n", a_numerical_attr_list)
    print("\nnominal_attr_list: \n", a_nominal_attr_list)

    return a_numerical_attr_list, a_nominal_attr_list


def check_for_duplicate_observations(a_df: pd.DataFrame) -> None:
    print('check_for_duplicate_observations:')

    dedup_a_df = a_df.drop_duplicates()
    print('a_df.shape:', a_df.shape)
    print('dedup_a_df.shape:', dedup_a_df.shape)

    if dedup_a_df.shape[0] < a_df.shape[0]:
        print('caution: data set contains duplicate observations!!!')
    else:
        print('no duplicate observations observed in data set')


def check_out_missingness(a_df: pd.DataFrame, sample_size_threshold: int = 250, verbose: bool = True) -> None:
    print('check_out_missingness:')

    if verbose:
        print('\nNA (np.nan or None) count - a_df[an_attr_list].isna().sum():\n', a_df.isna().sum())
        print('\nNA (np.nan or None) fraction - a_df[an_attr_list].isna().sum() / a_df.shape[0]:\n',
              a_df.isna().sum() / a_df.shape[0])

    if a_df.isna().sum().sum() > 0:
        print('\nmissing values in data set!!!')

        sample_size = a_df.shape[0]
        if a_df.shape[0] > sample_size_threshold:
            sample_size = sample_size_threshold

        print('\nuse missingno to understand pattern of missingness')
        print('a_df.shape[0]:', a_df.shape[0])
        print('missingno sample_size:', sample_size)

        msno.matrix(a_df.sample(sample_size, random_state=42))
        plt.show()
        msno.heatmap(a_df.sample(sample_size, random_state=42))
        plt.show()

    else:
        print('\nno missing values in data set.')


def check_out_target_distribution(a_df: pd.DataFrame, a_target_attr: list) -> None:
    print('check_out_target_distribution:')
    print('\na_df[a_target_attr].describe():\n', a_df[a_target_attr].describe())
    print('\n')
    a_df[a_target_attr].hist()
    plt.grid()
    plt.show()
    statistic, p_value = stats.normaltest(a_df[a_target_attr].dropna())
    print('\ntest data for normality:\n')
    print(f'\nnull hypothesis: data comes from a normal distribution - p_value: {p_value}')


def drop_obs_with_nans(a_df: pd.DataFrame) -> pd.DataFrame:
    if a_df.isna().sum().sum() > 0:
        print(f'\nfound observations with nans - pre obs. drop a_df.shape: {a_df.shape}')
        a_df = a_df.dropna(axis=0, how='any')
        print(f'post obs. drop a_df.shape: {a_df.shape}')

    return a_df


# Numerical attributes

def get_flattened_corr_matrix(corr_df: pd.DataFrame, corr_threshold: float = 0.75) -> pd.DataFrame:
    corr_df = corr_df.where(np.tril(np.ones(corr_df.shape)).astype(np.bool_))
    flat_attr_correlations_df = corr_df.stack().reset_index()
    flat_attr_correlations_df = flat_attr_correlations_df.rename(
        columns={'level_0': 'attribute_x', 'level_1': 'attribute_y', 0: 'correlation'})
    flat_attr_correlations_df = (flat_attr_correlations_df[
                                     (flat_attr_correlations_df.correlation != 1) &
                                     (flat_attr_correlations_df.correlation.abs() > corr_threshold)]
                                 .sort_values('correlation')
                                 .reset_index(drop=True))

    print('\n', f'{flat_attr_correlations_df.shape[0]} pairs of attributes with correlations row > {corr_threshold}')
    print(flat_attr_correlations_df)

    return flat_attr_correlations_df


def get_correlation_data_frame(a_df: pd.DataFrame, a_num_attr_list: list, method: str = 'pearson',
                               corr_threshold: float = 0.75) -> pd.DataFrame:
    print(f'\ncorrelation data frame using {method} method with threshold {corr_threshold}:\n')

    attr_correlations_df = a_df[a_num_attr_list].corr(method=method)
    flat_attr_correlations_df = get_flattened_corr_matrix(attr_correlations_df, corr_threshold=corr_threshold)

    return flat_attr_correlations_df


def print_corr_of_num_attrs(a_df: pd.DataFrame, a_num_attr_list: list, method: str = 'pearson',
                            corr_threshold: float = 0.50) -> None:
    print('heatmap of design matrix attribute correlations:\n')

    if a_df[a_num_attr_list].shape[1] < 20:
        fig, ax = plt.subplots(figsize=(15, 15))
        sns.heatmap(a_df[a_num_attr_list].corr(method=method), annot=True, ax=ax)
        plt.show()
        _ = get_correlation_data_frame(a_df, a_num_attr_list, method=method, corr_threshold=corr_threshold)
    else:
        print(f'\nSkip correlation heat map - {a_df[a_num_attr_list].shape[1]} attributes is too many for a useful '
              f'visual output.')
        _ = get_correlation_data_frame(a_df, a_num_attr_list, method=method, corr_threshold=corr_threshold)


def print_pair_plot(a_df: pd.DataFrame, a_num_attr_list: list) -> None:
    print('investigate multi co-linearity: pair plots of the numerical attributes:\n')
    if a_df[a_num_attr_list].shape[1] < 20:
        sns.pairplot(a_df[a_num_attr_list], height=1)
        plt.tight_layout()
        plt.show()
    else:
        print(f'\nSkip pair plots - {a_df[a_num_attr_list].shape[1]} attributes is too many for a useful visual '
              f'output.')


def prep_data_for_vif_calc(a_df: pd.DataFrame, a_num_attr_list: list) -> (pd.DataFrame, str):
    # drop observations with nans
    a_df = drop_obs_with_nans(a_df[a_num_attr_list])

    # prepare the data - make sure you perform the analysis on the design matrix
    design_matrix = None
    bias_attr = None
    for attr in a_df[a_num_attr_list]:
        if a_df[attr].nunique() == 1 and a_df[attr].iloc[0] == 1:  # found the bias attribute
            design_matrix = a_df[a_num_attr_list]
            bias_attr = attr
            print('found the bias term - no need to add one')
            break

    if design_matrix is None:
        design_matrix = sm.add_constant(a_df[a_num_attr_list])
        bias_attr = 'const'
        a_num_attr_list = [bias_attr] + a_num_attr_list
        print('\nAdded a bias term to the data frame to construct the design matrix for assessment of vifs.')

    # if numerical attributes in the data frame are not scaled then scale them - don't scale the bias term
    a_num_attr_list.remove(bias_attr)
    if not (a_df[a_num_attr_list].mean() <= 1e-14).all():
        print('scale the attributes - but not the bias term')
        design_matrix[a_num_attr_list] = StandardScaler().fit_transform(design_matrix[a_num_attr_list])

    return design_matrix, bias_attr


def print_vifs(a_df: pd.DataFrame, a_num_attr_list: list) -> pd.DataFrame:
    print('investigate multi co-linearity: calculate variance inflation factors:\n')

    design_matrix, bias_attr = prep_data_for_vif_calc(a_df, a_num_attr_list)

    # calculate the vifs
    vif_df = pd.DataFrame()
    vif_df['attribute'] = design_matrix.columns.tolist()
    vif_df['vif'] = [variance_inflation_factor(design_matrix.values, i) for i in range(design_matrix.shape[1])]
    vif_df['vif'] = vif_df['vif'].round(2)

    print('\n', vif_df)
    time.sleep(2)

    return vif_df


def print_hist_of_num_attrs(a_df: pd.DataFrame, a_num_attr_list: list) -> None:
    print('histograms of the numerical attributes:')
    a_df[a_num_attr_list].hist(figsize=(10, 10))
    plt.tight_layout()
    plt.show()


def print_boxplots_of_num_attrs(a_df, a_num_attr_list, tukey_outliers=False, show_outliers=False):
    print('boxplots of the numerical attributes:\n')

    tukey_univariate_poss_outlier_dict = {}
    tukey_univariate_prob_outlier_dict = {}
    for attr in a_num_attr_list:
        print('\n', 20 * '*')
        print(attr)
        a_df.boxplot(column=attr, figsize=(5, 5))
        plt.show()
        if tukey_outliers:
            outliers_prob, outliers_poss = tukeys_method(a_df, attr)
            tukey_univariate_prob_outlier_dict[attr] = outliers_prob
            tukey_univariate_poss_outlier_dict[attr] = outliers_poss
            if show_outliers:
                print('univariate outliers:')
                print('\ntukey\'s method - outliers_prob indices:\n', outliers_prob)
                print('\ntukey\'s method - outliers_poss indices:\n', outliers_poss)

    return tukey_univariate_poss_outlier_dict, tukey_univariate_prob_outlier_dict


def tukeys_method(a_df: pd.DataFrame, variable: str):

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
    for index, x in zip(a_df.index, a_df[variable]):
        if x <= inner_fence_le or x >= inner_fence_ue:
            outliers_poss.append(index)

    return outliers_prob, outliers_poss


def use_tukeys_method(a_df, a_num_attr_list):
    print('use_tukeys_method to identify outliers:\n')

    tukey_univariate_poss_outlier_dict = {}
    tukey_univariate_prob_outlier_dict = {}
    for attr in a_num_attr_list:
        print('\n', attr)
        outliers_prob, outliers_poss = tukeys_method(a_df, attr)
        print('tukey\'s method - outliers_prob indices: ', outliers_prob)
        tukey_univariate_prob_outlier_dict[attr] = outliers_prob
        print('tukey\'s method - outliers_poss indices: ', outliers_poss)
        tukey_univariate_poss_outlier_dict[attr] = outliers_poss

    return tukey_univariate_poss_outlier_dict, tukey_univariate_prob_outlier_dict


def check_out_univariate_outliers_in_cap_x(a_df, a_num_attr_list, show_outliers=False):
    print('check_out_univariate_outliers_in_cap_x:')

    print_hist_of_num_attrs(a_df, a_num_attr_list)

    tukey_outliers = True
    tukey_univariate_poss_outlier_dict, tukey_univariate_prob_outlier_dict = \
        print_boxplots_of_num_attrs(a_df, a_num_attr_list, tukey_outliers=tukey_outliers, show_outliers=show_outliers)

    if not tukey_outliers:
        tukey_univariate_poss_outlier_dict, tukey_univariate_prob_outlier_dict = \
            use_tukeys_method(a_df, a_num_attr_list)

    if show_outliers:
        print('\ntukey_univariate_prob_outlier_dict:')
    attrs_with_tukey_prob_outliers_list = []
    univariate_outlier_list = []
    for attr, outliers_prob in tukey_univariate_prob_outlier_dict.items():
        if show_outliers:
            print('\n   attr:', attr, '; outliers_prob:', outliers_prob)
        if len(outliers_prob) > 0:
            attrs_with_tukey_prob_outliers_list.append(attr)
            univariate_outlier_list.extend(outliers_prob)

    print('\n', 30 * '*')
    print('univariate outlier summary:')
    print(f'\ncount of attributes with probable tukey univariate outliers:\n{len(attrs_with_tukey_prob_outliers_list)}')
    print(f'\nlist of attributes with probable tukey univariate outliers:\n{attrs_with_tukey_prob_outliers_list}')
    print(f'\ncount of unique probable tukey univariate outliers across all attributes:\n'
          f'{len(set(univariate_outlier_list))}')
    if show_outliers:
        print(f'\nlist of observations with probable tukey univariate outliers:\n{set(univariate_outlier_list)}')

    return tukey_univariate_poss_outlier_dict, tukey_univariate_prob_outlier_dict


def mahalanobis_method(a_df):
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
    print('check_out_multivariate_outliers_in_cap_x:')
    outlier, _ = mahalanobis_method(a_df[a_num_attr_list])

    print('\nmultivariate outlier summary:\n')
    print(f'count of multivariate outliers using mahalanobis method: {len(outlier)}')
    if show_outliers:
        print('\nmultivariate outliers using mahalanobis method:', outlier)


# Categorical Attributes

def explore_cardinality_of_categorical_attrs(a_df, a_cat_attr_list):
    print('explore_cardinality_of_categorical_attrs:')
    for attr in a_cat_attr_list:
        print('\n', 20 * '*')
        print(attr)
        print('a_df[attr].nunique():', a_df[attr].nunique())
        print('a_df[attr].value_counts(dropna=False):\n', a_df[attr].value_counts(dropna=False))


def check_out_skew_and_kurtosis(a_df):
    print('\ncheck out skewness and kurtosis:')
    for attr in a_df.columns:
        print('\nattr: ', attr)
        print(f'kurtosis: {a_df[attr].kurtosis()}')
        print(f'skewness: {a_df[attr].skew()}')


def drop_categories_with_lt_n_instances(a_df, attr, a_target_attr, n):

    print(f'\ncheck category counts and drop categories with count < {n}')
    cat_drop_list = []
    for category in a_df[attr].unique():
        value_count = a_df.loc[a_df[attr] == category, a_target_attr].shape[0]
        if value_count < n:
            print(f'   category {category} has value count = {value_count} - drop it')
            cat_drop_list.append(category)

    a_df = a_df[~a_df[attr].isin(cat_drop_list)]

    return a_df


def do_kruskal_wallis(a_df, attr, a_target_attr):

    print(f'\nperform the kruskal-wallis test to understand if there is a difference in {a_target_attr} means between '
          f'the categories:')

    a_df = a_df.loc[:, [attr, a_target_attr]]
    a_df = drop_categories_with_lt_n_instances(a_df, attr, a_target_attr, 5)

    groups = [a_df.loc[a_df[attr] == group, a_target_attr].values for group in a_df[attr].unique()]
    results = stats.kruskal(*groups)

    kruskal_wallis_alpha = 0.05
    dunns_test_alpha = 0.05
    if results.pvalue < kruskal_wallis_alpha:

        print(f'\n   kruskal-wallis p-value: {results.pvalue}')
        print(f'   at least one mean is different then the others at alpha = {kruskal_wallis_alpha} level - conduct '
              f'the dunn\'s test')

        results = posthoc_dunn(a_df, val_col=a_target_attr, group_col=attr, p_adjust='bonferroni')

        sym_matrix_df = common_utils.convert_symmetric_matrix_to_df(results, 'p_value')

        sym_matrix_df = sym_matrix_df[sym_matrix_df.p_value < dunns_test_alpha]

        print(f'\ndunn\'s test results:')
        print(sym_matrix_df)
    else:
        print(f'   differences in means are not significant at alpha = {kruskal_wallis_alpha} level')


def print_catplots(a_df, a_cat_attr_list, a_target_attr, a_kinds_list, num_unique_levels_threshold=18,
                   num_obs_threshold=1000):

    if a_df.shape[0] > num_obs_threshold:
        print('\n', f'too many observations for other kinds of plots - only plot strip plots')
        a_kinds_list = ['strip']

    for attr in a_cat_attr_list:
        print('\n\n', 50 * '*', '\n', 50 * '*')
        print(attr)
        num_unique_levels = a_df[attr].nunique()
        print('\na_df[attr].nunique():', num_unique_levels)
        print('\na_df[attr].value_counts(dropna=False):\n', a_df[attr].value_counts(dropna=False))
        if num_unique_levels > num_unique_levels_threshold:
            print('\n', f'num_unique_levels = {num_unique_levels} which exceeds the num_unique_levels_threshold '
                        f'{num_unique_levels_threshold} - do not plot!')
            do_kruskal_wallis(a_df, attr, a_target_attr)
            continue
        for kind in a_kinds_list:
            print('\nkind of catplot:', kind)
            print(f'\ndrop rows with nan in {attr} attribute')
            a_df = a_df.dropna(subset=attr)
            if kind == 'violin':
                sns.catplot(x=attr, y=a_target_attr, kind=kind, inner='stick', data=a_df)
            else:
                sns.catplot(x=attr, y=a_target_attr, kind=kind, data=a_df)
            plt.xticks(rotation=90)
            plt.grid()
            plt.show()

        # plot box plot
        sns.catplot(x=attr, y=a_target_attr, kind='box', data=a_df)
        plt.xticks(rotation=90)
        plt.grid()
        plt.show()

        # kruskal-wallis
        do_kruskal_wallis(a_df, attr, a_target_attr)
