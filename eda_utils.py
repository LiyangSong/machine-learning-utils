import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import scipy.stats as stats
import numpy as np
import seaborn as sns
import time
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


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


def check_out_missingness(a_df: pd.DataFrame, sample_size_threshold: int = 250, verbose: bool = True,
                          nullity_corr_method: str = 'spearman', nullity_corr_threshold: float = 0.75) -> None:
    print('check_out_missingness:')

    if verbose:
        print('\nNA (np.nan or None) count - a_df[an_attr_list].isna().sum():\n', a_df.isna().sum(), sep='')
        print('\nNA (np.nan or None) fraction - a_df[an_attr_list].isna().sum() / a_df.shape[0]:\n',
              a_df.isna().sum() / a_df.shape[0], sep='')

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

        print('\n',
              f'nullity correlation using the {nullity_corr_method} method with corr row threshold {nullity_corr_threshold}')
        _ = get_flattened_corr_matrix(a_df.isnull().corr(method=nullity_corr_method),
                                      corr_threshold=nullity_corr_threshold)

    else:
        print('\nno missing values in data set.')


def check_out_target_distribution(a_df: pd.DataFrame, a_target_attr: list) -> None:
    print('check_out_target_distribution:')
    print('\na_df[a_target_attr].describe():\n', a_df[a_target_attr].describe(), sep='')
    print('\n')
    a_df[a_target_attr].hist()
    plt.grid()
    plt.show()
    statistic, p_value = stats.normaltest(a_df[a_target_attr].dropna())
    print('\ntest data for normality:\n')
    print(f'\nnull hypothesis: data comes from a normal distribution - p_value: {p_value}')


# check correlations

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
              f'visual output.', sep='')
        _ = get_correlation_data_frame(a_df, a_num_attr_list, method=method, corr_threshold=corr_threshold)


def print_pair_plot(a_df: pd.DataFrame, a_num_attr_list: list) -> None:
    print('investigate multi co-linearity: pair plots of the numerical attributes:\n')
    if a_df[a_num_attr_list].shape[1] < 20:
        sns.pairplot(a_df[a_num_attr_list], height=1)
        plt.tight_layout()
        plt.show()
    else:
        print(f'\nSkip pair plots - {a_df[a_num_attr_list].shape[1]} attributes is too many for a useful visual '
              f'output.', sep='')


def drop_obs_with_nans(a_df: pd.DataFrame) -> pd.DataFrame:
    if a_df.isna().sum().sum() > 0:
        print(f'\nfound observations with nans - pre obs. drop a_df.shape: {a_df.shape}')
        a_df = a_df.dropna(axis=0, how='any')
        print(f'post obs. drop a_df.shape: {a_df.shape}')

    return a_df


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
        print('\nAdded a bias term to the data frame to construct the design matrix for assessment of vifs.', sep='')

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

    print('\n', vif_df, sep='')
    time.sleep(2)

    return vif_df
