import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import scipy.stats as stats


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
