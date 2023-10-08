import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import scipy.stats as stats


def check_for_duplicate_obs(a_df: pd.DataFrame) -> None:
    print('\nCheck for duplicate observations:')

    dedup_a_df = a_df.drop_duplicates()
    print('a_df.shape:', a_df.shape)
    print('dedup_a_df.shape:', dedup_a_df.shape)

    if dedup_a_df.shape[0] < a_df.shape[0]:
        print('caution: data set contains duplicate observations!!!')
    else:
        print('no duplicate observations observed in data set.')


def check_out_missingness(a_df: pd.DataFrame, sample_size_threshold: int = 250, verbose: bool = True) -> None:
    print('\nCheck out missingness:')

    if verbose:
        print('\nNA (np.nan or None) count - a_df[an_attr_list].isna().sum():\n', a_df.isna().sum())
        print('\nNA (np.nan or None) fraction - a_df[an_attr_list].isna().sum() / a_df.shape[0]:\n',
              a_df.isna().sum() / a_df.shape[0])

    if a_df.isna().sum().sum() > 0:
        print('\nmissing values in data set!!!')

        sample_size = a_df.shape[0]
        if a_df.shape[0] > sample_size_threshold:
            sample_size = sample_size_threshold

        print('\nuse missingno to understand pattern of missingness:')
        print('a_df.shape[0]:', a_df.shape[0])
        print('missingno sample_size:', sample_size)

        msno.matrix(a_df.sample(sample_size, random_state=42))
        plt.show()
        msno.heatmap(a_df.sample(sample_size, random_state=42))
        plt.show()

    else:
        print('\nno missing values in data set.')


def drop_obs_with_nans(a_df: pd.DataFrame) -> pd.DataFrame:
    print('drop_obs_with_nans:')
    if a_df.isna().sum().sum() > 0:
        print(f'\nfound observations with nans - pre obs. drop a_df.shape: {a_df.shape}')
        a_df = a_df.dropna(axis=0, how='any')
        print(f'post obs. drop a_df.shape: {a_df.shape}')
    return a_df


def check_out_skew_and_kurtosis(a_df: pd.DataFrame) -> None:
    print('\nCheck out skewness and kurtosis:')
    for attr in a_df.columns:
        print('\nattr: ', attr)
        print(f'kurtosis: {a_df[attr].kurtosis()}')
        print(f'skewness: {a_df[attr].skew()}')


def check_out_target_distribution(a_df: pd.DataFrame, a_target_attr: list) -> None:
    print('\nCheck out target distribution:')
    print('\na_df[a_target_attr].describe():\n', a_df[a_target_attr].describe(), '\n')
    a_df[a_target_attr].hist()
    plt.grid()
    plt.show()
    statistic, p_value = stats.normaltest(a_df[a_target_attr].dropna())
    print('\ntest data for normality:\n')
    print(f'\nnull hypothesis: data comes from a normal distribution - p_value: {p_value}')


def check_out_target_imbalance(a_df, a_target_attr):
    print('\nCheck out target imbalance:')

    print(f'\nnumber of classes in target attribute: {a_df[a_target_attr].nunique()}')
    print(f'\nclasses in target attribute: {a_df[a_target_attr].unique()}')
    print(f'\nclass balance:\n{a_df[a_target_attr].value_counts(normalize=True)}')


def split_nominal_and_numerical_attr(a_df: pd.DataFrame, a_target_attr_list: list) -> (list, list):
    print('\nSplit nominal and numerical attr:')

    a_df = a_df.drop(columns=a_target_attr_list)
    a_numerical_attr_list = a_df.select_dtypes(include=['number']).columns.tolist()
    a_nominal_attr_list = a_df.select_dtypes(exclude=['number']).columns.tolist()

    print("\ntarget_attr_list: \n", a_target_attr_list)
    print("\nnumerical_attr_list: \n", a_numerical_attr_list)
    print("\nnominal_attr_list: \n", a_nominal_attr_list)

    return a_numerical_attr_list, a_nominal_attr_list
