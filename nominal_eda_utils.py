import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from scikit_posthocs import posthoc_dunn

import general_utils


def explore_cardinality_of_cat_attrs(a_df: pd.DataFrame, a_cat_attr_list: list) -> None:
    print('\nExplore cardinality of categorical attributions:\n')

    for attr in a_cat_attr_list:
        print('\n', 20 * '*')
        print(attr)
        print('a_df[attr].nunique():', a_df[attr].nunique())
        print('a_df[attr].value_counts(dropna=False):\n', a_df[attr].value_counts(dropna=False))


def drop_cat_with_lt_n_instances(a_df: pd.DataFrame, attr: str, n) -> pd.DataFrame:
    print(f'\nCheck {attr} category counts and drop categories with count < {n}\n')

    cat_drop_list = []
    for category in a_df[attr].unique():
        value_count = a_df.loc[a_df[attr] == category].shape[0]
        if value_count < n:
            print(f'   category {category} has value count = {value_count} - drop it')
            cat_drop_list.append(category)

    a_df = a_df[~a_df[attr].isin(cat_drop_list)]

    return a_df


def do_kruskal_wallis(a_df: pd.DataFrame, a_cat_attr_list: list, a_target_attr: str) -> None:
    print(f'\nPerform the kruskal-wallis test to understand if there is a difference in {a_target_attr} means between the categories:\n')

    for attr in a_cat_attr_list:
        a_df_attr = a_df.loc[:, [attr, a_target_attr]]
        a_df_attr = drop_cat_with_lt_n_instances(a_df_attr, attr, 5)

        groups = [a_df_attr.loc[a_df_attr[attr] == group, a_target_attr].values for group in a_df[attr].unique()]
        results = stats.kruskal(*groups)

        kruskal_wallis_alpha = 0.05
        dunns_test_alpha = 0.05
        if results.pvalue < kruskal_wallis_alpha:

            print(f'\n   kruskal-wallis p-value: {results.pvalue}')
            print(f'   at least one mean is different then the others at alpha = {kruskal_wallis_alpha} level - conduct '
                  f'the dunn\'s test')

            results = posthoc_dunn(a_df_attr, val_col=a_target_attr, group_col=attr, p_adjust='bonferroni')

            sym_matrix_df = general_utils.convert_symmetric_matrix_to_df(results, 'p_value')

            sym_matrix_df = sym_matrix_df[sym_matrix_df.p_value < dunns_test_alpha]

            print(f'\ndunn\'s test results:')
            print(sym_matrix_df)
        else:
            print(f'   differences in means are not significant at alpha = {kruskal_wallis_alpha} level')


def print_cat_plots(a_df, a_cat_attr_list, a_target_attr, a_kinds_list, num_unique_levels_threshold=18,
                    num_obs_threshold=1000):
    print('\n Plots for categories: \n')

    if a_df.shape[0] > num_obs_threshold:
        print(f'\ntoo many observations for other kinds of plots - only plot strip plots')
        a_kinds_list = ['strip']

    for attr in a_cat_attr_list:
        print('\n', 20 * '*')
        print(attr)
        num_unique_levels = a_df[attr].nunique()
        print('\na_df[attr].nunique():', num_unique_levels)
        print('\na_df[attr].value_counts(dropna=False):\n', a_df[attr].value_counts(dropna=False))

        if num_unique_levels > num_unique_levels_threshold:
            print('\n', f'num_unique_levels = {num_unique_levels} which exceeds the num_unique_levels_threshold '
                        f'{num_unique_levels_threshold} - do not plot!')
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
