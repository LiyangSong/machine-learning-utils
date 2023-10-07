import pandas as pd
from sklearn.model_selection import train_test_split


def load_csv_into_df(a_path: str) -> pd.DataFrame:
    a_df = pd.read_csv(a_path)
    print("data loaded: ", a_df.shape)
    return a_df


def split_train_test_df(a_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> (
pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    cap_x_df, y_df = a_df.iloc[:, :-1], a_df.iloc[:, -1].to_frame()
    a_train_cap_x_df, a_test_cap_x_df, a_train_y_df, a_test_y_df = train_test_split(
        cap_x_df, y_df,
        test_size=test_size,
        train_size=None,
        random_state=random_state,
        shuffle=True,
        stratify=None)

    print("train set:")
    print(a_train_cap_x_df.shape, a_train_y_df.shape)
    print("test set:")
    print(a_test_cap_x_df.shape, a_test_y_df.shape)

    return a_train_cap_x_df, a_test_cap_x_df, a_train_y_df, a_test_y_df


def save_df_to_csv(a_cap_x_df: pd.DataFrame, a_y_df: pd.DataFrame, a_csv_filename: str) -> None:
    pd.concat([a_cap_x_df, a_y_df], axis=1).to_csv(a_csv_filename, index=True, index_label='index')
    print("file saved:", a_csv_filename)