import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

num_cols = ['VehicleAge', 'VehOdo', 'VNZIP1', 'IsOnlineSale', 'WarrantyCost', 'MMRAcquisitionAuctionAveragePrice',
            'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice',
            'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice',
            'MMRCurrentRetailCleanPrice']
cat_cols = ['Auction', 'Make', 'Model', 'Trim', 'SubModel', 'Color', 'Transmission', 'Nationality', 'Size',
            'TopThreeAmericanName', 'VNST', 'BYRNO', 'PRIMEUNIT', 'WheelTypeID', 'AUCGUART']


def summarize_df(df, cl=''):
    print "\n## DS Shape: "
    print df.shape
    print "\n## DS Data Types: "
    print df.dtypes
    print "\n## DS Head: "
    print df.head(5)
    print "\n## DS Summary: "
    print df.describe()
    if len(cl) > 0 and cl in df:
        print "\n## DS Class - {} - Distributions: ".format(cl)
        print df.groupby(cl).size()
        print "\n\n"


def down_sample_df(df, cl=''):
    # alternatives are defining sample_weights or up-sampling
    if len(cl) > 0 and cl in df:
        df0 = df.loc[df[cl] == 0]
        df1 = df.loc[df[cl] == 1]
        df1_size = len(df1)
        df0 = df0.sample(df1_size)
        df = pd.concat([df0, df1])
    return df


def fill_missing_values(df):
    for cat_col in cat_cols:
        mode = df[cat_col].mode()[0]
        df[cat_col] = df[cat_col].fillna(mode)
        items = list(df[cat_col].unique())
        df[cat_col] = df[cat_col].map(lambda x: items.index(x))
    for num_col in num_cols:
        df[num_col] = df[num_col].fillna(df[num_col].median())
    return df


def preprocess_df(df, class_column_name, ds):
    df = df.drop('WheelType', 1)  # redundant
    # date feature engineering, adding month and day of the week instead
    df['PurchDate'] = pd.to_datetime(df['PurchDate'])
    df['month'] = [x.month for x in df['PurchDate']]
    df['dayofweek'] = [x.dayofweek for x in df['PurchDate']]
    df = df.drop(['PurchDate'], axis=1)
    if ds:
        df = down_sample_df(df, class_column_name)
    df = fill_missing_values(df)
    # ToDo: regularization...
    return df


def split_df_train_test(df, class_column_name, vs, seed):
    if class_column_name in df:
        temp = df.drop(class_column_name, axis=1)
    tr_X, te_X, tr_Y, te_Y = train_test_split(temp, df.IsBadBuy, test_size=vs, random_state=seed)
    tr_X = pd.DataFrame(tr_X, columns=df.columns[1:])
    te_X = pd.DataFrame(te_X, columns=df.columns[1:])
    tr_Y = pd.DataFrame(tr_Y, columns=[class_column_name])
    te_Y = pd.DataFrame(te_Y, columns=[class_column_name])
    return tr_X, te_X, tr_Y, te_Y
