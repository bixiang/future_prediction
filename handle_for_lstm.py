import pandas as pd
from sklearn import preprocessing
import numpy as np

def handle_for_lstm(window):
    df = pd.read_csv('traindata_full.csv')
    # print(df)
    base_df = df[(df.data_time <= '2018-05-17') & (df.data_time >= '2009-04-17')]
    # print(base_df[['price_changerate_1_label']].describe())
    # exc_cols = ['data_time', 'label_1', 'label_3', 'label_5', 'price', 'price_change_1', 'price_change_3', 'price_change_5','price_changerate_1', 'price_changerate_3', 'price_changerate_5', 'price_new']
    exc_cols = ['data_time', 'label_1', 'label_3', 'label_5', 'price_changerate_1','price_changerate_3','price_changerate_5','price_changerate_1_label','price_changerate_3_label','price_changerate_5_label']
    cols = [c for c in base_df.columns if c not in exc_cols]
    # cols = ['NetCJ_CHG','pos','volume', 'price']
    # print(cols)
    base_df[base_df.columns] = base_df[base_df.columns].fillna(0)
    feature_df = base_df[cols]
    # print(type(feature_df.ix[:,'price_changerate_1'].values))

    # base_df['price_changerate_1'] = base_df['price_changerate_1'].fillna(0)
    # print(feature_df)

    # print(base_df.ix[:,'price_changerate_1'].shape)
    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(feature_df.ix[:, cols].values)
    # print(X_scaled.shape)
    X_scaled = np.concatenate((X_scaled, base_df.ix[:,'price_changerate_5'].values.reshape(X_scaled.shape[0],1)),axis=1)
    # print(X_scaled.shape)
    # print(X_scaled)


    y = base_df.loc[:, 'price_changerate_5_label'].rolling(window).apply(lambda x: x[-1])[window - 1:].values
    # print(type(y))


    base_data = []
    num_data = X_scaled.shape[0]-(window-1)
    # print(num_data)
    i = 0
    while(i < num_data):
        x = X_scaled[i:i+window,:]
        # x = x.reshape(x.shape[0] * x.shape[1])
        # print(x.shape)
        if list(x):
            base_data.append(x)
        # print(x)
        i = i + 1
    # print(len(base_data))

    base_data = np.array(base_data)
    # print(base_data.shape)


    return base_data[0:1900,:,:], y[0:1900,], base_data[1901:,:,:], y[1901:,]

handle_for_lstm(20)
