'''
dataset：criteo dataset sample
features：
- Label - Target variable that indicates if an ad was clicked (1) or not (0).
- I1-I13 - A total of 13 columns of integer features (mostly count features).
- C1-C26 - A total of 26 columns of categorical features.
The values of these features have been hashed onto 32 bits for anonymization purposes.
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

def create_criteo_dataset(file, read_part=True, sample_num=10000,  embed_dim=8, test_size=0.2):
      # file = './criteo_sampled_data.csv'
    data_df = pd.read_csv(file, iterator=True)
    data_df = data_df.get_chunk(sample_num)

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    # Missing Data: if sparse feature is null, fill -1
    data_df[sparse_features] = data_df[sparse_features].fillna(-1)
    data_df[dense_features] = data_df[dense_features].fillna(0)

    # dense_features = [feat for feat in data_df.columns if feat not in sparse_features]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    print(dense_features)
    data_df[dense_features] = MinMaxScaler(feature_range=(0,1)).fit_transform(data_df[dense_features])

    # ----------------- Label Encoder----------
    for feat in sparse_features:   
        # print(feat)
        data_df[feat] = LabelEncoder().fit_transform(data_df[feat].astype('str'))
    
    feature_columns = [[DenseFeature(feat) for feat in dense_features]] + \
                            [[SparseFeature(feat, len(data_df[feat].unique()), embed_dim=embed_dim) for feat in sparse_features]]
    
    train, test = train_test_split(data_df, test_size=test_size)
    train, val = train_test_split(train, test_size=0.1)
    print(train[sparse_features].values)
    train_X = [train[dense_features].values, train[sparse_features].values]
    train_y = train['label'].values
    val_X = [val[dense_features].values, val[sparse_features].values]
    val_y = val['label'].values
    test_X = [test[dense_features].values, test[sparse_features].values]
    test_y = test['label'].values
        
    return feature_columns, (train_X, train_y), (test_X, test_y), (val_X, val_y)
'''
# file = './criteo_sampled_data.csv'
data_df = pd.read_csv('./criteo_sampled_data.csv', iterator=True)
sample_num=100000
data_df = data_df.get_chunk(sample_num)

sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]

names = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11',
             'I12', 'I13', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
             'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22',
             'C23', 'C24', 'C25', 'C26']


# ------------Label Encoder & MinMaxScaler--------
dense_features = ['I' + str(i) for i in range(1, 14)]
print(dense_features)
data_df[dense_features] = MinMaxScaler(feature_range=(0,1)).fit_transform(data_df[dense_features])

def create_criteo_dataset(data, embed_dim=8, test_size=0.2):  
    """A example about creating criteo dataset

    Args:
        file ([path]): [dataset's path]
        embed_dim (int, optional): [the embedding dimension of sparse features]. Defaults to 8.
        read_part (bool, optional): [whether to read part of it]. Defaults to True.
        sample_num (int, optional): [the number of instances if read_part is True]. Defaults to 100000.
        test_size (float, optional): [ratio of test dataset]. Defaults to 0.2.
        return: feature columns, train, test
    """
    feature_columns = [[DenseFeature(feat) for feat in dense_features]] + \
                      [[SparseFeature(feat, len(data_df[feat].unique()), embed_dim=8) for feat in sparse_features]]
    
    print(data)
    train, test = train_test_split(data, test_size=test_size)
    print('train', train)
    train_X = [train[dense_features].values, train[sparse_features].values]
    train_y = train['label'].values
    test_X = [test[dense_features].values, test[sparse_features].values]
    test_y = test['label'].values
    
    return feature_columns, (train_X, train_y), (test_X, test_y)
'''
def SparseFeature(feat, feat_num, embed_dim=4):    
    """
    create dictionary for sparse feature
    Args:
        feat ([str]): [feature name]
        feat_num ([int]): [total number of sparse features that do not repeat]
        embed_dim (int, optional): [embedding dimension]. Defaults to 4.
    """
    return {'feat':feat, 'feat_num':feat_num, 'embed_dim':embed_dim}

def DenseFeature(feat):  
    """create dictionary for dense feature

    Args:
        feat ([str]): [dense feature name]
    """
    return {'feat':feat}

    


if __name__ == "__main__":
    # create_criteo_dataset('./criteo_sample.txt')
    create_criteo_dataset('./criteo_sampled_data.csv')
    
    