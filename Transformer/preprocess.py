import numpy as np
import pandas as pd

users = pd.read_csv('E:/32.Project4CV/SVD4RS/ml-1m/users.dat', sep='::', names=['user_id', 'sex', 'age_group', 'occupation','zip_code'])
ratings = pd.read_csv('E:/32.Project4CV/SVD4RS/ml-1m/ratings.dat', sep='::', names=['user_id', 'movie_id', 'rating', 'unix_timestamp'])
movies = pd.read_csv('E:/32.Project4CV/SVD4RS/ml-1m/movies.dat', sep="::", names=['movie_id', 'title', 'genres'])

# 合并为一个大DataFrame
matrix = pd.merge(users, ratings, on='user_id', how='left')
matrix = pd.merge(matrix, movies, on='movie_id', how='left')

# 性别特征做0-1编码
temp = pd.get_dummies(matrix['sex'], prefix = 'sex')
matrix = pd.concat([matrix, temp], axis=1)
matrix.drop('sex', axis=1, inplace=True)


# 特征工程：单标签编码
# 这里不使用LabelEncoder，否则后面再处理 DataFrame中list变为 str 会出错
users["user_id"] = users["user_id"].apply(lambda x: f"user_{x}")
users["age_group"] = users["age_group"].apply(lambda x: f"group_{x}")
users["occupation"] = users["occupation"].apply(lambda x: f"occupation_{x}")

movies["movie_id"] = movies["movie_id"].apply(lambda x: f"movie_{x}")

ratings["movie_id"] = ratings["movie_id"].apply(lambda x: f"movie_{x}")
ratings["user_id"] = ratings["user_id"].apply(lambda x: f"user_{x}")
ratings["rating"] = ratings["rating"].apply(lambda x: float(x))

# 特征工程：电影类型genres多标签编码
genres = ["Action","Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary","Drama","Fantasy", "Film-Noir", "Horror", "Musical","Mystery",
    "Romance","Sci-Fi","Thriller","War","Western"]

for genre in genres: 
    movies[genre] = movies['genres'].apply(lambda x: int(genre in x.split('|')))
    
# 特征工程：电影评分、电影id序列化
# 我们按照时间（unix_timestamp）对评分进行排序，再将 movie_id 和 ratings 按 user_id分组（groupby)
ratings_group = ratings.sort_values(by=["unix_timestamp"]).groupby("user_id")

ratings_data = pd.DataFrame({"user_id": list(ratings_group.groups.keys()),
                            "movie_ids": list(ratings_group.movie_id.apply(list)),
                            "ratings": list(ratings_group.rating.apply(list)),
                            "timestamps": list(ratings_group.unix_timestamp.apply(list)),}
)

# 构建数据集：用户看过电影id每4个做一组，用户评分每4个做一组
# 将 movie_ids 列表分成一组固定长度的序列，ratings也是如此。 设置sequence_length 变量就可以更改输入序列的长度；更改step_size以控制要为每个用户生成的sequences。
# movie_ids[-sequence_length: ] # 最近200个用户点击行为的价值更大
sequence_length = 4
step_size = 2

def create_sequences(values, window_size, step_size):
    sequences = []
    start_index = 0
    while True:
        end_index = start_index + window_size
        seq = values[start_index:end_index]
        if len(seq) < window_size:
            seq = values[-window_size:]
            if len(seq) == window_size:
                sequences.append(seq)
            break
        sequences.append(seq)
        start_index += step_size
    return sequences

ratings_data.movie_ids = ratings_data.movie_ids.apply(
    lambda ids: create_sequences(ids, sequence_length, step_size)
)

ratings_data.ratings = ratings_data.ratings.apply(
    lambda ids: create_sequences(ids, sequence_length, step_size)
)

del ratings_data["timestamps"]

# 每个序列在DataFrame中单独成行
ratings_data_movies = ratings_data[["user_id", "movie_ids"]].explode("movie_ids", #ignore_index=True
)
ratings_data_rating = ratings_data[["ratings"]].explode("ratings") # ignore_index=True
ratings_data_transformed = pd.concat([ratings_data_movies, ratings_data_rating], axis=1)
ratings_data_transformed = ratings_data_transformed.join(
    users.set_index("user_id"), on="user_id"
)

# pandas中的 list --> str : [4.0, 4.0, 5.0, 5.0]  --> 4.0, 4.0, 5.0, 5.0
ratings_data_transformed.movie_ids = ratings_data_transformed.movie_ids.apply(
    lambda x: ",".join(x)
)
ratings_data_transformed.ratings = ratings_data_transformed.ratings.apply(
    lambda x: ",".join([str(v) for v in x])
)

del ratings_data_transformed["zip_code"]

ratings_data_transformed.rename(
    columns={"movie_ids": "sequence_movie_ids", "ratings": "sequence_ratings"},
    inplace=True,
)

'''
    user_id	     sequence_movie_ids	                  sequence_ratings	sex	age_group occupation
0	user_1	movie_3186,movie_1721,movie_1270,movie_1022	4.0,4.0,5.0,5.0	F	group_1	occupation_10
1	user_1	movie_1270,movie_1022,movie_2340,movie_1836	5.0,5.0,3.0,5.0	F	group_1	occupation_10
'''

# 在sequence_length=4且 step_size=2的情况下，我们得到498,623个序列。
# 85％数据作为训练集， 15％的数据作为测试集

random_selection = np.random.rand(len(ratings_data_transformed.index)) <= 0.85
train_data = ratings_data_transformed[random_selection]
test_data = ratings_data_transformed[~random_selection]

train_data.to_csv("E:/32.Project4CV/SVD4RS/transformer/train_data.csv", index=False, sep="|", header=False)
test_data.to_csv("E:/32.Project4CV/SVD4RS/transformer/test_data.csv", index=False, sep="|", header=False)
'''
user_1|movie_3186	movie_1721	movie_1270	movie_1022|4.0	4	5	5.0|F|group_1|occupation_10
user_1|movie_1246	movie_2762	movie_661	movie_2918|4.0	4	3	4.0|F|group_1|occupation_10
user_1|movie_2791	movie_1029	movie_2321	movie_1197|4.0	5	3	3.0|F|group_1|occupation_10
user_1|movie_594	movie_2398	movie_1545	movie_527|4.0	4	4	5.0|F|group_1|occupation_10
'''

print('finished')