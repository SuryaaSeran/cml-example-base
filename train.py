import pandas as pd
import numpy as np
import copy
import re
import math
from scipy import spatial
from sklearn.neighbors import NearestNeighbors



df = pd.read_csv('netflix_titles.csv')
df.fillna('missing', inplace = True)

country = []
release_year = [] 
rating = []
duration = [] 
genres = []

recommendation_cols = ['country', 'release_year', 'rating', 'duration', 'listed_in']
df_new = copy.deepcopy(df[recommendation_cols])

def split_by_delimeters(target_list):
    """
    this method splits a target list by some delimeters
    """
    result = []
    for i in target_list:
        delimiters = ",", "&"
        regex_pattern = '|'.join(map(re.escape, delimiters))
        result.extend(re.split(regex_pattern, i))
    result = [i.strip() if i not in ['', 'missing'] else i for i in result]
    return result

# preparing all columns for the dataset
country = list(set(split_by_delimeters(df_new['country'])))
release_year = list(set(df_new['release_year']))
release_year = [str(i) for i in release_year]
ratings = list(set(df_new['rating']))
seasons_durations = ['1_season', '2_season', '3_season', '4_season','5+_season']
movies_durations = ['0_25_min', '26_50_min', '51_75_min', '76_100_min', 
                    '101_125_min', '126_150_min', '151+_min' ]
durations = seasons_durations + movies_durations
genres = list(set(split_by_delimeters(df_new['listed_in'])))

# combining all columns for the one hot encoded vector form
all_columns = country + release_year + ratings + durations + genres
all_columns.remove('missing')

# initializes a df with '0' values for the one-hot-encoded vector
ohe_df = pd.DataFrame(0, index = np.arange(len(df_new)), columns = all_columns) 

def duration_adjustment(duration: str) -> str:
    try:
        dur_list = []
        if 'Season' in duration:
            temp_res = duration.split()
            no_of_seasons = int(temp_res[0])
            if no_of_seasons <5:
                return seasons_durations[no_of_seasons - 1]
            return seasons_durations[-1]

        else:
            temp_res = duration.split()
            runtime_mins = int(temp_res[0])
            if runtime_mins <= 150:
                index = math.ceil((runtime_mins/25) - 1.0)
                return movies_durations[index]
            return movies_durations[-1]
    except:
        return 'missing'

def return_columns(row):
    """
    recieves a df row and returns the respective columns/features
    that the item i.e. movie falls in
    """
    result_cols = []
    result_cols.extend(split_by_delimeters([row['country']]))
    result_cols.extend(split_by_delimeters([row['listed_in']]))
    result_cols.append(str(row['release_year']))
    result_cols.append(row['rating'])
    result_cols.append(duration_adjustment(str(row['duration'])))
    if 'missing' in result_cols:
        result_cols.remove('missing')
    return result_cols

# preparing the one hot encoded df of all items i.e. movies as vectors
for ind,row in df_new.iterrows():
    ohe_df.loc[ind, return_columns(row)] = 1

def recommend_by_knn(movie, top_items):
    """
    recommends top_similar movies based on knn algorithm
    """
    movie_index = df[df['title'] == movie].index[0]
    vector = ohe_df.iloc[movie_index]
    knn = NearestNeighbors(n_neighbors= top_items + 1, algorithm='auto')
    knn.fit(ohe_df.values)
    indexes = list(knn.kneighbors([vector], top_items + 1, return_distance=False)[0])
    return list(df.iloc[indexes]['title'])[1:]

# the first row is the movie itself and the rest are recommendations
recommend_by_knn('Friends', 10)
