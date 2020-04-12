import pandas as pd 
import numpy as np 
from ast import literal_eval
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

#  helper functions
def json_to_arr(cell, wanted = "name"): 
    cell = literal_eval(cell)
    if cell == [] or (isinstance(cell, float) and cell.isna()):
        return np.nan
    result = []
    counter = 0
    for element in cell:
        if counter < 3:
            result.append(element[wanted])
            counter += 1
        else:
            break
    return result[:3]

# returns the values and occurance times or "limiter" amount of different parameters in a 2D list
def list_counter(col, limiter = 9999, log = True):
    result = dict()
    for cell in col:
        if isinstance(cell, float):
            continue
        for element in cell:
            if element in result:
                result[element] += 1
            else:
                result[element] = 1
    if log:
        print("Size of words:", len(result))
    result = {k: v for k, v in sorted(result.items(), key=lambda item: item[1], reverse=True)}
    if log:
        print("Sorted result is:")
    counter = 1
    sum_selected = 0
    total_selected = 0
    rest = 0
    returned = []
    for i in result: 
        if counter > limiter:
            total_selected += result[i]
        else:
            counter += 1
            sum_selected += result[i]
            total_selected += result[i]
            if log:
                print(result[i], " - ", i) 
            returned.append([i, result[i]])
    if log:
        print("Covered:", sum_selected, "out of", total_selected, "\n")
    return returned

# Load movies_metadata.csv
movies=pd.read_csv('./dataset/trimmed/initial_movies_metadata.csv', low_memory=False)

# normalize some simple features
movies["runtime"] = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(movies["runtime"].values.reshape(-1, 1))).values
movies["release_month"] = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(movies["release_month"].values.reshape(-1, 1))).values

# only movies with a budget > 250,000 and revenue > 500,000
movies = movies.loc[movies['budget'] > 250000]
movies = movies.loc[movies['revenue'] > 500000]

# genres -> initialize to 0
movies[['genres']] = movies[['genres']].applymap(json_to_arr)
genres = list_counter(movies["genres"].values, log=False)
for g in genres:
    cat_name = 'genre_' + g[0].lower().replace(" ", "_")
    movies[cat_name] = 0

for index, mov in movies.iterrows():
    # genre -> set 1 where has genre
    if isinstance(mov['genres'], list):
        for g in mov['genres']:
            cat_name = 'genre_' + g.lower().replace(" ", "_")
            movies.at[index, cat_name] = 1
    # movie has profit
    if mov['revenue'] > mov['budget']:
        movies.at[index, 'has_profit'] = 1
    else:
        movies.at[index, 'has_profit'] = 0
    # categorize revenue
    movies.at[index, 'rev_cat_01'] = 0
    movies.at[index, 'rev_cat_02'] = 0
    movies.at[index, 'rev_cat_03'] = 0
    movies.at[index, 'rev_cat_04'] = 0
    if mov['revenue'] <= 1000000:
        movies.at[index, 'rev_cat_01'] = 1
    elif mov['revenue'] <= 40000000:
        movies.at[index, 'rev_cat_02'] = 1
    elif mov['revenue'] <= 150000000:
        movies.at[index, 'rev_cat_03'] = 1
    else:
        movies.at[index, 'rev_cat_04'] = 1

# encode budget and reveneu
movies["budget"] = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(movies["budget"].values.reshape(-1, 1))).values
movies["revenue_minmax"] = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(movies["revenue"].values.reshape(-1, 1))).values


# encode the director, main_actor, and production company 'director', 'main_actor', 
for col in ['production_company']:
    print(col)
    movies = movies.join(pd.get_dummies(pd.DataFrame(pd.unique(movies[col].values), columns=[col]), columns=[col], prefix=[col] ))

# cleanup
movies = movies.drop(["revenue", "genres", "production_countries", "spoken_languages", 'director', 'main_actor', 'production_company'], axis=1) # drops the selected columns

y_encoded_categorical = movies[['rev_cat_01', 'rev_cat_02', 'rev_cat_03', 'rev_cat_04']]
y_encoded_profit = movies[['has_profit']]
y_encoded_minmax = movies[['revenue_minmax']]

movies = movies.drop(['rev_cat_01', 'rev_cat_02', 'rev_cat_03', 'rev_cat_04', 'has_profit', 'revenue_minmax'], axis=1) # drops the selected columns

# save out files
movies.to_csv('./dataset/trimmed/movies_metadata_x.csv', index=False)
y_encoded_categorical.to_csv('./dataset/trimmed/movies_metadata_y_categorical.csv', index=False)
y_encoded_profit.to_csv('./dataset/trimmed/movies_metadata_y_profit.csv', index=False)
y_encoded_minmax.to_csv('./dataset/trimmed/movies_metadata_y_minmax.csv', index=False)