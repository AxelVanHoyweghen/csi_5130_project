import pandas as pd 
import numpy as np 
from ast import literal_eval
from easymoney.money import EasyPeasy
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

ep = EasyPeasy()

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


# Function to get the director from the cast property
def get_director(x):
    x = literal_eval(x)
    for i in x:
        if i == "[]" or isinstance(i, float):
            return np.nan
        if i['job'] == 'Director':
            return i['name']
    return np.nan

# Load movies_metadata.csv
movies=pd.read_csv('./dataset/original/movies_metadata.csv', low_memory=False)

# clean movies_metadata.csv

# drop rows where following are null: release_date, title, budget, revenue, runtime, status
movies.dropna(subset=["budget"], inplace=True)
movies.dropna(subset=["revenue"], inplace=True)
movies.dropna(subset=["runtime"], inplace=True)
movies.dropna(subset=["original_language"], inplace=True)
movies.dropna(subset=["status"], inplace=True)
movies.dropna(subset=["release_date"], inplace=True)


movies['id'] =pd.to_numeric(movies['id'], errors='coerce', downcast="integer")
movies['popularity'] =pd.to_numeric(movies['popularity'], errors='coerce', downcast="float") 
movies['budget'] =pd.to_numeric(movies['budget'], errors='coerce', downcast="float") 
movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
movies['release_year'] = movies['release_date'].dt.year
movies['release_month'] = movies['release_date'].dt.month

movies = movies.drop(["homepage", "poster_path", "video", "imdb_id", "overview", "original_title", "title", "vote_average", "vote_count", "popularity", "release_date"], axis=1) 

# convert collection info to 0 or 1
movies['belongs_to_collection'] = movies['belongs_to_collection'].fillna("None")
movies['belongs_to_collection'] = (movies['belongs_to_collection'] != "None").astype(int)

# convert tagline to 0 or 1
movies['tagline'] = movies['tagline'].fillna("None")
movies['has_tagline'] = (movies['tagline'] != "None").astype(int)
movies.drop(['tagline'], axis=1, inplace=True)

# filter based on conditions

# only English movies
movies = movies.loc[movies['original_language'] == 'en']
movies.drop(['original_language'], axis=1, inplace=True)

# only released movies
movies = movies.loc[movies['status'] == 'Released']
movies.drop(['status'], axis=1, inplace=True)

# remove adult movies
movies = movies.loc[movies['adult'] == 'False']
movies.drop(['adult'], axis=1, inplace=True)

# remove short films (runtime off 45 minutes or less)
movies = movies.loc[movies['runtime'] > 45]
movies["runtime"] = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(movies["runtime"].values.reshape(-1, 1))).values

# release_month 
enc_rm = pd.DataFrame(OneHotEncoder(handle_unknown='ignore').fit_transform(movies[['release_month']]).toarray())
enc_rm.columns = ['month_01', 'month_02', 'month_03', 'month_04', 'month_05', 'month_06', 'month_07', 'month_08', 'month_09', 'month_10', 'month_11', 'month_12']
movies = movies.join(enc_rm)
movies.drop(['release_month'], axis=1, inplace=True)

# genres -> initialize to 0
movies[['genres']] = movies[['genres']].applymap(json_to_arr)
genres = list_counter(movies["genres"].values, log=False)
# initialize all at 0
for g in genres:
    cat_name = 'genre_' + g[0].lower().replace(" ", "_")
    movies[cat_name] = 0

# production companies -> initialize to 0
movies[['production_companies']] = movies[['production_companies']].applymap(json_to_arr)
p_companies = list_counter(movies["production_companies"].values, log=False)
# initialize all at 0
for g in p_companies:
    cat_name = 'prod_comp_' + g[0].lower().replace(" ", "_")
    movies[cat_name] = 0

# account for inflation for older movies
for index, mov in movies.iterrows():
    movies.at[index, 'budget'] = ep.normalize(amount=mov['budget'], region="US", from_year=mov['release_year'], to_year="latest", pretty_print=False)
    movies.at[index, 'revenue'] = ep.normalize(amount=mov['revenue'], region="US", from_year=mov['release_year'], to_year="latest", pretty_print=False)

# only movies with a budget > 250,000 and revenue > 500,000
movies = movies.loc[movies['budget'] > 250000]
movies = movies.loc[movies['revenue'] > 500000]

movies.drop(['release_year'], axis=1, inplace=True)

movies["budget"] = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(movies["budget"].values.reshape(-1, 1))).values

for index, mov in movies.iterrows():
    # genre -> set 1 where has genre
    if isinstance(mov['genres'], list):
        for g in mov['genres']:
            cat_name = 'genre_' + g.lower().replace(" ", "_")
            movies.at[index, cat_name] = 1
    # production company -> set 1 where has company
    if isinstance(mov['production_companies'], list):
        for g in mov['production_companies']:
            cat_name = 'prod_comp_' + g.lower().replace(" ", "_")
            movies.at[index, cat_name] = 1
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

# cleanup
movies = movies.drop(["revenue", "genres", "production_countries", "production_companies", "spoken_languages"], axis=1) # drops the selected columns

merged_data = movies
merged_data = merged_data.drop(['id'], axis=1) # drops the selected columns

# reorder
cols  = list(merged_data)
cols.insert(0, cols.pop(cols.index('rev_cat_01')))
cols.insert(1, cols.pop(cols.index('rev_cat_02')))
cols.insert(2, cols.pop(cols.index('rev_cat_03')))
cols.insert(3, cols.pop(cols.index('rev_cat_04')))

reordered_data = merged_data[np.array(cols)]

# save out the trimmed data
reordered_data.to_csv('./dataset/trimmed/movies_metadata.csv', index=False)