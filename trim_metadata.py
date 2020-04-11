import pandas as pd 
import numpy as np 
from ast import literal_eval
from easymoney.money import EasyPeasy
from sklearn import preprocessing

# with help from: https://www.kaggle.com/barisbatuhan/movies-dataset-data-cleaning-and-analysis
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
movies.dropna(subset=["release_date"], inplace=True)
movies.dropna(subset=["title"], inplace=True)
movies.dropna(subset=["budget"], inplace=True)
movies.dropna(subset=["revenue"], inplace=True)
movies.dropna(subset=["runtime"], inplace=True)
movies.dropna(subset=["original_language"], inplace=True)
movies.dropna(subset=["status"], inplace=True)


movies["id"] =pd.to_numeric(movies['id'], errors='coerce', downcast="integer")
movies["popularity"] =pd.to_numeric(movies['popularity'], errors='coerce', downcast="float") 
movies["budget"] =pd.to_numeric(movies['budget'], errors='coerce', downcast="float") 
movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
movies['release_year'] = movies['release_date'].dt.year
movies['release_month'] = movies['release_date'].dt.month

# convert collection info to 0 or 1
movies['belongs_to_collection'] = movies['belongs_to_collection'].fillna("None")
movies['belongs_to_collection'] = (movies['belongs_to_collection'] != "None").astype(int)

# convert tagline to 0 or 1
movies['tagline'] = movies['tagline'].fillna("None")
movies['has_tagline'] = (movies['tagline'] != "None").astype(int)

# filter based on conditions

# only English movies
movies = movies.loc[movies['original_language'] == 'en']

# only released movies
movies = movies.loc[movies['status'] == 'Released']

# remove adult movies
movies = movies.loc[movies['adult'] == 'False']

# remove short films (runtime off 45 minutes or less)
movies = movies.loc[movies['runtime'] > 45]

# account for inflation for older movies
for index, mov in movies.iterrows():
    movies.at[index, 'budget'] = ep.normalize(amount=mov['budget'], region="US", from_year=mov['release_year'], to_year="latest", pretty_print=False)
    movies.at[index, 'revenue'] = ep.normalize(amount=mov['revenue'], region="US", from_year=mov['release_year'], to_year="latest", pretty_print=False)

# only movies with a budget > 250,000 and revenue > 500,000
movies = movies.loc[movies['budget'] > 250000]
movies = movies.loc[movies['revenue'] > 500000]

# modify data objects

# genres
movies[['genres']] = movies[['genres']].applymap(json_to_arr)
genres = list_counter(movies["genres"].values, log=False)
# initialize all at 0
for g in genres:
    cat_name = 'genre_' + g[0].lower().replace(" ", "")
    movies[cat_name] = 0

# production countries
# movies[['production_countries']] = movies[['production_countries']].applymap(lambda row: json_to_arr(row, "iso_3166_1"))
# p_countries = list_counter(movies["production_countries"].values, log=False)
# # initialize all at 0
# for g in p_countries:
#     cat_name = 'prod_country_' + g[0].lower().replace(" ", "")
#     movies[cat_name] = 0

# production comapnies
movies[['production_companies']] = movies[['production_companies']].applymap(json_to_arr)
p_companies = list_counter(movies["production_companies"].values, log=False)
# initialize all at 0
for g in p_companies:
    cat_name = 'prod_comp_' + g[0].lower().replace(" ", "")
    movies[cat_name] = 0

# # spoken languages
# movies[['spoken_languages']] = movies[['spoken_languages']].applymap(lambda row: json_to_arr(row, "iso_639_1"))
# spoken_languages = list_counter(movies["spoken_languages"].values, log=False)
# # initialize all at 0
# for g in spoken_languages:
#     cat_name = 'spoken_l_' + g[0].lower().replace(" ", "")
#     movies[cat_name] = 0    

# match categories
for index, mov in movies.iterrows():
    # genre
    if isinstance(mov['genres'], list):
        for g in mov['genres']:
            cat_name = 'genre_' + g.lower().replace(" ", "")
            movies.at[index, cat_name] = 1
    # production country
    # if isinstance(mov['production_countries'], list):
    #     for g in mov['production_countries']:
    #         cat_name = 'prod_country_' + g.lower().replace(" ", "")
    #         movies.at[index, cat_name] = 1
    # production company
    if isinstance(mov['production_companies'], list):
        for g in mov['production_companies']:
            cat_name = 'prod_comp_' + g.lower().replace(" ", "")
            movies.at[index, cat_name] = 1
    # # spoken language
    # if isinstance(mov['spoken_languages'], list):
    #     for g in mov['spoken_languages']:
    #         cat_name = 'spoken_l_' + g.lower().replace(" ", "")
    #         movies.at[index, cat_name] = 1
    
    # categorize revenue
    if mov['revenue'] <= 1000000: 
        movies.at[index, 'rev_cat'] = 1
    elif mov['revenue'] <= 40000000:
        movies.at[index, 'rev_cat'] = 2
    elif mov['revenue'] <= 150000000:
        movies.at[index, 'rev_cat'] = 3
    else:
        movies.at[index, 'rev_cat'] = 4

# drop columns that are not needed anymore
drop_df = ["adult", "homepage", "poster_path", "video", "imdb_id", "overview", "original_title", "title", "original_language", "vote_average", "vote_count", "popularity", "status", "tagline", "release_date", "release_year", "genres", "production_countries", "production_companies", "spoken_languages"]
movies = movies.drop(drop_df, axis=1) # drops the selected columns

# scale features
movies["budget"] = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(movies["budget"].values.reshape(-1, 1))).values
movies["revenue"] = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(movies["revenue"].values.reshape(-1, 1))).values
movies["runtime"] = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(movies["runtime"].values.reshape(-1, 1))).values
# movies["popularity"] = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(movies["popularity"].values.reshape(-1, 1))).values

# movies["vote_average"] = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(movies["vote_average"].values.reshape(-1, 1))).values
# movies["vote_count"] = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(movies["vote_count"].values.reshape(-1, 1))).values

# movies["release_year"] = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(movies["release_year"].values.reshape(-1, 1))).values
movies["release_month"] = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(movies["release_month"].values.reshape(-1, 1))).values

# movies["budget"] = preprocessing.normalize(movies["budget"], norm='l2')

# movies['revenue'] = budget_scaler.fit_transform(movies["revenue"])
# join on credits.csv
# movie_credits = pd.read_csv('./dataset/original/credits.csv', low_memory=False)

# # transform cast
# movie_credits['cast'] = movie_credits[['cast']].applymap(json_to_arr)

# # extract director
# movie_credits['director'] = movie_credits['crew'].apply(get_director)

# # drop unnecessary columns or with empty director and cast
# movie_credits.drop(['crew'], axis=1, inplace=True)
# movie_credits.drop(movie_credits[(movie_credits['cast'].isna())&(movie_credits['director'].isna())].index, inplace=True)

# # drop cast for now
# movie_credits.drop(['cast'], axis=1, inplace=True)

# # merge movie_credits and movie
# merged_data = pd.merge(movies, movie_credits, on=['id'], how='left')

# remove id (not needed for now)
merged_data = movies
merged_data = merged_data.drop(['id'], axis=1) # drops the selected columns

# reorder
cols  = list(merged_data)
cols.insert(0, cols.pop(cols.index('rev_cat')))

reordered_data = merged_data[np.array(cols)]

# reordered_data = reordered_data[['revenue', 'budget', 'runtime', 'release_month']]
# save out the trimmed data
reordered_data.to_csv('./dataset/trimmed/movies_metadata.csv', index=False)