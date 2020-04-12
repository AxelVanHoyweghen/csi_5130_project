import pandas as pd 
import numpy as np 
from ast import literal_eval
from easymoney.money import EasyPeasy
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

ep = EasyPeasy()

#  helper functions

# Function to get the director from the cast property
def get_director(x):
    x = literal_eval(x)
    for i in x:
        if i == "[]" or isinstance(i, float):
            return np.nan
        if i['job'] == 'Director':
            return i['name'].replace(" ", "_").lower()
    return np.nan

# get the main actor
def get_main_actor(x):
    x = literal_eval(x)
    for i in x:
        if i == "[]" or isinstance(i, float):
            return np.nan
        if i['order'] == 0:
            return i['name'].replace(" ", "_").lower()
    return np.nan

# Function to get the director from the cast property
def get_production_company(x):
    x = literal_eval(x)
    for i in x:
        if i == "[]" or isinstance(i, float):
            return np.nan
        return i['name'].replace(" ", "_").lower()
    return np.nan

# Load movies_metadata.csv
movies=pd.read_csv('./dataset/original/movies_metadata.csv', low_memory=False)

# clean movies_metadata.csv

# drop rows where following are null: release_date, title, budget, revenue, runtime, status
movies.dropna(subset=["budget", "revenue", "runtime", "original_language", "status", "release_date"], inplace=True)

movies['id'] =pd.to_numeric(movies['id'], errors='coerce', downcast="integer")
movies['popularity'] =pd.to_numeric(movies['popularity'], errors='coerce', downcast="float") 
movies['budget'] =pd.to_numeric(movies['budget'], errors='coerce', downcast="float") 
movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
movies['release_year'] = movies['release_date'].dt.year
movies['release_month'] = movies['release_date'].dt.month

# These features are not
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

# account for inflation for older movies
for index, mov in movies.iterrows():
    movies.at[index, 'budget'] = ep.normalize(amount=mov['budget'], region="US", from_year=mov['release_year'], to_year="latest", pretty_print=False)
    movies.at[index, 'revenue'] = ep.normalize(amount=mov['revenue'], region="US", from_year=mov['release_year'], to_year="latest", pretty_print=False)

movies.drop(['release_year'], axis=1, inplace=True)

# select most prominent production company
movies['production_company'] = movies['production_companies'].apply(get_production_company)
movies = movies.drop(["production_companies"], axis=1)

# merge on credits.csv
movie_credits = pd.read_csv('./dataset/original/credits.csv', low_memory=False)

# extract director
movie_credits['director'] = movie_credits['crew'].apply(get_director)
movie_credits.drop(['crew'], axis=1, inplace=True)

# extract main actor
movie_credits['main_actor'] = movie_credits['cast'].apply(get_main_actor)
movie_credits.drop(['cast'], axis=1, inplace=True)

# drop unnecessary columns or with empty director and cast
movie_credits.drop(movie_credits[(movie_credits['main_actor'].isna())&(movie_credits['director'].isna())].index, inplace=True)

# merge movie_credits and movie
merged_data = pd.merge(movies, movie_credits, on=['id'], how='left')

# remove id (not needed for now)
merged_data = merged_data.drop(['id'], axis=1) # drops the selected columns

# drop row if any of the columns are nan, we want clean data with all columns available
cols  = list(merged_data)
merged_data.dropna(subset=np.array(cols), inplace=True)

# save file
merged_data.to_csv('./dataset/trimmed/initial_movies_metadata.csv', index=False)
