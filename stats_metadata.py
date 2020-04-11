import pandas as pd 
import numpy as np 
from ast import literal_eval
import matplotlib.pyplot as plt

# with help from: https://www.kaggle.com/barisbatuhan/movies-dataset-data-cleaning-and-analysis

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
trimmed_movies=pd.read_csv('./dataset/trimmed/movies_metadata.csv', low_memory=False)

# show and save the statistics

# trimmed_movies[["popularity", "revenue", "budget", "runtime", "vote_average", "vote_count", "release_year"]].describe().to_csv("./stats/after_trim_stats.csv")
# print(movies[["popularity", "revenue", "budget", "runtime", "vote_average", "vote_count", "release_year"]].describe())

# genres
genres_occur = list_counter(trimmed_movies["cast"].values, log=False)
genres = pd.DataFrame.from_records(genres_occur, columns=["cast", "count"])
genres.to_csv("./stats/after_trim_cast.csv")
genres.plot(kind = 'bar', x="cast")
plt.savefig('./stats/after_trim_cast.png')
plt.clf()
plt.close()

# # genres
# genres_occur = list_counter(trimmed_movies["genres"].values, log=False)
# genres = pd.DataFrame.from_records(genres_occur, columns=["genres", "count"])
# genres.to_csv("./stats/after_trim_genres.csv")
# genres.plot(kind = 'bar', x="genres")
# plt.savefig('./stats/after_trim_genres.png')
# plt.clf()
# plt.close()

# # countries
# countries_occur = list_counter(trimmed_movies["production_countries"].values, log=False)
# countries = pd.DataFrame.from_records(countries_occur, columns=["countries", "count"])
# countries.to_csv("./stats/after_trim_countries.csv")
# countries.plot(kind = 'bar', x="countries")
# plt.savefig('./stats/after_trim_countries.png')
# plt.clf()
# plt.close()

# # companies
# companies_occur = list_counter(trimmed_movies["production_companies"].values, log=False)
# companies = pd.DataFrame.from_records(companies_occur, columns=["companies", "count"])
# companies.to_csv("./stats/after_trim_companies.csv")
# companies.plot(kind = 'bar', x="companies")
# plt.savefig('./stats/after_trim_companies.png')
# plt.clf()
# plt.close()