#------------------------------------------------------------------------------
# Carson Sytner
# Honors Data Science Project
# This program analyzes data from student and entrepreneur tweets
#------------------------------------------------------------------------------

import pandas as pd
from collections import Counter
import emoji 
import re
import matplotlib.pyplot as plt
from datetime import date
import numpy as np

# Phase 1: Tweets before March 1st
# Pahse 2: Tweets from March 1st through March 15th
# Phase 3: Tweets made after March 15th

# Calculates the total, average length, URL frequency, Retweet
# frequency, and emoji frequency of the tweet data
def phase_data(tweets):
    return {"Tweets" : len(tweets), 
            "Avg len" : sum([len(t) for t in tweets]) / len(tweets), 
            "URL Freq" : tweets.str.contains("https").sum() / len(tweets),
            "RT Freq" : tweets.str.startswith("RT").values.sum() / len(tweets),
            "Emoji Freq" : len( [ 1 for tweet in tweets for char in tweet if char in emojis] ) / len(tweets) 
            }

# Returns a dataframe with various statistics
def stats(data):
    median = data.reset_index(0).groupby(["Phase", "Type"]).median()
    median["Stat"] = "Median"
    
    mean = data.reset_index(0).groupby(["Phase", "Type"]).mean()
    mean["Stat"] = "Mean"
    
    std = data.reset_index(0).groupby(["Phase", "Type"]).std()
    std["Stat"] = "Std"
    
    d_max = data.reset_index(0).groupby(["Phase", "Type"]).max()
    d_max["Stat"] = "Max"
    
    d_min = data.reset_index(0).groupby(["Phase", "Type"]).min()
    d_min["Stat"] = "Min"
    
    return  pd.concat([median, mean, std, d_max, d_min])
    
# emojis set
emojis = set(emoji.EMOJI_UNICODE['en'].values())

# Read the files
students = pd.read_pickle("DataSet/students.p")
entrepreneurs = pd.read_pickle("DataSet/entrepreneurs.p")

# Calculate and group by phase and id
students["Phase"] = (students.date >= "2020-03-01").astype(int) + (students.date > "2020-03-15").astype(int)
s_summary = students.groupby(["Phase", "id"])

entrepreneurs["Phase"] = (students.date >= "2020-03-01").astype(int) + (students.date > "2020-03-15").astype(int)
e_summary = entrepreneurs.groupby(["Phase", "id"])

# Number of days in each phase
p0_days = (date(2020, 3, 1) - students.date.min().date()).days
p1_days = (date(2020, 3, 16) - date(2020, 3, 1)).days
p2_days = (students.date.max().date() - date(2020, 3, 16)).days

s_data = s_summary["text"].apply(phase_data).unstack()
s_data["Type"] = "S"

e_data = e_summary["text"].apply(phase_data).unstack()
e_data["Type"] = "E"

final = pd.concat([s_data, e_data])
final_data = final.reset_index(0)

final_data.loc[final_data["Phase"] == 0, "Tweet Frequency"] = \
    final_data.loc[final_data["Phase"] == 0, "Tweets"] / p0_days
    
final_data.loc[final_data["Phase"] == 1, "Tweet Frequency"] = \
    final_data.loc[final_data["Phase"] == 1, "Tweets"] / p1_days
    
final_data.loc[final_data["Phase"] == 2, "Tweet Frequency"] = \
    final_data.loc[final_data["Phase"] == 2, "Tweets"] / p2_days


data = final_data[['Avg len', 'Tweet Frequency', 'URL Freq', 'RT Freq', 'Emoji Freq', \
        'Type', 'Phase']]

data.to_csv("tweet_data.csv") # All the data collected

data_cols = ['Phase', 'Stat', 'Tweets', 'Avg len', 'URL Freq', 'RT Freq',
       'Emoji Freq', 'Tweet Frequency']

# All the statistics from the data
data_stats = (stats(final_data).drop(columns=["id"])).sort_values(by=["Stat","Phase", "Type"]) \
    .reset_index(0).reindex(columns=data_cols)
    
data_stats.to_csv("tweet_data_stats.csv")
