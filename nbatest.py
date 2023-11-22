import numpy as np
import requests
from bs4 import BeautifulSoup
from io import StringIO
import pandas as pd
from sklearn.linear_model import LogisticRegression

url = "https://www.basketball-reference.com/leagues/NBA_2024.html"
response = requests.get(url)

games_url = "https://www.basketball-reference.com/leagues/NBA_2024_games.html"
games_response = requests.get(games_url)

if games_response.status_code == 200:  # success lol
    html = games_response.text
    soup = BeautifulSoup(html, 'html.parser')

    games_stats_table = soup.find('div', {'id': 'div_schedule'})  # team stats per game

    if games_stats_table:
        div_contents = games_stats_table.encode_contents()
        div_contents_io = StringIO(div_contents.decode())
        df_games = pd.read_html(div_contents_io)[0]
    else:
        print("Games stats not found")

df_games.drop(columns=['Start (ET)', 'Unnamed: 6', 'Unnamed: 7', 'Notes'], inplace=True)

print("games stats:\n", df_games)

if response.status_code == 200:  # success
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')

    team_stats_table = soup.find('div', {'id': 'div_per_game-team'})  # team stats per game

    if team_stats_table:
        div_contents = team_stats_table.encode_contents()
        div_contents_io = StringIO(div_contents.decode())
        df_team = pd.read_html(div_contents_io)[0]
    else:
        print("Team stats not found")

    opponent_stats_table = soup.find('div', {'id': 'div_per_game-opponent'})  # opponent stats per game

    if opponent_stats_table:
        div_contents = opponent_stats_table.encode_contents()
        div_contents_io = StringIO(div_contents.decode())
        df_opponent = pd.read_html(div_contents_io)[0]
    else:
        print("Opponent stats not found")

    adv_team_stats_table = soup.find('div', {'id': 'all_advanced_team'})  # advanced team stats per game

    if adv_team_stats_table:
        div_contents = adv_team_stats_table.encode_contents()
        div_contents_io = StringIO(div_contents.decode())
        df_adv = pd.read_html(div_contents_io)[0]
    else:
        print("Team advanced stats not found")
else:
    print('Failed to fetch main page')

df_team.drop(30, inplace=True)  # remove league average
df_team.drop(['Rk', 'G', 'MP'], axis=1, inplace=True)  # remove old index, and games/mins played
df_team.rename(columns={'PTS': 'PPG'}, inplace=True)
df_team = df_team.sort_values(by='Team')  # sort alphabetically
df_team = df_team.reset_index(drop=True)  # reset indices to be alphabetically (i like it more this way)

df_opponent.drop(30, inplace=True)  # remove league average
df_opponent.drop(['Rk', 'G', 'MP'], axis=1, inplace=True)  # remove old index, and games/mins played
df_opponent.rename(columns={'PTS': 'PPG'}, inplace=True)
df_opponent = df_opponent.sort_values(by='Team')  # sort alphabetically
df_opponent = df_opponent.reset_index(drop=True)  # reset indices to be alphabetically (i like it more this way)
new_columns = ['o' + col if col != 'Team' else col for col in df_opponent.columns]
df_opponent.rename(columns=dict(zip(df_opponent.columns, new_columns)),
                   inplace=True)  # prepend all opponent stats with 'o'

df_merged = df_team.merge(df_opponent, on='Team')  # merge team and opponent stats

df_adv.columns = df_adv.columns.get_level_values(1)  # ignore top header (unnecessary categories)
df_adv.drop(30, inplace=True)  # remove league average
df_adv = df_adv[['Team', 'MOV', 'ORtg', 'DRtg', 'NRtg', 'Pace', 'TS%', 'eFG%']]  # select desired stats only
df_adv = df_adv.sort_values(by='Team')  # sort alphabetically
df_adv = df_adv.reset_index(drop=True)  # reset indices to be alphabetically (i like it more this way)

df_merged = df_merged.merge(df_adv, on='Team')  # merge team and opponent stats with advanced stats

print(df_merged)

# example queries
dubs_data = df_merged[df_merged['Team'] == 'Golden State Warriors']
print(dubs_data['PPG'])
print(dubs_data['oPPG'])
pacers_data = df_merged[df_merged['Team'] == 'Indiana Pacers']
print(pacers_data['Pace'])

# ipynb?
# last 5, rest, home and away
# other desirable features:
# last 5/10 games record (streak), days since last game (rest)
# https://www.basketball-reference.com/leagues/NBA_2024_standings.html - home vs away record, h2h matchups?
# PER (advanced stat)?, a/to ratio?
# individual player stats vs injuries?
# get per possession rather than per game? (mixed w pace?)
# get df_adv_opponent?


'''
# use simple logistic regression as starting point?
# predicting games
def display_probs(y_pred, X_test):
   for g in range(len(y_pred)):
       win_prob = round(y_pred[g], 2)
       away_team = X_test.reset_index().drop(columns = 'index').loc[g,'away_name']
       home_team = X_test.reset_index().drop(columns = 'index').loc[g,'home_name']
       print(f'The {away_team} have a probability of {win_prob} of beating the {home_team}.')




# splitting into train and test sets
msk = np.random.rand(len(df_merged)) < 0.8
train_df = df_merged[msk]
test_df = df_merged[~msk]


# feature matrix (X) and target variable (y) for training and testing
# need a 'result' W/L between each team they go against from the games_df for each team
X_train = train_df
y_train = train_df[['result']]
X_test = test_df
y_test = test_df[['result']]


# can use different parameters
clf = LogisticRegression(
   penalty='l1', dual=False, tol=0.001, C=1.0, fit_intercept=True,
   intercept_scaling=1, class_weight='balanced', random_state=None,
   solver='liblinear', max_iter=1000, multi_class='ovr', verbose=0
)


clf.fit(X_train, np.ravel(y_train.values))
y_pred = clf.predict_proba(X_test)
y_pred = y_pred[:, 1]
display_probs(y_pred, test_df)
'''
