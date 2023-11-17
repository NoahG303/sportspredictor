import requests
from bs4 import BeautifulSoup
from io import StringIO
import pandas as pd

url = "https://www.basketball-reference.com/leagues/NBA_2024.html"
response = requests.get(url)

if response.status_code == 200: # success
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')

    team_stats_table = soup.find('div', {'id': 'div_per_game-team'}) # team stats per game

    if team_stats_table:
        div_contents = team_stats_table.encode_contents()
        div_contents_io = StringIO(div_contents.decode())
        df_team = pd.read_html(div_contents_io)[0]
    else:
        print("Team stats not found")

    opponent_stats_table = soup.find('div', {'id': 'div_per_game-opponent'}) # opponent stats per game

    if opponent_stats_table:
        div_contents = opponent_stats_table.encode_contents()
        div_contents_io = StringIO(div_contents.decode())
        df_opponent = pd.read_html(div_contents_io)[0]
    else:
        print("Opponent stats not found")

    adv_team_stats_table = soup.find('div', {'id': 'all_advanced_team'}) # advanced team stats per game

    if adv_team_stats_table:
        div_contents = adv_team_stats_table.encode_contents()
        div_contents_io = StringIO(div_contents.decode())
        df_adv = pd.read_html(div_contents_io)[0] 
    else:
        print("Team advanced stats not found")
else:
    print('Failed to fetch main page')

df_team.drop(30, inplace=True) # remove league average
df_team.drop(['Rk', 'G', 'MP'], axis=1, inplace=True) # remove old index, and games/mins played
df_team.rename(columns={'PTS': 'PPG'}, inplace=True)
df_team = df_team.sort_values(by='Team') # sort alphabetically
df_team = df_team.reset_index(drop=True) # reset indices to be alphabetically (i like it more this way)

df_opponent.drop(30, inplace=True) # remove league average
df_opponent.drop(['Rk', 'G', 'MP'], axis=1, inplace=True) # remove old index, and games/mins played
df_opponent.rename(columns={'PTS': 'PPG'}, inplace=True)
df_opponent = df_opponent.sort_values(by='Team') # sort alphabetically
df_opponent = df_opponent.reset_index(drop=True) # reset indices to be alphabetically (i like it more this way)
new_columns = ['o' + col if col != 'Team' else col for col in df_opponent.columns]
df_opponent.rename(columns=dict(zip(df_opponent.columns, new_columns)), inplace=True) # prepend all opponent stats with 'o'

df_merged = df_team.merge(df_opponent, on='Team') # merge team and opponent stats

df_adv.columns = df_adv.columns.get_level_values(1) # ignore top header (unnecessary categories)
df_adv.drop(30, inplace=True) # remove league average
df_adv = df_adv[['Team', 'MOV', 'ORtg', 'DRtg', 'NRtg', 'Pace', 'TS%', 'eFG%']] # select desired stats only
df_adv = df_adv.sort_values(by='Team') # sort alphabetically
df_adv = df_adv.reset_index(drop=True) # reset indices to be alphabetically (i like it more this way)

df_merged = df_merged.merge(df_adv, on='Team') # merge team and opponent stats with advanced stats

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