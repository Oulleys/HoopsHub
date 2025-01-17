import pandas as pd

# Load the dataset
file_path = "../data/nba_games.csv"  # Adjust the path if necessary
nba_games_data = pd.read_csv(file_path)

# Extract unique team IDs
unique_teams = pd.concat([nba_games_data['team'], nba_games_data['team_opp']]).unique()

# Print or save the unique teams
print("Unique Team IDs:", unique_teams)

# Optionally save the unique team IDs to a file
with open("../data/unique_teams.txt", "w") as f:
    for team in unique_teams:
        f.write(f"{team}\n")