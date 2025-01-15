from nba_api.stats.static import teams

#NBA teams
nba_teams = teams.get_teams()

for team in nba_teams:
    print(f"ID: {team['id']}, Name: {team['full_name']}, Abbreviation: {team['abbreviation']}")