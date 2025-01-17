from nba_api.stats.endpoints import TeamGameLog
import pandas as pd

# Example: Atlanta Hawks)
team_id = 1610612737

team_game_log = TeamGameLog(team_id=team_id)

team_game_log_df = team_game_log.get_data_frames()[0]

print(team_game_log_df.head())