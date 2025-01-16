import streamlit as st
from nba_api.stats.static import teams
from nba_api.stats.endpoints import TeamGameLog
from datetime import datetime
import pandas as pd
import pickle
import requests
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
model_path = './models/nba_model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Initialize session state if not already set
if 'team_game_log_df' not in st.session_state:
    st.session_state['team_game_log_df'] = pd.DataFrame()

if 'head_to_head_teams' not in st.session_state:
    st.session_state['head_to_head_teams'] = {'team_1': None, 'team_2': None}

if 'parlay_teams' not in st.session_state:
    st.session_state['parlay_teams'] = {'team_1': None, 'team_2': None}

# Add custom CSS for styling
st.markdown("""
    <style>
        .main-header {
            font-size: 2.5em;
            font-weight: bold;
            color: #ff6600;
            text-align: center;
            margin-bottom: 20px;
        }
        .input-section {
            background-color: #1e1e1e;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 30px;
        }
        .selectbox, .radio, .button {
            font-size: 1.2em;
            padding: 12px;
            margin-bottom: 15px;
            background-color: #333;
            border: 1px solid #555;
            border-radius: 5px;
            color: white;
        }
        .selectbox:hover, .radio:hover, .button:hover {
            background-color: #ff6600;
        }
        .results-section {
            margin-top: 30px;
            background-color: #2a2a2a;
            border-radius: 10px;
            padding: 20px;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            font-size: 1.2em;
            color: #888;
            font-family: 'Arial', sans-serif;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar for navigation
app_mode = st.sidebar.radio("Select an option", ["Home", "Upcoming & Live Games", "NBA Standings", "Head-to-Head Predictor", "Parlay Creator", "NBA News"], key="app_mode_radio")

# Fetch all NBA teams
nba_teams = teams.get_teams()
team_choices = {team['full_name']: team['id'] for team in nba_teams}
team_choices = dict(sorted(team_choices.items()))

# Main page
def main_page():
    st.title("Welcome to HoopsHub Analytics")

    # App description
    st.subheader("What HoopsHub Analytics Does")
    st.write("""
    **HoopsHub Analytics** is a dynamic sports analytics platform for NBA enthusiasts. 
    The app provides several features including:

    - **Head-to-Head Game Prediction**: Predict the outcome of games between selected NBA teams.
    - **Parlay Creator**: Create your own parlays and calculate potential odds and payouts.
    - **NBA News and Injury Updates**: Get the latest news and injury updates for NBA players.

    Whether you're a fan, a bettor, or just someone looking to understand the game better, HoopsHub Analytics will help you gain insightful data-driven predictions.

    **Please Note**: This app is still in progress and more features will be added soon! 
    Your feedback is highly appreciated!
    """)

    # Add a footer message for users
    st.markdown("---")
    st.write("Developed by **Oulleys** | HoopsHub Analytics - All Rights Reserved")


# Unified function to fetch both injury and news data
def get_nba_data(data_type):
    url = ""
    if data_type == "injuries":
        url = "https://api.sportsdata.io/v3/nba/scores/json/Injuries"
    elif data_type == "news":
        url = "https://api.sportsdata.io/v3/nba/scores/json/News"

    headers = {
        "Ocp-Apim-Subscription-Key": "c00bcbe68d8345feaf2648e2caba5b9c"
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()  # Returns the data based on the type (injury or news)
    else:
        return None

# Fetch injury data
def get_nba_injuries():
    url = "https://api.sportsdata.io/v3/nba/scores/json/Injuries"
    headers = {
        "Ocp-Apim-Subscription-Key": "c00bcbe68d8345feaf2648e2caba5b9c"
    }
    response = requests.get(url, headers=headers)
    print(f"Status Code: {response.status_code}")  # Log the status code
    if response.status_code == 200:
        injury_data = response.json()  # Get the JSON data
        print(injury_data)  # Log the actual injury data for inspection
        return injury_data
    else:
        return None  # API error

# Fetch news data
def get_nba_news():
    return get_nba_data("news")


# Define function to adjust for key player injuries
def adjust_for_injuries(team_name, team_stats):
    injuries = get_nba_injuries()  # Get injury reports using your function
    if injuries:
        for injury in injuries:
            player_name = injury.get('Player', '')
            player_team = injury.get('Team', '')
            injury_status = injury.get('Status', '')

            # Adjust team stats if a key player is injured
            if player_team.lower() == team_name.lower() and injury_status.lower() == 'out':
                # Example of adjusting stats if star players are out
                if player_name in ['LeBron James', 'Giannis Antetokounmpo', 'Kevin Durant']:  # Example of star players
                    team_stats['ortg_max'] *= 0.8  # Decrease Offensive Rating by 20% if star player is out
                    team_stats['drtg_max'] *= 0.9  # Slightly decrease Defensive Rating
                    st.write(
                        f"**Warning:** {player_name} is injured and will not be playing. This may affect {team_name}'s performance.")
    return team_stats


# Function to fetch and display player injuries for selected teams
def display_injury_report(team_name):
    injuries = get_nba_injuries()  # Get injury reports using your function
    if injuries:
        injured_players = []
        for injury in injuries:
            player_name = injury.get('Player', 'Unknown Player')
            player_team = injury.get('Team', 'Unknown Team')
            injury_status = injury.get('Status', 'No status available')

            # Match the team name, ignoring case
            if team_name.lower() in player_team.lower() and injury_status.lower() == 'out':
                injured_players.append(f"{player_name} - {injury_status}")

        if injured_players:
            st.write(f"**Injured Players for {team_name}:**")
            for player in injured_players:
                st.write(f"- {player}")
        else:
            st.write(f"**No injuries reported for {team_name}.**")
    else:
        st.write(f"**No injury data available for {team_name} at the moment.**")

# Head-to-Head Predictor Section
if app_mode == "Head-to-Head Predictor":
    st.markdown('<div class="main-header">NBA Head-to-Head Game Predictor</div>', unsafe_allow_html=True)

    # Team Selection for Head-to-Head
    team_1_name = st.selectbox("Select Team 1", list(team_choices.keys()), key="team_1")
    team_2_name = st.selectbox("Select Team 2", list(team_choices.keys()), key="team_2")

    if team_1_name == team_2_name:
        st.error("Both teams cannot be the same. Please select two different teams.")
        st.stop()

    st.session_state['head_to_head_teams']['team_1'] = team_1_name
    st.session_state['head_to_head_teams']['team_2'] = team_2_name

    team_1_home = st.radio(f"Is {team_1_name} playing at Home or Away?", ['Home', 'Away'], key="team_1_home")
    team_2_home = st.radio(f"Is {team_2_name} playing at Home or Away?", ['Home', 'Away'], key="team_2_home")

    # Validate if both teams are not selected as Home or Away
    if team_1_home == team_2_home:
        st.error("Both teams cannot be at the same location (Home or Away). Please select different options.")
        st.stop()  # Stops execution and forces user to correct the selection

    team_1_home_value = 1 if team_1_home == "Home" else 0
    team_2_home_value = 1 if team_2_home == "Home" else 0

    # Display injury report for both teams
    st.subheader(f"Injury Report for {team_1_name}")
    display_injury_report(team_1_name)

    st.subheader(f"Injury Report for {team_2_name}")
    display_injury_report(team_2_name)

    if st.button("Fetch Head-to-Head Stats", key="fetch_button", help="Click to fetch stats and make predictions",
                 use_container_width=True):
        team_1_id = team_choices[team_1_name]
        team_2_id = team_choices[team_2_name]

        team_1_game_log = TeamGameLog(team_id=team_1_id).get_data_frames()[0]
        team_2_game_log = TeamGameLog(team_id=team_2_id).get_data_frames()[0]

        team_1_avg_stats = {
            'fg%': team_1_game_log['FG_PCT'].mean(),
            '3p%': team_1_game_log['FG3_PCT'].mean(),
            'ft': team_1_game_log['FTM'].mean(),
            'tov%': team_1_game_log['TOV'].mean(),
            'ortg_max': team_1_game_log['PTS'].mean(),
            'drtg_max': team_1_game_log['REB'].mean(),
            'home': team_1_home_value
        }

        team_2_avg_stats = {
            'fg%': team_2_game_log['FG_PCT'].mean(),
            '3p%': team_2_game_log['FG3_PCT'].mean(),
            'ft': team_2_game_log['FTM'].mean(),
            'tov%': team_2_game_log['TOV'].mean(),
            'ortg_max': team_2_game_log['PTS'].mean(),
            'drtg_max': team_2_game_log['REB'].mean(),
            'home': team_2_home_value
        }

        # Adjust for injuries
        team_1_avg_stats = adjust_for_injuries(team_1_name, team_1_avg_stats)
        team_2_avg_stats = adjust_for_injuries(team_2_name, team_2_avg_stats)

        st.write(f"**Team 1 ({team_1_name}) Stats:**", team_1_avg_stats)
        st.write(f"**Team 2 ({team_2_name}) Stats:**", team_2_avg_stats)

        team_1_input = pd.DataFrame([[team_1_avg_stats['fg%'], team_1_avg_stats['3p%'], team_1_avg_stats['ft'],
                                      team_1_avg_stats['tov%'], team_1_avg_stats['ortg_max'],
                                      team_1_avg_stats['drtg_max'], team_1_avg_stats['home']]],
                                    columns=['fg%', '3p%', 'ft', 'tov%', 'ortg_max', 'drtg_max', 'home'])

        team_2_input = pd.DataFrame([[team_2_avg_stats['fg%'], team_2_avg_stats['3p%'], team_2_avg_stats['ft'],
                                      team_2_avg_stats['tov%'], team_2_avg_stats['ortg_max'],
                                      team_2_avg_stats['drtg_max'], team_2_avg_stats['home']]],
                                    columns=['fg%', '3p%', 'ft', 'tov%', 'ortg_max', 'drtg_max', 'home'])

        team_1_proba = model.predict_proba(team_1_input)[0][1]
        team_2_proba = model.predict_proba(team_2_input)[0][1]

        st.write(f"**Team 1 ({team_1_name}) Win Probability:** {team_1_proba:.2f}")
        st.write(f"**Team 2 ({team_2_name}) Win Probability:** {team_2_proba:.2f}")

        if team_1_proba > team_2_proba:
            st.success(f"Prediction: {team_1_name} is more likely to win!")
        elif team_2_proba > team_1_proba:
            st.success(f"Prediction: {team_2_name} is more likely to win!")
        else:
            st.warning("Prediction: It's too close to call. Both teams are evenly matched!")

# Parlay Creator Section
elif app_mode == "Parlay Creator":
    st.title("NBA In-Game Parlay Creator")

    available_teams = [
        "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets", "Chicago Bulls",
        "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets", "Detroit Pistons", "Golden State Warriors",
        "Houston Rockets", "Indiana Pacers", "Los Angeles Clippers", "Los Angeles Lakers", "Memphis Grizzlies",
        "Miami Heat", "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
        "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns", "Portland Trail Blazers",
        "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors", "Utah Jazz", "Washington Wizards"
    ]

    # Team selection for Parlay
    team_1 = st.selectbox("Select Team 1 for Parlay", available_teams,
                          index=available_teams.index(st.session_state['parlay_teams']['team_1']) if
                          st.session_state['parlay_teams']['team_1'] else 0)
    team_2 = st.selectbox("Select Team 2 for Parlay", available_teams,
                          index=available_teams.index(st.session_state['parlay_teams']['team_2']) if
                          st.session_state['parlay_teams']['team_2'] else 0)

    # Update session state when team selection changes
    st.session_state['parlay_teams']['team_1'] = team_1
    st.session_state['parlay_teams']['team_2'] = team_2

    # Moneyline winner selection
    moneyline_winner = st.radio("Pick the team to win the game", [team_1, team_2], key="moneyline_winner")

    # Spread selection (User will select spread for each team)
    spread_team_1 = st.number_input(f"Enter spread for {team_1}", value=6.0, key=f"spread_team_1")

    # Automatically calculate spread for Team 2 as the opposite of Team 1's spread
    spread_team_2 = -spread_team_1  # Opposite spread for Team 2

    # Display spreads for both teams
    st.write(f"Spread for {team_1}: {spread_team_1}")
    st.write(f"Spread for {team_2}: {spread_team_2} (Automatically adjusted)")

    # Parlay input fields for odds
    moneyline_team_1 = st.number_input(f"Moneyline Odds for {team_1}", value=110.0, key=f"moneyline_team_1")
    moneyline_team_2 = st.number_input(f"Moneyline Odds for {team_2}", value=-130.0, key=f"moneyline_team_2")
    spread_team_1_odds = st.number_input(f"Odds for {team_1} to cover spread", value=1.5, key=f"spread_team_1_odds")
    spread_team_2_odds = st.number_input(f"Odds for {team_2} to cover spread", value=1.5, key=f"spread_team_2_odds")

    over_under = st.radio("Over/Under", ("Over", "Under"), key=f"over_under_{team_1}_{team_2}")
    total_points = st.number_input("Enter Total Points", value=220, key="total_points")
    over_under_odds = st.number_input(f"Odds for {over_under} {total_points} points", value=1.8, key="over_under_odds")

    # Total number of odds
    odds_list = [moneyline_team_1, moneyline_team_2, spread_team_1_odds, spread_team_2_odds]


    # Function to calculate combined parlay odds
    def calculate_parlay_odds(odds_list):
        combined_odds = 1
        for odds in odds_list:
            if odds > 0:  # Positive odds
                combined_odds *= (1 + (odds / 100))  # Convert to decimal odds
            else:  # Negative odds
                combined_odds *= (1 + (100 / abs(odds)))  # Convert negative odds to decimal odds
        return (combined_odds - 1) * 100  # Convert decimal odds back to American odds


    # Function to calculate payout
    def calculate_payout(odds, stake):
        if odds > 0:  # Positive odds
            payout = stake * (1 + (odds / 100))  # Profit from positive odds
        else:  # Negative odds
            payout = stake * (1 + (100 / abs(odds)))  # Profit from negative odds
        return payout


    # User input for stake amount
    stake_amount = st.number_input("Enter your stake amount (in CAD)", value=10.0)

    # Calculate parlay odds and potential payout
    parlay_odds = calculate_parlay_odds(odds_list)
    potential_payout = calculate_payout(parlay_odds, stake_amount)

    # Styling section
    st.markdown("---")  # Horizontal line separator for neatness

    # Display the results with enhanced styling
    if st.button("Calculate Parlay Odds"):
        # Display Parlay Odds
        st.markdown(f"### Parlay Odds for {team_1} vs {team_2}:")
        st.markdown(f"**+{round(parlay_odds)}**")  # Format the odds as a positive number

        # Display Potential Payout
        st.markdown(f"### Potential Payout for ${stake_amount} CAD:")
        st.markdown(f"**${round(potential_payout, 2)} CAD**")  # Display the payout with two decimal places

        # Optional: Add some final message or good luck message
        st.markdown("#### Good luck with your bet!")

# NBA News Section
elif app_mode == "NBA News":
    st.title("Latest NBA News and Injury Updates")

    if st.button("Refresh News"):
        st.session_state['refresh_news_key'] = not st.session_state.get('refresh_news_key', False)

    news_articles = get_nba_data("news")

    if news_articles:
        st.subheader("Latest NBA News")
        for article in news_articles:
            title = article.get('Title', 'No Title Available')
            content = article.get('Content', 'No content available.')
            url = article.get('Url', '#')
            date_str = article.get('Updated', 'No Date Available')  # Get the raw date string

            # Convert the raw date string to a more readable format
            try:
                date_obj = datetime.fromisoformat(date_str.replace('T', ' '))  # Parse ISO format date
                formatted_date = date_obj.strftime("%B %d, %Y at %I:%M %p")  # e.g., January 14, 2025 at 12:00 AM
            except:
                formatted_date = date_str

            # Display the article title with bold and larger font
            st.markdown(f"### **{title}**")

            # Display the formatted publication date
            st.markdown(f"*Published on: {formatted_date}*")

            # Display a shortened version of the content (if available)
            summary = content[:250] + "..." if content != 'No content available.' else "Summary not provided for this article."
            st.write(f"**Summary**: {summary}")  # Show the first 250 characters of the content

            # Display the "Read more" link as a styled button with a hover effect
            st.markdown(f"[**Read more**]({url})", unsafe_allow_html=True)

            # Optional separator
            st.markdown("---")
    else:
        st.write("No news available at the moment.")

#NBA Standings

# Function to fetch NBA standings from SportsDataIO API
def fetch_nba_standings(season, api_key):
    url = f"https://api.sportsdata.io/v3/nba/scores/json/Standings/{season}"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to fetch standings: {response.status_code}")
        return None

# Process the API response into a DataFrame
def process_standings_data(data):
    standings = []
    for team in data:
        standings.append({
            "Conference": team["Conference"],
            "Team": team["Name"],
            "Wins": team["Wins"],
            "Losses": team["Losses"],
            "Win Percentage": f"{team['Percentage'] * 100:.2f}%",
            "Division Rank": team["DivisionRank"]
        })
    return pd.DataFrame(standings)

# Define NBA Standings feature
def nba_standings():
    st.title("🏀 NBA Standings")

    current_year = datetime.now().year
    season = st.selectbox("Select Season:", [str(current_year), "2024"])

    api_key = "c00bcbe68d8345feaf2648e2caba5b9c"

    # Add a refresh button to update the standings
    if st.button("Refresh Standings"):
        st.session_state['refresh_key'] = not st.session_state.get('refresh_key', False)

    standings_data = fetch_nba_standings(season, api_key)
    if standings_data:
        standings_df = process_standings_data(standings_data)

        # Split data by conference
        st.subheader("Eastern Conference")
        east_df = standings_df[standings_df["Conference"] == "Eastern"]
        st.table(east_df)

        st.subheader("Western Conference")
        west_df = standings_df[standings_df["Conference"] == "Western"]
        st.table(west_df)

#Upcoming/Live games

def fetch_games_by_date(date):
    url = f"https://api.sportsdata.io/v3/nba/scores/json/GamesByDate/{date}"
    headers = {"Ocp-Apim-Subscription-Key": "c00bcbe68d8345feaf2648e2caba5b9c"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to fetch games: {response.status_code}")
        return None

# Function to process games data
def process_games_data(games):
    upcoming_games = []
    live_games = []
    for game in games:
        game_time_raw = game.get('DateTime', '')
        status = game.get('Status', '')
        home_team = game.get('HomeTeam', 'Unknown')
        away_team = game.get('AwayTeam', 'Unknown')

        # Format the game time into a more readable format
        try:
            game_time = datetime.fromisoformat(game_time_raw).strftime('%I:%M %p')  # "07:00 PM"
        except ValueError:
            game_time = "TBD"

        if status == 'InProgress':
            live_games.append({
                "Home Team": home_team,
                "Away Team": away_team,
                "Status": "Live",
                "Score": f"{game.get('HomeTeamScore', 0)} - {game.get('AwayTeamScore', 0)}"
            })
        else:
            upcoming_games.append({
                "Home Team": home_team,
                "Away Team": away_team,
                "Status": "Scheduled",
                "Time": game_time
            })

    return pd.DataFrame(upcoming_games), pd.DataFrame(live_games)


#App modes

if app_mode == "Home":
    main_page()
elif app_mode == "Head-to-Head Predictor":
    # Call the function for Head-to-Head Predictor section
    pass
elif app_mode == "Parlay Creator":
    # Call the function for Parlay Creator section
    pass
elif app_mode == "NBA News":
    # Call the function for NBA News section
    pass
elif app_mode == "NBA Standings":
    nba_standings()

elif app_mode == "Upcoming & Live Games":
    st.title("🏀 Upcoming Games and Live Games")

    date = datetime.now().date()

    # Fetch games automatically when the section is loaded
    if "games_data" not in st.session_state:
        games = fetch_games_by_date(date)
        if games:
            st.session_state['games_data'] = games

    games = st.session_state.get('games_data', [])
    if games:
        upcoming_df, live_df = process_games_data(games)

        # Display live games
        if not live_df.empty:
            st.subheader("🔴 Live Games")
            st.dataframe(live_df)
        else:
            st.write("No live games at the moment.")

        # Display upcoming games
        if not upcoming_df.empty:
            st.subheader("📅 Upcoming Games")
            st.dataframe(upcoming_df)  # For an interactive table
        else:
            st.write("No upcoming games for the selected date.")
    else:
        st.write("Failed to load game data.")