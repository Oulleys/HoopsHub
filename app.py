import streamlit as st
from nba_api.stats.static import teams
from nba_api.stats.endpoints import TeamGameLog
from datetime import datetime
import pandas as pd
import pickle
import requests
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
            font-size: 1em;
            color: #888;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar for navigation
app_mode = st.sidebar.radio("Select an option", ["Head-to-Head Predictor", "Parlay Creator", "NBA News"], key="app_mode_radio")


# Fetch all NBA teams
nba_teams = teams.get_teams()
team_choices = {team['full_name']: team['id'] for team in nba_teams}

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

    if st.button("Fetch Head-to-Head Stats", key="fetch_button", help="Click to fetch stats and make predictions", use_container_width=True):
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

        st.write(f"**Team 1 ({team_1_name}) Stats:**", team_1_avg_stats)
        st.write(f"**Team 2 ({team_2_name}) Stats:**", team_2_avg_stats)

        team_1_input = pd.DataFrame([[team_1_avg_stats['fg%'], team_1_avg_stats['3p%'], team_1_avg_stats['ft'],
                                      team_1_avg_stats['tov%'], team_1_avg_stats['ortg_max'],
                                      team_1_avg_stats['drtg_max'], team_1_avg_stats['home']]], columns=['fg%', '3p%', 'ft', 'tov%', 'ortg_max', 'drtg_max', 'home'])

        team_2_input = pd.DataFrame([[team_2_avg_stats['fg%'], team_2_avg_stats['3p%'], team_2_avg_stats['ft'],
                                      team_2_avg_stats['tov%'], team_2_avg_stats['ortg_max'],
                                      team_2_avg_stats['drtg_max'], team_2_avg_stats['home']]], columns=['fg%', '3p%', 'ft', 'tov%', 'ortg_max', 'drtg_max', 'home'])

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

    # Define function to get NBA news
    def get_nba_news():
        url = "https://api.sportsdata.io/v3/nba/scores/json/News"
        headers = {
            "Ocp-Apim-Subscription-Key": "c00bcbe68d8345feaf2648e2caba5b9c"  # Replace with your actual API key
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()  # Assuming 'response.json()' contains the news
        else:
            return None

    # Define function to get NBA injury reports
    def get_nba_injuries():
        url = "https://api.sportsdata.io/v3/nba/scores/json/Injuries"
        headers = {
            "Ocp-Apim-Subscription-Key": "c00bcbe68d8345feaf2648e2caba5b9c"  # Replace with your actual API key
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()  # Assuming 'response.json()' contains injury reports
        else:
            return None

    # Fetch live NBA news
    news_articles = get_nba_news()

    if news_articles:
        st.subheader("Latest NBA News")
        for article in news_articles:
            title = article.get('Title', 'No Title Available')
            content = article.get('Content', 'No content available.')
            url = article.get('Url', '#')
            date_str = article.get('Updated', 'No Date Available')  # Get the raw date string

            # Convert the raw date string to a more readable format
            try:
                date_obj = datetime.fromisoformat(date_str.replace('T', ' '))
                formatted_date = date_obj.strftime("%B %d, %Y at %I:%M %p")  # e.g., January 14, 2025 at 12:00 AM
            except:
                formatted_date = date_str

            # Display the article title with bold and larger font
            st.markdown(f"### **{title}**")

            # Display the formatted publication date
            st.markdown(f"*Published on: {formatted_date}*")

            # Display a shortened version of the content (if available)
            summary = content[
                      :250] + "..." if content != 'No content available.' else "Summary not provided for this article."
            st.write(f"**Summary**: {summary}")  # Show the first 250 characters of the content

            # Display the "Read more" link as a styled button with a hover effect
            st.markdown(f"[**Read more**]({url})", unsafe_allow_html=True)

            # Optional separator
            st.markdown("---")
    else:
        st.write("No news available at the moment.")