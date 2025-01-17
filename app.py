import streamlit as st
from nba_api.stats.static import teams
from nba_api.stats.endpoints import TeamGameLog
from datetime import datetime
from tzlocal import get_localzone
from sklearn.ensemble import RandomForestClassifier

import streamlit_authenticator as stauth
import smtplib
import random
import sqlite3
import bcrypt
import pandas as pd
import pickle
import pytz
import requests
import time
import numpy as np
import re

#:)

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

#Helper functions

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


# Example session state to track login status
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if "app_mode" not in st.session_state:
    st.session_state["app_mode"] = "Home"



# Sidebar for navigation
app_mode = st.sidebar.radio(
    "Select an option",
    ["Home", "Sign up/Login", "Upcoming & Live Games", "NBA Standings", "Head-to-Head Predictor", "Parlay Creator", "NBA News"],
    index=0,
    key="app_mode_radio"
)

st.session_state["app_mode"] = app_mode

# Initialize session state variables
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None


# Sidebar Log In and Log Out Buttons
if st.sidebar.button("Log In", key="sidebar_login_button"):
    st.query_params(app_mode="Sign up/Login")
    st.sidebar.info("Redirecting to Sign Up/Login...")

if st.sidebar.button("Log Out", key="sidebar_logout_button"):
    st.session_state["logged_in"] = False
    st.session_state["username"] = None
    st.sidebar.info("Logged out successfully!")

conn = sqlite3.connect("hoopshub_users.db") # Database setup
c = conn.cursor()

# Create tables if they don't exist
c.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    hashed_password TEXT,
    email TEXT UNIQUE
)
""")
conn.commit()

c.execute("""
CREATE TABLE IF NOT EXISTS bet_slips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    bet_details TEXT,
    result TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (username) REFERENCES users(username)
)
""")


conn.commit()

# Helper functions
def hash_password(password):
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))

def signup(username, password, email):
    hashed_password = hash_password(password)
    try:
        c.execute("INSERT INTO users (username, hashed_password, email) VALUES (?, ?, ?)", (username, hashed_password, email))
        conn.commit()
        return True
    except sqlite3.IntegrityError as e:
        if "UNIQUE constraint failed: users.email" in str(e):
            return "email_exists"
        return "username_exists"

def login(username, password):
    c.execute("SELECT hashed_password FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    if user and check_password(password, user[0]):
        return True
    return False

def save_parlay_bet(username, bet_details, amount, odds, payout):
    c.execute("""
    INSERT INTO bet_slips (username, bet_details, result, created_at)
    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
    """, (username, f"{bet_details} | Bet: {amount} | Odds: {odds} | Payout: {payout}", "Pending"))
    conn.commit()

def fetch_user_bets(username):
    c.execute("SELECT bet_details, result, created_at FROM bet_slips WHERE username = ?", (username,))
    return c.fetchall()

def is_valid_email(email):
    allowed_domains = ["gmail.com", "outlook.com", "yahoo.com", "hotmail.com"]
    email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    if re.match(email_regex, email):
        # Extract the domain from the email
        domain = email.split("@")[-1]
        return domain in allowed_domains
    return False

def is_strong_password(password):
    # Regex for validating password
    password_regex = r'^(?=.*[A-Za-z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'
    return re.match(password_regex, password) is not None


def is_valid_username(username):
    return re.match(r'^[A-Za-z0-9]{3,}$', username) is not None

#Sign up/Login
if app_mode == "Sign up/Login":
    st.title("Sign Up or Log In")

    # Check if the user is logged in
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        st.session_state["username"] = None

    if not st.session_state.logged_in:
        choice = st.radio("Choose an option:", ["Login", "Sign Up"])

        if choice == "Sign Up":
            st.subheader("Create an Account")
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")

            if st.button("Sign Up"):
                if not is_valid_username(username):
                    st.error(
                        "Invalid username. Must be at least 3 characters long and contain only letters and numbers.")
                elif not is_valid_email(email):
                    st.error("Invalid email address. Only Gmail, Outlook, Yahoo, and Hotmail are allowed.")
                elif not is_strong_password(password):
                    st.error(
                        "Weak password. Must be at least 8 characters long, include a number, and a special character.")
                else:
                    result = signup(username, password, email)
                    if result == True:
                        st.success("Account created! Please log in.")
                    elif result == "email_exists":
                        st.error("Email already registered. Please use a different email.")
                    elif result == "username_exists":
                        st.error("Username already exists. Please choose a different username.")

        elif choice == "Login":
            st.subheader("Log In")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Log In"):
                if login(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success(f"Welcome, back {username}!")
                else:
                    st.error("Invalid username or password.")
    else:
        st.sidebar.write(f"Logged in as: **{st.session_state.username}**")

    #Additional bet slip functionality if logged in
    st.subheader("Your Bet Slips")
    c.execute("SELECT bet_details, result, created_at FROM bet_slips WHERE username = ?",
                  (st.session_state.username,))
    bet_slips = c.fetchall()

    if bet_slips:
        for bet in bet_slips:
                st.write(f"**Bet Details:** {bet[0]}")
                st.write(f"**Result:** {bet[1] if bet[1] else 'Pending'}")
                st.write(f"**Placed On:** {bet[2]}")
                st.markdown("---")
        else:
            st.write("No bet slips found.")

        # Add a new bet slip
        st.subheader("Place a New Bet")
        bet_details = st.text_area("Bet Details")
        if st.button("Save Bet"):
            c.execute("INSERT INTO bet_slips (username, bet_details, result) VALUES (?, ?, ?)",
                      (st.session_state.username, bet_details, None))
            conn.commit()
            st.success("Bet slip saved!")

# Fetch all NBA teams
nba_teams = teams.get_teams()
team_choices = {team['full_name']: team['id'] for team in nba_teams}
team_choices = dict(sorted(team_choices.items()))

#main page
def main_page():
    st.title("🏀 Welcome to HoopsHub Analytics")

    # App description
    st.subheader("Explore HoopsHub Analytics")
    st.markdown("""
    **HoopsHub Analytics** is your comprehensive platform for NBA analytics, delivering tools and insights for fans, bettors, and analysts. Here's what you can do:
    """)

    st.markdown("""
    - **🏀 Upcoming & Live Games**: Check out upcoming games and stay updated with live game statistics.
    - **📊 NBA Standings**: Monitor team standings and track playoff positions in real time.
    - **💡 Head-to-Head Game Predictor**: Use analytics to predict the outcomes of games between NBA teams.
    - **💰 Parlay Creator**: Build custom parlays, calculate odds, and project payouts for your bets.
    - **📰 NBA News and Injury Updates**: Stay informed with the latest news, player injuries, and lineup changes.
    """)

    st.markdown("""
    Whether you're here to enhance your betting strategy or gain a deeper understanding of the game, **HoopsHub Analytics** has you covered with data-driven insights and cutting-edge tools.
    """)

    st.markdown("---")

    st.write("""
    Developed by **Oulleys**  
    © 2025 **HoopsHub Analytics** - All Rights Reserved  
    *Bringing the game closer to you.*
    """)

    if st.button("📩 Contact Us"):
        st.write("For feedback or inquiries, please reach out at: **vulidslol@gmail.com**")

#Upcoming/Live games

def fetch_games_by_date(date):
    url = f"https://api.sportsdata.io/v3/nba/scores/json/GamesByDate/{date}"
    headers = {"Ocp-Apim-Subscription-Key": "c00bcbe68d8345feaf2648e2caba5b9c"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to fetch games: {response.status_code}")
        return []

# Function to process games data
def process_games_data(games):
    upcoming_games = []
    live_games = []
    for game in games:
        game_time_raw = game.get('DateTime', '')
        status = game.get('Status', '')
        home_team = game.get('HomeTeam', 'Unknown')
        away_team = game.get('AwayTeam', 'Unknown')

        # Format the game time into a readable format
        try:
            game_time = datetime.fromisoformat(game_time_raw).strftime('%I:%M %p')  # "07:00 PM"
        except ValueError:
            game_time = "TBD"

        # Check if the game is live or upcoming
        if status in ['InProgress', '1st Quarter', '2nd Quarter', 'HalfTime', '3rd Quarter', '4th Quarter']:
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


# Head-to-Head Predictor Section
if app_mode == "Head-to-Head Predictor":
    st.markdown('<div class="main-header">🏀 <strong>NBA Head-to-Head Game Predictor</strong></div>',
                unsafe_allow_html=True)

    # Team Selection
    st.markdown('<h4>Select Teams:</h4>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        team_1_name = st.selectbox("Team 1", list(team_choices.keys()), key="team_1")
    with col2:
        # Exclude Team 1 from Team 2 options
        available_teams_for_team_2 = [team for team in list(team_choices.keys()) if team != team_1_name]
        team_2_name = st.selectbox("Team 2", available_teams_for_team_2, key="team_2")

    st.session_state['head_to_head_teams'] = {'team_1': team_1_name, 'team_2': team_2_name}

    # Home/Away Settings
    st.markdown('<h4>Home/Away Settings:</h4>', unsafe_allow_html=True)

    # Columns for layout
    col_home_away, col_display_status = st.columns([3, 1])

    # Radio button for selecting home/away
    with col_home_away:
        team_1_home = st.radio(
            f"Where is {team_1_name} playing?",
            options=['Home', 'Away'],
            key="team_1_home",
            help="Select whether Team 1 is playing at home or away."
        )

    # Dynamically display the opposite setting for Team 2
    with col_display_status:
        team_2_home = "Away" if team_1_home == "Home" else "Home"
        st.markdown(
            f"<div style='text-align:center; font-size:16px; font-weight:bold;'>"
            f"<span style='color:#FF5733;'>{team_2_name}</span> is playing "
            f"<span style='color:#28A745;'>{team_2_home}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    # Fetch Injury Reports
    st.markdown('<h4>Injury Reports:</h4>', unsafe_allow_html=True)


    def display_injury_safe(team_name):
        """Display injury reports and handle errors."""
        try:
            display_injury_report(team_name)
        except Exception as e:
            st.error(f"Could not fetch injury report for {team_name}: {e}")


    st.markdown(f"<div><strong>{team_1_name}:</strong></div>", unsafe_allow_html=True)
    display_injury_safe(team_1_name)

    st.markdown(f"<div><strong>{team_2_name}:</strong></div>", unsafe_allow_html=True)
    display_injury_safe(team_2_name)

    # Fetch Stats Button
    st.markdown('<div style="text-align:center; margin-top:20px;">', unsafe_allow_html=True)
    if st.button("📊 Fetch Head-to-Head Stats", key="fetch_button", help="Click to fetch stats and make predictions"):
        try:
            team_1_id = team_choices[team_1_name]
            team_2_id = team_choices[team_2_name]


            # Retrieve game logs for both teams
            def fetch_game_logs(team_id):
                """Fetch game logs for a team."""
                return TeamGameLog(team_id=team_id).get_data_frames()[0]


            team_1_game_log = fetch_game_logs(team_1_id)
            team_2_game_log = fetch_game_logs(team_2_id)


            # Calculate average stats for both teams
            def calculate_avg_stats(game_log, home_status):
                """Calculate average statistics."""
                return {
                    'fg%': game_log['FG_PCT'].mean(),
                    '3p%': game_log['FG3_PCT'].mean(),
                    'ft': game_log['FTM'].mean(),
                    'tov%': game_log['TOV'].mean(),
                    'ortg_max': game_log['PTS'].mean(),
                    'drtg_max': game_log['REB'].mean(),
                    'home': 1 if home_status == "Home" else 0
                }


            team_1_avg_stats = calculate_avg_stats(team_1_game_log, team_1_home)
            team_2_avg_stats = calculate_avg_stats(team_2_game_log, team_2_home)

            # Adjust stats for injuries
            team_1_avg_stats = adjust_for_injuries(team_1_name, team_1_avg_stats)
            team_2_avg_stats = adjust_for_injuries(team_2_name, team_2_avg_stats)

            # Display stats
            st.write(f"**Team 1 ({team_1_name}) Stats:**", team_1_avg_stats)
            st.write(f"**Team 2 ({team_2_name}) Stats:**", team_2_avg_stats)


            # Prepare input for prediction model
            def prepare_input(avg_stats):
                return pd.DataFrame([[avg_stats['fg%'], avg_stats['3p%'], avg_stats['ft'], avg_stats['tov%'],
                                      avg_stats['ortg_max'], avg_stats['drtg_max'], avg_stats['home']]],
                                    columns=['fg%', '3p%', 'ft', 'tov%', 'ortg_max', 'drtg_max', 'home'])


            team_1_input = prepare_input(team_1_avg_stats)
            team_2_input = prepare_input(team_2_avg_stats)

            # Predict win probabilities
            team_1_proba = model.predict_proba(team_1_input)[0][1]
            team_2_proba = model.predict_proba(team_2_input)[0][1]

            # Display predictions
            st.write(f"**Team 1 ({team_1_name}) Win Probability:** {team_1_proba:.2f}")
            st.write(f"**Team 2 ({team_2_name}) Win Probability:** {team_2_proba:.2f}")

            if team_1_proba > team_2_proba:
                st.success(f"Prediction: {team_1_name} is more likely to win!")
            elif team_2_proba > team_1_proba:
                st.success(f"Prediction: {team_2_name} is more likely to win!")
            else:
                st.warning("Prediction: It's too close to call. Both teams are evenly matched!")

        except Exception as e:
            st.error(f"An error occurred during data processing: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# Parlay Creator Section
elif app_mode == "Parlay Creator":
    st.title("🏀 NBA In-Game Parlay Creator")

    team_logos = {
        "Atlanta Hawks": "https://a.espncdn.com/i/teamlogos/nba/500/atl.png",
        "Boston Celtics": "https://a.espncdn.com/i/teamlogos/nba/500/bos.png",
        "Brooklyn Nets": "https://a.espncdn.com/i/teamlogos/nba/500/bkn.png",
        "Charlotte Hornets": "https://a.espncdn.com/i/teamlogos/nba/500/cha.png",
        "Chicago Bulls": "https://a.espncdn.com/i/teamlogos/nba/500/chi.png",
        "Cleveland Cavaliers": "https://a.espncdn.com/i/teamlogos/nba/500/cle.png",
        "Dallas Mavericks": "https://a.espncdn.com/i/teamlogos/nba/500/dal.png",
        "Denver Nuggets": "https://a.espncdn.com/i/teamlogos/nba/500/den.png",
        "Detroit Pistons": "https://a.espncdn.com/i/teamlogos/nba/500/det.png",
        "Golden State Warriors": "https://a.espncdn.com/i/teamlogos/nba/500/gsw.png",
        "Houston Rockets": "https://a.espncdn.com/i/teamlogos/nba/500/hou.png",
        "Indiana Pacers": "https://a.espncdn.com/i/teamlogos/nba/500/ind.png",
        "Los Angeles Clippers": "https://a.espncdn.com/i/teamlogos/nba/500/lac.png",
        "Los Angeles Lakers": "https://a.espncdn.com/i/teamlogos/nba/500/lal.png",
        "Memphis Grizzlies": "https://a.espncdn.com/i/teamlogos/nba/500/mem.png",
        "Miami Heat": "https://a.espncdn.com/i/teamlogos/nba/500/mia.png",
        "Milwaukee Bucks": "https://a.espncdn.com/i/teamlogos/nba/500/mil.png",
        "Minnesota Timberwolves": "https://a.espncdn.com/i/teamlogos/nba/500/min.png",
        "New Orleans Pelicans": "https://a.espncdn.com/i/teamlogos/nba/500/no.png",
        "New York Knicks": "https://a.espncdn.com/i/teamlogos/nba/500/nyk.png",
        "Oklahoma City Thunder": "https://a.espncdn.com/i/teamlogos/nba/500/okc.png",
        "Orlando Magic": "https://a.espncdn.com/i/teamlogos/nba/500/orl.png",
        "Philadelphia 76ers": "https://a.espncdn.com/i/teamlogos/nba/500/phi.png",
        "Phoenix Suns": "https://a.espncdn.com/i/teamlogos/nba/500/phx.png",
        "Portland Trail Blazers": "https://a.espncdn.com/i/teamlogos/nba/500/por.png",
        "Sacramento Kings": "https://a.espncdn.com/i/teamlogos/nba/500/sac.png",
        "San Antonio Spurs": "https://a.espncdn.com/i/teamlogos/nba/500/sas.png",
        "Toronto Raptors": "https://a.espncdn.com/i/teamlogos/nba/500/tor.png",
        "Utah Jazz": "https://a.espncdn.com/i/teamlogos/nba/500/uta.png",
        "Washington Wizards": "https://a.espncdn.com/i/teamlogos/nba/500/was.png"
    }

    placeholder_logo = "https://via.placeholder.com/50?text=No+Logo"


    def normalize_team_name(team_name):
        return team_name.strip().title()

    # Fetch live games data for today's date
    today = datetime.now().strftime("%Y-%m-%d")
    games_data = fetch_games_by_date(today)
    upcoming_games, live_games = process_games_data(games_data)

    # Convert live games to a dictionary for quick lookup
    live_scores_dict = {
        f"{game['Home Team']} vs {game['Away Team']}": game["Score"]
        for _, game in live_games.iterrows()
    }


    @st.cache_data
    def fetch_odds_from_oddsapi():
        api_key = "86b5432485ce9d1cda72802f8cb8c17f"
        url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
        params = {
            "apiKey": api_key,
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american"
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            odds_data = response.json()
            for game in odds_data:
                game["start_time"] = game.get("commence_time", "N/A")  # Add start time
                game["is_live"] = game.get("live", False)  # Add live status
            return odds_data
        else:
            st.error(f"Failed to fetch odds: {response.status_code} - {response.json().get('message', '')}")
            return None


    def parse_odds_data(odds_data):
        games = []
        for game in odds_data:
            game_data = {
                "game_id": game["id"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "moneyline_home": None,
                "moneyline_away": None,
                "spread_home": None,
                "spread_away": None,
                "spread_home_odds": None,
                "spread_away_odds": None,
                "over_under": None,
                "over_odds": None,
                "under_odds": None,
                "start_time": game.get("commence_time", "N/A"),  # Add start time
                "is_live": game.get("live", False)  # Add live status
            }
            for bookmaker in game["bookmakers"]:
                if bookmaker["key"] == "draftkings":  # Use DraftKings-specific odds
                    for market in bookmaker["markets"]:
                        if market["key"] == "h2h":  # Moneyline
                            game_data["moneyline_home"] = market["outcomes"][0]["price"]
                            game_data["moneyline_away"] = market["outcomes"][1]["price"]
                        elif market["key"] == "spreads":  # Spread
                            game_data["spread_home"] = market["outcomes"][0]["point"]
                            game_data["spread_away"] = market["outcomes"][1]["point"]
                            game_data["spread_home_odds"] = market["outcomes"][0]["price"]  # Spread home odds
                            game_data["spread_away_odds"] = market["outcomes"][1]["price"]  # Spread away odds
                        elif market["key"] == "totals":  # Over/Under
                            game_data["over_under"] = market["outcomes"][0]["point"]
                            game_data["over_odds"] = market["outcomes"][0]["price"]  # Odds for Over
                            game_data["under_odds"] = market["outcomes"][1]["price"]  # Odds for Under
            games.append(game_data)
        return games


    def calculate_parlay_odds(odds_list):
        decimal_odds_list = []
        for odds in odds_list:
            if odds > 0:
                decimal_odds = 1 + (odds / 100)
            else:
                decimal_odds = 1 + (100 / abs(odds))
            decimal_odds_list.append(decimal_odds)

        combined_decimal_odds = 1
        for decimal_odd in decimal_odds_list:
            combined_decimal_odds *= decimal_odd

        if combined_decimal_odds >= 2:
            parlay_odds = (combined_decimal_odds - 1) * 100
        else:
            parlay_odds = -100 / (combined_decimal_odds - 1)

        return round(parlay_odds)


    def calculate_payout(parlay_odds, stake):
        if parlay_odds > 0:
            return round(stake * (1 + (parlay_odds / 100)), 2)
        else:
            return round(stake * (1 + (100 / abs(parlay_odds))), 2)


    if "bet_slip" not in st.session_state:
        st.session_state.bet_slip = []


    def add_bet_to_slip(game_name, bet_type, team_or_pick, odds):
        for bet in st.session_state.bet_slip:
            if bet["Game"] == game_name:
                for sub_bet in bet["Bets"]:
                    # Restrict betting on both teams in the same market
                    if sub_bet["Type"] == bet_type and sub_bet["Team/Pick"] != team_or_pick:
                        st.warning(f"You cannot bet on both teams for {bet_type} in the same game.")
                        return  # Stop adding the bet if there's a conflict

                    # Restrict placing the same bet repeatedly
                    if sub_bet["Type"] == bet_type and sub_bet["Team/Pick"] == team_or_pick and sub_bet["Odds"] == odds:
                        st.warning(f"You have already placed a {bet_type} bet on {team_or_pick} with odds {odds}.")
                        return  # Stop adding duplicate bet

                # Add the new bet to the existing game (no conflicts or duplicates found)
                bet["Bets"].append({"Type": bet_type, "Team/Pick": team_or_pick, "Odds": odds})
                return

        # If the game is not yet in the bet slip, add it as a new entry
        st.session_state.bet_slip.append({
            "Game": game_name,
            "Bets": [{"Type": bet_type, "Team/Pick": team_or_pick, "Odds": odds}]
        })


    def remove_bet_from_slip(game_name, bet_index):
        for bet in st.session_state.bet_slip:
            if bet["Game"] == game_name:
                del bet["Bets"][bet_index]
                # Check if there are any bets left in this game, if not, remove the game entirely
                if len(bet["Bets"]) == 0:
                    st.session_state.bet_slip.remove(bet)
                return


    odds_data = fetch_odds_from_oddsapi()
    if not odds_data:
        st.error("No games available.")
        st.stop()

    games = parse_odds_data(odds_data)

    if games:
        st.markdown("### Upcoming Games with Live Scores")
        for game in games:
            with st.container():
                st.markdown("---")
                col_logo1, col_text, col_logo2, col_status = st.columns([1, 4, 1, 2])
                home_team = normalize_team_name(game["home_team"])
                away_team = normalize_team_name(game["away_team"])
                home_logo = team_logos.get(home_team, placeholder_logo)
                away_logo = team_logos.get(away_team, placeholder_logo)

                # Fetch the live score for the game, if available
                game_key = f"{home_team} vs {away_team}"
                live_score = live_scores_dict.get(game_key, "Not Started")

                with col_logo1:
                    st.image(home_logo, width=50)
                with col_text:
                    st.markdown(f"### {home_team} vs {away_team}")
                with col_logo2:
                    st.image(away_logo, width=50)
                with col_status:
                    start_time = game["start_time"]
                    live_status = "Live Now" if game["is_live"] else f"Starts at {start_time}"
                    st.markdown(f"🕒 **{live_status}**")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**Moneyline**")
                    if game.get("moneyline_home"):
                        if st.button(f"{home_team}: {game['moneyline_home']}", key=f"ml_home_{game['game_id']}"):
                            add_bet_to_slip(f"{home_team} vs {away_team}", "Moneyline", home_team,
                                            game["moneyline_home"])

                    if game.get("moneyline_away"):
                        if st.button(f"{away_team}: {game['moneyline_away']}", key=f"ml_away_{game['game_id']}"):
                            add_bet_to_slip(f"{home_team} vs {away_team}", "Moneyline", away_team,
                                            game["moneyline_away"])

                with col2:
                    st.markdown("**Spread**")
                    if game.get("spread_home"):
                        if st.button(f"{home_team}: {game['spread_home']} ({game['spread_home_odds']})",
                                     key=f"spread_home_{game['game_id']}"):
                            add_bet_to_slip(f"{home_team} vs {away_team}", "Spread", home_team,
                                            game["spread_home_odds"])

                    if game.get("spread_away"):
                        if st.button(f"{away_team}: {game['spread_away']} ({game['spread_away_odds']})",
                                     key=f"spread_away_{game['game_id']}"):
                            add_bet_to_slip(f"{home_team} vs {away_team}", "Spread", away_team,
                                            game["spread_away_odds"])

                with col3:
                    st.markdown("**Totals (Over/Under)**")
                    if game.get("over_odds"):
                        if st.button(f"Over {game['over_under']} ({game['over_odds']})", key=f"over_{game['game_id']}"):
                            add_bet_to_slip(f"{home_team} vs {away_team}", "Total", "Over", game["over_odds"])

                    if game.get("under_odds"):
                        if st.button(f"Under {game['over_under']} ({game['under_odds']})",
                                     key=f"under_{game['game_id']}"):
                            add_bet_to_slip(f"{home_team} vs {away_team}", "Total", "Under", game["under_odds"])

        # Professional Bet Slip with Consistent Grey Background
        st.sidebar.title("📝 Bet Slip")

        bet_slip = st.session_state.get("bet_slip", [])
        if bet_slip:  # Ensure bet_slip is not empty
            st.sidebar.markdown("### Your Bets")
            total_odds = []

            # Iterate through games and their respective bets
            for i, game_bets in enumerate(bet_slip):
                st.sidebar.markdown(f"""
                <div style="background-color: #333333; padding: 15px; border: 1px solid #444444; border-radius: 5px; margin-bottom: 15px;">
                    <h4 style="margin: 0; color: #FFFFFF;">{game_bets['Game']}</h4>
                """, unsafe_allow_html=True)

                for j, sub_bet in enumerate(game_bets["Bets"]):
                    st.sidebar.markdown(f"""
                    <div style="margin: 10px 0;">
                        <span style="color: #AAAAAA; font-size: 14px;"><b>Type:</b> {sub_bet['Type']}</span><br>
                        <span style="color: #FFFFFF; font-size: 16px;"><b>Pick:</b> {sub_bet['Team/Pick']}</span><br>
                        <span style="color: #AAAAAA; font-size: 14px;"><b>Odds:</b> {sub_bet['Odds']}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.sidebar.button(f"❌ Remove ({sub_bet['Type']} - {sub_bet['Team/Pick']})",
                                         key=f"remove_bet_{i}_{j}"):
                        game_bets["Bets"].pop(j)
                        if not game_bets["Bets"]:
                            bet_slip.pop(i)
                        st.experimental_rerun()

                if st.sidebar.button(f"🗑️ Remove All ({game_bets['Game']})", key=f"remove_all_{i}"):
                    bet_slip.pop(i)
                    st.experimental_rerun()

                st.sidebar.markdown("</div>", unsafe_allow_html=True)

            total_odds = [sub_bet["Odds"] for game_bets in bet_slip for sub_bet in game_bets["Bets"]]
            if total_odds:
                parlay_odds = calculate_parlay_odds(total_odds)
                stake_amount = st.sidebar.number_input("Enter your stake amount (in CAD):", value=10.0)
                potential_payout = calculate_payout(parlay_odds, stake_amount)

                st.sidebar.markdown(f"""
                <div style="background-color: #444444; padding: 15px; border: 2px solid #28a745; border-radius: 5px; margin-top: 20px;">
                    <h4 style="margin: 0 0 10px; color: #28a745;">Parlay Summary</h4>
                    <span style="color: #AAAAAA; font-size: 16px;"><b>Total Odds:</b> <span style="color: #28a745;">{parlay_odds:+}</span></span><br>
                    <span style="color: #AAAAAA; font-size: 16px;"><b>Stake:</b> ${stake_amount:.2f} CAD</span><br>
                    <span style="color: #FFFFFF; font-size: 18px; font-weight: bold;"><b>Potential Payout:</b> ${potential_payout:.2f} CAD</span>
                </div>
                """, unsafe_allow_html=True)

                if st.sidebar.button("Place Bet", key="place_bet"):
                    if st.session_state.get("logged_in", False):
                        # Proceed with bet placement logic
                        try:
                            response = requests.post(
                                "https://your-backend-url.com/place-bet",
                                json={"bets": bet_slip, "stake": stake_amount, "user_id": st.session_state["user_id"]},
                            )
                            if response.status_code == 200:
                                st.sidebar.success("Bet placed successfully!")
                                st.session_state.bet_slip = []
                                st.experimental_rerun()
                            else:
                                st.sidebar.error(f"Error: {response.json().get('message', 'Unknown error')}")
                        except Exception as e:
                            st.sidebar.error(f"Error: {str(e)}")
                    else:
                        st.sidebar.warning("You must log in to place a bet.")
            else:
                st.sidebar.markdown("""
                <div style="background-color: #333333; padding: 15px; border: 1px solid #444444; border-radius: 5px; text-align: center;">
                    <span style="font-size: 14px; color: #AAAAAA;">No bets added yet.<br>Start selecting bets!</span>
                </div>
                """, unsafe_allow_html=True)

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
        return []

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

    if st.button("🔄 Refresh"):
        st.session_state.pop('games_data', None)  # Reload the app to fetch new data

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