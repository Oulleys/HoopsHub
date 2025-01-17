import requests

# Replace with the correct API key and endpoint
api_url = "https://api.nba.com/teams/DAL"  # Adjust this if needed
headers = {"Authorization": "Bearer your_actual_api_key_here"}  # Replace with your key

response = requests.get(api_url, headers=headers)

# Check the status and response
print(f"Status Code: {response.status_code}")
print(f"Response Text: {response.text}")