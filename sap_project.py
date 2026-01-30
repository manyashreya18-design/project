import pandas as pd
import requests

# Download phishing datasets
print("Downloading PhishTank...")
open("phishtank.csv","wb").write(requests.get(
    "https://data.phishtank.com/data/online-valid.csv").content)

print("Downloading OpenPhish...")
open("openphish.txt","wb").write(requests.get(
    "https://openphish.com/feed.txt").content)

# Load datasets
p1 = pd.read_csv("phishtank.csv", usecols=["url"])
p1["label"] = 1

p2 = pd.read_csv("openphish.txt", names=["url"])
p2["label"] = 1

# ⚠ Legit dataset must be downloaded manually
print("Now download legit dataset manually (Tranco Top 1M)")
print("https://tranco-list.eu/top-1m.csv.zip")

print("After downloading and unzipping, put top-1m.csv in this folder.")

