import pandas as pd

# Load phishing dataset
p1 = pd.read_csv("phishtank.csv", usecols=["url"])
p1["label"] = 1

# Load legit dataset
leg = pd.read_csv("top-1m.csv", names=["rank","domain"])
leg["url"] = "http://" + leg["domain"]
leg["label"] = 0

# Combine all
all_data = pd.concat([p1, leg[["url","label"]]])
all_data = all_data.drop_duplicates("url")
all_data = all_data.sample(frac=1)

# Save final dataset
all_data.to_csv("all_urls_dataset.csv", index=False)

print("DONE! all_urls_dataset.csv is created")
