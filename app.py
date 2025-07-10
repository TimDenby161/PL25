from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import pandas as pd

app = Flask(__name__)

# Load dataset
df = pd.read_csv("epl_player_stats_24_25.csv")
print("Columns:", df.columns.tolist())

safe_globals = [torch.nn.modules.container.Sequential]

with torch.serialization.safe_globals(safe_globals):
    model = torch.load("model.pth", map_location="cpu", weights_only=False)

model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    name = request.json.get("name")
    row = df[df["Player Name"] == name]

    if row.empty:
        return jsonify({"error": "Player not found"}), 404

    # Change to your actual feature column names:
    features = row[["Minutes", "Shots", "Big Chances Missed", "Hit Woodwork", "Offsides", "Touches"]].values
    x = torch.tensor(features, dtype=torch.float32)
    pred = float(model(x).item())
    actual = int(row["Goals"].values[0])  # Or float() if it's not an integer

    return jsonify({"name": name, "predicted": pred, "actual": actual})