import os
import json
import subprocess
from tqdm import tqdm
from statistics import mean
from Instructional.Accuracy.query import query_model
from Instructional.Data.format import format_input


def ensure_latest_from_drive(folder_id, dest_path):
    result = subprocess.run(
        ["gdown", f"--folder", f"https://drive.google.com/drive/folders/{folder_id}", "--output", "downloads_tmp"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError("Failed to fetch from Drive")

    downloaded_files = [
        os.path.join("downloads_tmp", f) for f in os.listdir("downloads_tmp") if f.endswith(".json")
    ]
    if not downloaded_files:
        raise FileNotFoundError("No JSON files found in Drive folder")

    latest_file = max(downloaded_files, key=os.path.getmtime)
    os.replace(latest_file, dest_path)


def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = query_model(prompt, model)
        try:
            scores.append(int(score))
        except ValueError:
            continue
    return scores


# 🔹 Config
drive_folder_id = "YOUR_FOLDER_ID"   # <-- replace with Finetuned_checkpoints folder ID
local_path = "Instructional/Accuracy/instruction-data-with-response.json"

# 🔹 Download latest JSON
ensure_latest_from_drive(drive_folder_id, local_path)

# 🔹 Load JSON
with open(local_path, "r") as f:
    json_data = json.load(f)

# 🔹 Run Scoring
scores = generate_model_scores(json_data, json_key="model_response", model="llama3")

print("Scores:", scores)
print("Average Score:", mean(scores) if scores else "N/A")
