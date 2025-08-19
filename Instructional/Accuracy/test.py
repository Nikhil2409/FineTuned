import os
import json
import subprocess
from tqdm import tqdm
from statistics import mean
# This will be your local query_model.py file
from Instructional.Accuracy.query import query_model
from dotenv import dotenv_values
from Instructional.Data.format import format_input


def download_single_file(file_id, dest_path):
    # Remove the old file if it exists to ensure a fresh download
    if os.path.exists(dest_path):
        os.remove(dest_path)

    try:
        # Use gdown with the file ID to download a single file
        subprocess.run(
            ["gdown", "--id", file_id, "--output", dest_path],
            check=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to fetch file from Drive: {e}")


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
config = dotenv_values(".env")
# Replace this with the specific ID of your instruction-data-with-response.json file
json_file_id = config.JSON_FILE_ID
local_path = config.LOCAL_PATH

# 🔹 Download the JSON file
download_single_file(json_file_id, local_path)

# 🔹 Load JSON
with open(local_path, "r") as f:
    json_data = json.load(f)

# 🔹 Run Scoring
scores = generate_model_scores(json_data, json_key="model_response", model="llama3")

print("Scores:", scores)
print("Average Score:", mean(scores) if scores else "N/A")