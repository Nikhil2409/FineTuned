import json
from tqdm import tqdm
from Instructional.Accuracy.query import query_model
from Instructional.Data.format import format_input

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
            print(f"Could not convert score: {score}")
            continue

    return scores


# Load your JSON file
with open("Instructional/Accuracy/instruction-data-with-response.json", "r") as f:
    json_data = json.load(f)

# Generate scores
scores = generate_model_scores(json_data, json_key="model_response", model="llama3")

print(scores)
