import os
import json
import torch
from tqdm import tqdm
from Instructional.Data.format import format_input
from Instructional.Training.generate_text import generate
from GPT_Model.functions import text_to_token_ids, token_ids_to_text
from Instructional.model import BASE_CONFIG


def post_training_generate(
    model, tokenizer, device, test_data, 
    name="instruction-data-with-response.json"
):
    """
    Adds 'model_response' to each entry in test_data by generating from the trained model.
    Saves the file inside Instructional/Accuracy/ with the given name.
    """
    model.eval()
    for i, entry in tqdm(enumerate(test_data), total=len(test_data), desc="Generating responses"):
        input_text = format_input(entry)

        with torch.no_grad():
            token_ids = generate(
                model=model,
                idx=text_to_token_ids(input_text, tokenizer).to(device),
                max_new_tokens=256,
                context_size=BASE_CONFIG["context_length"],
                eos_id=50256,
            )

        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

        test_data[i]["model_response"] = response_text

    # 🔹 Ensure path is inside Instructional/Accuracy
    output_path = os.path.join("Instructional", "Accuracy", name)
    
    with open(output_path, "w") as file:
        json.dump(test_data, file, indent=4)

    print(f"✅ Model responses saved to {output_path}")
    return test_data
