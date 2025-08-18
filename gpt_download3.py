import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
import requests
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm

def download_and_load_gpt2(model_size, models_dir):
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Check if all files already exist → then skip downloads
    if all(os.path.exists(os.path.join(model_dir, f)) for f in filenames):
        
        print(f"[INFO] Using cached GPT-2 {model_size} from {model_dir}")
    else:
        print(f"[INFO] Downloading GPT-2 {model_size} weights...")
        os.makedirs(model_dir, exist_ok=True)
        for filename in filenames:
            file_url = os.path.join(base_url, model_size, filename)
            file_path = os.path.join(model_dir, filename)
            download_file(file_url, file_path)

    # Load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def download_file(url, destination):
    try:
        response = requests.get(url, stream=True, verify=False)
        file_size = int(response.headers.get("content-length", 0))

        # If file exists and size matches → skip download
        if os.path.exists(destination) and file_size == os.path.getsize(destination):
            return  

        block_size = 1024
        desc = url.split("/")[-1]
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=desc) as pbar:
            with open(destination, "wb") as f:
                for chunk in response.iter_content(block_size):
                    pbar.update(len(chunk))
                    f.write(chunk)

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}
    for name, _ in tf.train.list_variables(ckpt_path):
        arr = np.squeeze(tf.train.load_variable(ckpt_path, name))
        parts = name.split("/")[1:]  # skip "model/"
        target = params
        if parts[0].startswith("h"):
            layer_num = int(parts[0][1:])
            target = params["blocks"][layer_num]
        for key in parts[1:-1]:
            target = target.setdefault(key, {})
        target[parts[-1]] = arr
    return params
