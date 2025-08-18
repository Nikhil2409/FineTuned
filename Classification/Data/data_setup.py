import urllib.request
import ssl
import zipfile
import os
from pathlib import Path
import pandas as pd

# Paths inside Classification/Data
data_dir = Path("Classification/Data")
zip_path = data_dir / "sms_spam_collection.zip"
extracted_path = data_dir / "sms_spam_collection"
data_file_path = extracted_path / "SMSSpamCollection.tsv"

# Download URL
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    # Create an unverified SSL context
    ssl_context = ssl._create_unverified_context()

    # Downloading the file
    with urllib.request.urlopen(url, context=ssl_context) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    # Unzipping the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # Rename to .tsv
    original_file_path = extracted_path / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")

# Run download/unzip
download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

# Load dataset
df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])

# Balance dataset
def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balanced_df

balanced_df = create_balanced_dataset(df)
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

# Split into train/validation/test
def random_split(df, train_frac, validation_frac):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]
    return train_df, validation_df, test_df

train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

# Save CSVs in Classification/Data
train_df.to_csv(data_dir / "train.csv", index=False)
validation_df.to_csv(data_dir / "validation.csv", index=False)
test_df.to_csv(data_dir / "test.csv", index=False)
print("Train, validation, and test CSVs saved in Classification/Data")
