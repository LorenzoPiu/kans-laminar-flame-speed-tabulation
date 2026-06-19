import gdown
import os

# =========================
# CONFIGURATION
# =========================

# Replace this with your Google Drive FILE ID
FILE_ID = "1Gkt6TpckD-m2q8myIbXsAidibqach6MF"

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "H2CO-DB-merged.csv")

# =========================
# DOWNLOAD
# =========================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    url = f"https://drive.google.com/uc?id={FILE_ID}"

    print("Downloading dataset from Google Drive...")
    print(f"Saving to: {OUTPUT_FILE}")

    gdown.download(url, OUTPUT_FILE, quiet=False)

    print("Download completed.")

if __name__ == "__main__":
    main()
