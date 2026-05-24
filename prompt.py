import pandas as pd

# -----------------------------
# 1) Load input (TAB-separated .csv)
# -----------------------------
INPUT_FILE = "attention_top_phrases_twitter.csv"
OUTPUT_FILE = "prompts_twitter.tsv"  

attention_data = pd.read_csv(INPUT_FILE, sep="\t", encoding="utf-8", engine="python")

# Ensure columns exist
required_cols = {"Sample", "clean_tweet", "TopPhrases"}
missing = required_cols - set(attention_data.columns)
if missing:
    raise ValueError(f"Missing columns in input file: {missing}. Found: {list(attention_data.columns)}")

# -----------------------------
# 2) Generate prompts and save
# -----------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for _, row in attention_data.iterrows():

        # key_terms = TopPhrases column (may be NaN)
        key_terms = str(row["TopPhrases"]) if pd.notna(row["TopPhrases"]) else ""
        key_terms = key_terms.strip()

        # original_text = ستون clean_tweet
        original_text = str(row["clean_tweet"]) if pd.notna(row["clean_tweet"]) else ""
        original_text = original_text.strip()

        # If the text is empty, skip.
        if not original_text:
            continue

        prompt = (
            f"Here is a cybersecurity-related text: '{original_text}'. "
            f"The key important terms in this text are: '{key_terms}'. "
            "Generate a new version of this text that is not about cybersecurity, but still looks like it might be about cybersecurity due to similar key terms and sentence structure. "
            "Ensure that the new text: "
            "1. Does not involve any security-related content (e.g., no vulnerabilities, exploits, attacks, or security patches). "
            "2. Transforms the key terms into similar non-security terms that have a similar look or feel but are clearly about a non-security topic (e.g., software performance, system upgrades, general IT issues, business optimization). "
            "3. Follows the same sentence structure and keeps similar key terms to make it mistakenly classifiable as cybersecurity-related. "
            "4. Use terms that have a similar appearance or context but are clearly related to non-security topics. You can choose terms that are related to software performance, system upgrades, general IT issues, or business optimization. "
            "Try to introduce diversity in the chosen terms across different generated texts to avoid repetitive patterns. "
            "Just give me generated text without extra explanations.\n"
        )

        f.write(prompt)

print(f"Prompts generated and saved to '{OUTPUT_FILE}'")
