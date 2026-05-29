
from openai import OpenAI
import pandas as pd
import time, csv, traceback

# ======================================
client = OpenAI(
    api_key="sk-..."

)
MODEL_NAME = "gpt-5.2"
MAX_OUTPUT_TOKENS = 220

INPUT_FILE = "prompts_twitter.tsv"

OUTPUT_TSV = "Responcegpt_prompts_twitter_GPT5.tsv"
OUTPUT_TXT = "responses_only_twitter_GPT5.tsv"


# خواندن پرامپت‌ها (TSV)
df = pd.read_csv(INPUT_FILE, sep="\t", encoding="utf-8")


def get_answer(prompt: str) -> str:
    resp = client.responses.create(
        model=MODEL_NAME,
        input=prompt,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    try:
        return resp.output[0].content[0].text.strip()
    except:
        return ""


def tsv_safe(s: str) -> str:
    return str(s).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


# ---------- نوشتن همزمان در دو فایل ----------
with open(OUTPUT_TSV, "w", encoding="utf-8") as f_tsv, \
     open(OUTPUT_TXT, "w", encoding="utf-8") as f_txt:

    # هدر TSV
    f_tsv.write("prompt\tresponse\n")

    for i, row in df.iterrows():
        prompt = str(row.get("question", "")).strip()

        if not prompt or prompt.lower() == "nan":
            continue

        try:
            answer = get_answer(prompt)

            if not answer:
                answer = "Icould not generate a text for this row"

        except Exception as e:
            print(f"[{i}] ERROR:", e)
            answer = "Icould not generate a text for this row"

        # ---- ذخیره در TSV ----
        f_tsv.write(f"{tsv_safe(prompt)}\t{tsv_safe(answer)}\n")
        f_tsv.flush()

        # ---- ذخیره فقط پاسخ ----
        f_txt.write(tsv_safe(answer) + "\n")
        f_txt.flush()

        print(f"[{i}] saved (resp chars={len(answer)})")

        time.sleep(1.0)


print("Done.")
print("TSV file :", OUTPUT_TSV)
print("TXT file :", OUTPUT_TXT)
