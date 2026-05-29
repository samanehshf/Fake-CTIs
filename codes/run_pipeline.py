"""Run the complete phrase-attention counterfactual generation pipeline.

Example:
    python run_pipeline.py --dataset twitter.csv

For local testing without calling the OpenAI API:
    python run_pipeline.py --dataset twitter.csv --skip-generation
"""

import argparse, os, re, subprocess, sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def safe_dataset_name(path: Path) -> str:
    """Create a filesystem-safe dataset name from the input filename."""
    name = path.stem.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    return name or "dataset"


def run_stage(script_name: str, env: dict[str, str]) -> None:
    """Run one pipeline stage inside the attention project directory."""
    print(f"\n[PIPELINE] Running {script_name}")
    subprocess.run(
        [sys.executable, script_name],
        cwd=SCRIPT_DIR,
        env=env,
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run attention.py -> prompt.py -> ChatGpt-5.py for one dataset."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Input TSV/CSV dataset path. Relative paths are resolved from this folder.",
    )
    parser.add_argument(
        "--dataset-name",
        help="Name used in output filenames. Defaults to the dataset filename stem.",
    )
    parser.add_argument(
        "--text-col",
        default="clean_tweet",
        help="Text column name in the dataset.",
    )
    parser.add_argument(
        "--label-col",
        default="relevant",
        help="Label column name in the dataset.",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Run only attention extraction and prompt construction, without OpenAI API calls.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = SCRIPT_DIR / dataset_path
    dataset_path = dataset_path.resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset_name = args.dataset_name or safe_dataset_name(dataset_path)

    attention_top = f"attention_top_phrases_{dataset_name}.csv"
    per_sample = f"per_sample_metrics_{dataset_name}.csv"
    summary = f"metrics_summary_{dataset_name}.csv"
    prompts = f"prompts_{dataset_name}.tsv"
    responses_tsv = f"Responcegpt_prompts_{dataset_name}_GPT5.tsv"
    responses_txt = f"responses_only_{dataset_name}_GPT5.tsv"

    env = os.environ.copy()
    env.update(
        {
            "ATTENTION_INPUT_FILE": str(dataset_path),
            "ATTENTION_TEXT_COL": args.text_col,
            "ATTENTION_LABEL_COL": args.label_col,
            "ATTENTION_TOP_OUTPUT": attention_top,
            "ATTENTION_PER_SAMPLE_OUTPUT": per_sample,
            "ATTENTION_SUMMARY_OUTPUT": summary,
            "PROMPT_INPUT_FILE": attention_top,
            "PROMPT_OUTPUT_FILE": prompts,
            "CHATGPT_INPUT_FILE": prompts,
            "CHATGPT_OUTPUT_TSV": responses_tsv,
            "CHATGPT_OUTPUT_TXT": responses_txt,
        }
    )

    run_stage("attention.py", env)
    run_stage("prompt.py", env)

    if args.skip_generation:
        print("\n[PIPELINE] Skipped ChatGPT generation.")
    else:
        if not env.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it first or use --skip-generation."
            )
        run_stage("ChatGpt-5.py", env)

    print("\n[PIPELINE] Done.")
    print("Outputs:")
    print(f" - {attention_top}")
    print(f" - {per_sample}")
    print(f" - {summary}")
    print(f" - {prompts}")
    if not args.skip_generation:
        print(f" - {responses_tsv}")
        print(f" - {responses_txt}")


if __name__ == "__main__":
    main()
