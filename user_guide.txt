Overview

This repository implements a three-stage pipeline for generating counterfactual
texts based on phrase-level explainability. The complete workflow can be run
with run_pipeline.py after setting the OPENAI_API_KEY environment variable.

1. attention.py - Phrase Extraction
Applies a phrase-level attention model to the input corpus and identifies the
most informative expressions in each text. The output includes the original
message and its top explanatory phrases, which serve as semantic anchors for
controlled generation.

2. prompt.py - Prompt Construction
Converts the extracted phrases into structured prompts. For each instance, the
script combines the original text with its key terms and formulates an
instruction that guides a language model to produce a non-cybersecurity version
while preserving surface structure.

3. ChatGpt-5.py - Text Generation via API
Sends the generated prompts to the ChatGPT API and collects the model outputs.
Before running this stage, provide the API key through the environment:

export OPENAI_API_KEY="your_api_key_here"

The three scripts above can be run one by one when you want to inspect the
intermediate outputs after each stage. However, for a complete experiment it is
usually easier to run all three stages through a single wrapper script. This is
what run_pipeline.py does: it receives the dataset once, passes the correct
input and output filenames to each stage, and executes the full workflow in the
right order.

4. run_pipeline.py - Full Pipeline Runner
Runs the three stages in order:

attention.py -> prompt.py -> ChatGpt-5.py

Example:

python run_pipeline.py --dataset twitter.csv

Together, these scripts form an end-to-end workflow:
explain -> prompt -> generate, allowing controlled transformation of
cybersecurity texts for robustness testing and fact-checking research.
