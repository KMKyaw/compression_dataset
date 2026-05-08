# Import Python's built-in JSON module so we can save results as a .json file.
import json
# Import Path so file paths work cleanly across different operating systems.
from pathlib import Path

# Import tqdm so the script shows a progress bar while compressing prompts.
from tqdm import tqdm

# Import the toolkit compressor wrapper used to create an LLMLingua2 compressor.
from pctoolkit.compressors import PromptCompressor
# Import the toolkit dataset loader used to load GSM, arxiv, LongBench, etc.
from pctoolkit.datasets import load_dataset


# Choose which dataset to load; "GSM" means GSM8K in this repository.
DATASET_NAME = "GSM"
# Some datasets, such as LongBench or BBH, need a subdataset name; GSM does not.
SUBDATASET_NAME = ""
# Choose the compression ratio to pass into compressor.compressgo().
RATIO = 0.5
# Choose where the final JSON results file will be written.
OUTPUT_PATH = Path("results") / f"{DATASET_NAME.lower()}_llmlingua2_results.json"


# Define a helper function that extracts the prompt text from one dataset row.
def get_prompt(sample: dict, dataset_name: str) -> str:
    # GSM rows store the prompt in the "question" field.
    if dataset_name == "GSM":
        # Return only the question string, not the whole dataset row.
        return sample["question"]
    # LongBench rows usually split the prompt into context plus input/question.
    if dataset_name == "LongBench":
        # Safely read the long context, or use an empty string if it is missing.
        context = sample.get("context", "")
        # Safely read the input/question, or use an empty string if it is missing.
        question = sample.get("input", "")
        # Join context and question into one prompt string and remove edge whitespace.
        return f"{context}\n\n{question}".strip()
    # BBH rows store each example question in the "input" field.
    if dataset_name == "BBH":
        # Return the BBH input string.
        return sample["input"]
    # Many text datasets in this repo use a field named "text".
    if "text" in sample:
        # Return the text field if it exists.
        return sample["text"]
    # Some datasets may use a field named "content".
    if "content" in sample:
        # Return the content field if it exists.
        return sample["content"]
    # Some datasets may use a field named "question".
    if "question" in sample:
        # Return the question field if it exists.
        return sample["question"]
    # Some datasets may use a field named "input".
    if "input" in sample:
        # Return the input field if it exists.
        return sample["input"]
    # If no known prompt field exists, raise a helpful error with the available keys.
    raise KeyError(f"Could not find a prompt field. Available keys: {list(sample.keys())}")


# Define a helper function that keeps useful non-prompt information from a row.
def get_metadata(sample: dict) -> dict:
    # Start with an empty metadata dictionary.
    metadata = {}
    # Check common answer/title/id fields that are useful to save with results.
    for key in ["answer", "answers", "target", "title", "entry_id"]:
        # Only copy the field if it actually exists in this dataset row.
        if key in sample:
            # Store the original metadata value under the same key.
            metadata[key] = sample[key]
    # Return the metadata dictionary, which may be empty.
    return metadata


# Define a helper function that writes all collected results to disk.
def save_results(results: list, output_path: Path) -> None:
    # Create the output folder if it does not exist yet.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary file first so the final JSON is less likely to be corrupted.
    tmp_path = output_path.with_suffix(".tmp")
    # Open the temporary output file using UTF-8 so non-ASCII text is preserved.
    with tmp_path.open("w", encoding="utf-8") as f:
        # Dump the Python list of dictionaries as nicely indented JSON.
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    # Replace the final output file with the completed temporary file.
    tmp_path.replace(output_path)


# Define the main script logic.
def main() -> None:
    # Create the LLMLingua2 compressor; device="mps" targets Apple Silicon when available.
    compressor = PromptCompressor(type="LLMLingua2Compressor", device="mps")
    # Load the selected dataset using the toolkit's dataset wrapper.
    dataset = load_dataset(DATASET_NAME, SUBDATASET_NAME)

    # Create an empty list that will hold one result dictionary per prompt.
    results = []
    # Loop over every row in the dataset while showing a progress bar.
    for index, sample in enumerate(tqdm(dataset.data, desc=f"Compressing {DATASET_NAME}")):
        # Extract the actual prompt string from the current row.
        prompt = get_prompt(sample, DATASET_NAME)
        # Build the base result row that will be saved to JSON.
        row = {
            # Save the row number so you can match outputs back to the dataset.
            "index": index,
            # Save the original prompt before compression.
            "prompt": prompt,
            # Save any useful labels or identifiers from the original dataset row.
            "metadata": get_metadata(sample),
        }

        # Try compressing this prompt without letting one failure stop the full run.
        try:
            # Run LLMLingua2 compression and store the compressor's output dictionary.
            row["result"] = compressor.compressgo(original_prompt=prompt, ratio=RATIO)
        # Catch any error raised for this individual prompt.
        except Exception as exc:
            # Save the error text in the JSON so you can inspect failed rows later.
            row["error"] = repr(exc)

        # Add this row's output to the full list of results.
        results.append(row)
        # Save after every prompt so completed work is preserved if the script stops.
        save_results(results, OUTPUT_PATH)

    # Print a short success message when the loop finishes.
    print(f"Saved {len(results)} results to {OUTPUT_PATH}")


# Only run main() when this file is executed directly with python3 test.py.
if __name__ == "__main__":
    # Start the script.
    main()
