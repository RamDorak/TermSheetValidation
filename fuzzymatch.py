import json
import re
from fuzzywuzzy import fuzz

with open("structured_output.json", "r") as f:
    extracted_data = json.load(f)

with open("rates.json", "r") as f:
    validation_data = json.load(f)

# Function to extract numeric values from a string
def extract_numeric(text):
    numbers = re.findall(r"\d+\.?\d*", text)
    return float(numbers[0]) if numbers else None

# Function for fuzzy matching on numeric values
def fuzzy_match_numeric(extracted, reference, threshold=90):
    extracted_num = extract_numeric(extracted)
    reference_num = extract_numeric(reference)

    if extracted_num is None or reference_num is None:
        return None

    # Convert to strings for fuzzy matching (handles minor formatting differences)
    score = fuzz.token_sort_ratio(str(extracted_num), str(reference_num))
    match_status = (
        "✅ Exact Match" if score >= 95 else 
        "⚠️ Partial Match" if score >= threshold else 
        "❌ Mismatch"
    )
    return {
        "Extracted": extracted_num,
        "Expected": reference_num,
        "Score": score,
        "Status": match_status
    }

# Perform fuzzy matching only for numeric fields
numeric_results = {}
for field, extracted_value in extracted_data.items():
    reference_value = validation_data.get(field, "N/A")
    match_result = fuzzy_match_numeric(extracted_value, reference_value)

    if match_result:
        numeric_results[field] = match_result

# Save results to a JSON file
with open("fuzzy_match_results.json", "w") as f:
    json.dump(numeric_results, f, indent=4)

print("✅ Numeric fuzzy matching completed. Results saved to fuzzy_match_results.json!")
