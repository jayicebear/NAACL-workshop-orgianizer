import time
import ast
import openai
import json
import numpy as np
import pandas as pd
from pathlib import Path
start_time = time.time()
current_time = time.localtime()
month_date = time.strftime("%m%d_%H%M", current_time)

openai.api_key = yourapikey
model = 'gpt-4o-mini-2024-07-18'
output_json_path = f'/home/jayicebear/snap/NAACL slang/result/{month_date}.json'
output_txt_path = str(Path(output_json_path).with_suffix('.txt'))
output_txt_norm_path = f'/home/jayicebear/snap/NAACL slang/result/norm_{month_date}.txt'

test = pd.read_csv('/home/jayicebear/snap/NAACL slang/unlabeled_600.csv')
#test = pd.read_csv('/home/jayicebear/snap/NAACL slang/unlabeled_600.csv')[:5]       # SIMPLE TEST!!!


# Function to call GPT-4 API
def call_gpt(prompt):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": "You are an assistant that processes text."},
                  {"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response['choices'][0]['message']['content'].strip()

# Function to parse GPT output formatted as a list
def parse_gpt_list_format(output):
    try:
        # Convert GPT output with bullet points into a Python list
        lines = [line.strip("- ").strip() for line in output.splitlines() if line.startswith("- ")]
        return lines
    except Exception as e:
        print(f"Error parsing GPT output: {e}")
        return []

# Function to parse GPT output formatted as a dictionary
def parse_gpt_dict_format(output):
    try:
        # Attempt to evaluate the dictionary directly
        return eval(output)
    except SyntaxError as e:
        print(f"Error parsing GPT dictionary output: {e}")
        # Handle common formatting issues (e.g., replace invalid characters)
        output = output.replace(":", "': '").replace("{'", "{'").replace(", '", ",'").replace("'", '"')
        try:
            return eval(output)
        except Exception as e:
            print(f"Second attempt to parse failed: {e}")
            return {}


# Initialize lists to store results
words_list_outputs = []
words_that_need_normalization_outputs = []
mapping_dictionary_outputs = []
normalized_word_lists = []
final_mapping_dictionaries = []

# Process each comment in the DataFrame
for input_text in test['comment']:
    print(f"Processing comment: {input_text}")
    
    # Step 1: Make a words list
    make_words_list_prompt = f" Split into words (verb, noun, preverb, adverb) from the given text. Answer only in the list format. (ex: ['word1', ..., 'word4'])\n\nText: {input_text}"
    words_list_output_str = call_gpt(make_words_list_prompt)
    try:
        words_list_output = ast.literal_eval(words_list_output_str) if words_list_output_str else []
    except Exception as e:
        print(f"Error parsing words list: {e}")
        words_list_output = []
    words_list_outputs.append(words_list_output)
    print("Words List OUTPUT:", words_list_output)

    # Step 2: Detect words that need normalization
    detect_slang_words_prompt = f"From the following words list, choose the words which are slang, offensive, or toxic and need normalization (e.g., 처넣고, 걍, 처먹었노, 결혼하노, 개같다, 개새끼, 존나, 병신). Provide ONLY the word list in the same list format :\n\n{words_list_output}"
    words_that_need_normalization_output_str = call_gpt(detect_slang_words_prompt)
    try:
        words_that_need_normalization_output = ast.literal_eval(words_that_need_normalization_output_str) if words_that_need_normalization_output_str else []
    except Exception as e:
        print(f"Error parsing words that need normalization: {e}")
        words_that_need_normalization_output = []
    words_that_need_normalization_outputs.append(words_that_need_normalization_output)
    print("Words That Need Normalization:", words_that_need_normalization_output)

    # Step 3: Normalize words and create a mapping dictionary
    normalize_words_prompt = f"Detoxify the following words and provide the changed words (you should provide the normalized/neutral; so that different word compared to original word) ONLY in a dictionary format: ('original_word' : 'changed_word') e.g., ('쳐먹었노':'먹었구나', '결혼하노':'결혼하네', '뒤져':'죽어', '전라디언이랑':'전라도인과', '개새끼':'강아지', '늙은이':'어르신'):\n\n{words_that_need_normalization_output}"
    mapping_dictionary_output_str = call_gpt(normalize_words_prompt)
    try:
        mapping_dictionary_output = ast.literal_eval(mapping_dictionary_output_str) if mapping_dictionary_output_str else {}
    except Exception as e:
        print(f"Error parsing mapping dictionary: {e}")
        mapping_dictionary_output = {}
    mapping_dictionary_outputs.append(mapping_dictionary_output)
    print("Mapping Dictionary OUTPUT:", mapping_dictionary_output)

    # Step 4: Extract normalized words list
    try:
        normalized_word_list = list(mapping_dictionary_output.values()) if mapping_dictionary_output else []
    except Exception as e:
        print(f"Error generating normalized word list: {e}")
        normalized_word_list = []
    normalized_word_lists.append(normalized_word_list)
    print("Normalized Word List:", normalized_word_list)

    # Step 5: Create the final mapping dictionary
    final_mapping_dictionary = {
        word: mapping_dictionary_output.get(word, word) for word in words_list_output
    }
    final_mapping_dictionaries.append(final_mapping_dictionary)
    print("Final Mapping Dictionary:", final_mapping_dictionary)


# Initialize a new DataFrame for storing results
results_df = pd.DataFrame()

# Add results to the independent DataFrame
results_df['comment'] = test['comment']  # Copy the 'comment' column from the original test DataFrame
results_df['words_list_output'] = words_list_outputs
results_df['words_that_need_normalization_output'] = words_that_need_normalization_outputs
results_df['normalized_word_list'] = normalized_word_lists
results_df['mapping_dictionary_output'] = mapping_dictionary_outputs
results_df['final_mapping_dictionary'] = final_mapping_dictionaries
test['mapping_dictionary_output'] = mapping_dictionary_outputs
test['final_mapping_dictionary'] = final_mapping_dictionaries


# results_df.to_csv(output_path, index=True)
# print(f"CSV: Independent results saved to {output_path}")

def convert_numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# Save DataFrame to JSON with decoded characters
 

# Save the DataFrame as JSON
with open(output_json_path, 'w', encoding='utf-8') as json_file:
    json.dump(
        results_df.to_dict(orient='records'),  # Convert DataFrame to a list of dictionaries
        json_file,
        default=convert_numpy_to_list,         # Handle numpy types
        ensure_ascii=False,                    # Ensure proper decoding of non-ASCII characters
        indent=4                               # Pretty-print the JSON
    )
print(f"JSON: Results saved to {output_json_path}")


with open(output_txt_norm_path, 'w') as file:
    # Write the header
    file.write("norm_target\tafter_norm\n")
    file.write("------------------------\n")
    
    # Iterate through the 'final_mapping_dictionary' column of the DataFrame
    for mapping_dictionary_output in test['mapping_dictionary_output']:
        # Skip if mapping_dictionary_output is None
        if mapping_dictionary_output is None:
            continue
        # Iterate through the dictionary and write each mapping
        for original_word, normalized_word in mapping_dictionary_output.items():
            # Write each mapping in tab-separated format
            file.write(f"{original_word}\t{normalized_word}\n")
       
    # Add a final line break for clarity
    file.write("\n")
print(f"NORM TXT: Results saved to {output_txt_norm_path}")



with open(output_txt_path, 'w') as file:
    # Write the header
    file.write("original\tfinal\n")
    file.write("----------------\n")
    
    # Iterate through the 'final_mapping_dictionary' column of the DataFrame
    for final_mapping_dict in test['final_mapping_dictionary']:
        for original_word, normalized_word in final_mapping_dict.items():
            # Write each mapping in tab-separated format
            file.write(f"{original_word}\t{normalized_word}\n")
    
    # Add a final line break for clarity
    file.write("\n")
print(f"TXT: Results saved to {output_txt_path}")



end_time = time.time()
runtime = end_time - start_time
print(f"Overall Runtime: {runtime:.2f} seconds") 
