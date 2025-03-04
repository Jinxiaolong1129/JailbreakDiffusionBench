# import json
# import os
# from openai import OpenAI
# import time

# # Initialize the OpenAI client
# client = OpenAI()

# # Constants
# CATEGORIES = [
#     "Illegal activity",
#     "Hate content",
#     "Physical harm and violence",
#     "Nudity and sexual content",
#     "Discrimination and disinformation",
#     "Copyright infringement",
#     "Shocking",
#     "Harassment",
#     "Abuse"
# ]

# # 1. Read the JSON file
# def read_json_file(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         data = json.load(file)
#     return data

# # 2. Create a JSONL file for batch processing
# def create_batch_input(data, output_file):
#     with open(output_file, 'w', encoding='utf-8') as f:
#         for i, prompt in enumerate(data['prompts']):
#             # Create a system message explaining the classification task
#             system_message = {
#                 "role": "system", 
#                 "content": f"You are a content moderation classifier. Classify the given text prompt into exactly ONE of the following categories: {', '.join(CATEGORIES)}. Return ONLY the category name, nothing else."
#             }
            
#             # Create a user message with the text to classify
#             user_message = {
#                 "role": "user",
#                 "content": prompt['text']
#             }
            
#             # Create a batch request
#             request = {
#                 "custom_id": str(prompt['id']),
#                 "method": "POST",
#                 "url": "/v1/chat/completions",
#                 "body": {
#                     "model": "gpt-4o-mini",
#                     "messages": [system_message, user_message],
#                     "max_tokens": 100,  # We only need a short response for the category
#                     "temperature": 0.2  # Lower temperature for more consistent results
#                 }
#             }
            
#             f.write(json.dumps(request) + '\n')
    
#     return output_file

# # 3. Upload the batch input file
# def upload_batch_file(file_path):
#     with open(file_path, 'rb') as f:
#         file = client.files.create(
#             file=f,
#             purpose="batch"
#         )
#     return file.id

# # 4. Create and run a batch job
# def create_batch_job(file_id):
#     batch = client.batches.create(
#         input_file_id=file_id,
#         endpoint="/v1/chat/completions",
#         completion_window="24h",
#         metadata={
#             "description": "Prompt classification batch job"
#         }
#     )
#     return batch.id

# # 5. Wait for batch completion and check status
# def wait_for_batch_completion(batch_id, check_interval=30):
#     while True:
#         batch = client.batches.retrieve(batch_id)
#         print(f"Batch status: {batch.status}, Completed: {batch.request_counts.completed}/{batch.request_counts.total}")
        
#         if batch.status in ['completed', 'failed', 'expired', 'cancelled']:
#             return batch
        
#         time.sleep(check_interval)

# # 6. Process batch results and update the original data
# def process_batch_results(batch, original_data):
#     # Get the output file
#     output_file_id = batch.output_file_id
#     if not output_file_id:
#         print("No output file available")
#         return original_data
    
#     # Download the output
#     file_response = client.files.content(output_file_id)
#     output_content = file_response.text
    
#     # Process each line in the output
#     id_to_category = {}
#     for line in output_content.strip().split('\n'):
#         result = json.loads(line)
#         custom_id = result['custom_id']
        
#         if result['error'] is not None:
#             print(f"Error for prompt {custom_id}: {result['error']}")
#             continue
        
#         # Extract the category from the assistant's response
#         response_body = result['response']['body']
#         category = response_body['choices'][0]['message']['content'].strip()
        
#         # Normalize the category (in case of slight variations)
#         category = next((c for c in CATEGORIES if category.lower() in c.lower()), category)
        
#         id_to_category[custom_id] = category
    
#     # Update the original data with classifications
#     for prompt in original_data['prompts']:
#         prompt_id = str(prompt['id'])
#         if prompt_id in id_to_category:
#             prompt['category'] = id_to_category[prompt_id]
    
#     return original_data

# # 7. Save the updated data
# def save_updated_data(data, output_file):
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(data, f, ensure_ascii=False, indent=4)
#     print(f"Updated data saved to {output_file}")

# # Main process
# def main():
#     # File paths
#     input_file = "data/harmful/civitai/civitai_nsfw_prompts_2000_translate.json"
#     batch_input_file = "batch_classification_input.jsonl"
#     output_file = "data/harmful/civitai/civitai_nsfw_prompts_2000_classified.json"
    
#     # 1. Read data
#     print("Reading input data...")
#     data = read_json_file(input_file)
    
#     # 2. Create batch input
#     print("Creating batch input file...")
#     create_batch_input(data, batch_input_file)
    
#     # 3. Upload batch file
#     print("Uploading batch file...")
#     file_id = upload_batch_file(batch_input_file)
#     print(f"File uploaded with ID: {file_id}")
    
#     # 4. Create batch job
#     print("Creating batch job...")
#     batch_id = create_batch_job(file_id)
#     print(f"Batch job created with ID: {batch_id}")
    
#     # 5. Wait for completion
#     print("Waiting for batch job to complete...")
#     batch = wait_for_batch_completion(batch_id)
    
#     # 6. Process results
#     print("Processing batch results...")
#     updated_data = process_batch_results(batch, data)
    
#     # 7. Save results
#     save_updated_data(updated_data, output_file)
    
#     print("Classification process complete!")

# if __name__ == "__main__":
#     main()



import json
import os
from openai import OpenAI
import time
from datetime import datetime

# Initialize the OpenAI client
client = OpenAI()

# Constants
CATEGORIES = [
    "Illegal activity",
    "Hate content",
    "Physical harm and violence",
    "Nudity and sexual content",
    "Discrimination and disinformation",
    "Copyright infringement",
    "Shocking",
    "Harassment",
    "Abuse"
]

# 1. Read the JSON file
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 2. Create a JSONL file for batch processing
def create_batch_input(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, prompt in enumerate(data['prompts']):
            # Skip prompts that already have a category
            if 'category' in prompt and prompt['category']:
                continue
                
            # Create a system message explaining the classification task
            system_message = {
                "role": "system", 
                "content": f"You are a content moderation classifier. Classify the given text prompt into exactly ONE of the following categories: {', '.join(CATEGORIES)}. Return ONLY the category name, nothing else."
            }
            
            # Create a user message with the text to classify
            user_message = {
                "role": "user",
                "content": prompt['text']
            }
            
            # Create a batch request
            request = {
                "custom_id": str(prompt['id']),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [system_message, user_message],
                    "max_tokens": 100,
                    "temperature": 0.2
                }
            }
            
            f.write(json.dumps(request) + '\n')
    
    return output_file

# 3. Upload the batch input file
def upload_batch_file(file_path):
    with open(file_path, 'rb') as f:
        file = client.files.create(
            file=f,
            purpose="batch"
        )
    return file.id

# 4. Create and run a batch job
def create_batch_job(file_id):
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "Prompt classification batch job"
        }
    )
    return batch.id

# 5. Wait for batch completion and check status with incremental processing
def wait_for_batch_completion(batch_id, original_data, output_file, check_interval=30, save_interval=300):
    last_save_time = time.time()
    last_completed = 0
    processed_ids = set()
    
    while True:
        batch = client.batches.retrieve(batch_id)
        completed = batch.request_counts.completed
        total = batch.request_counts.total
        
        print(f"Batch status: {batch.status}, Completed: {completed}/{total}")
        
        # If there are new completed requests, process them
        if completed > last_completed:
            print(f"New completed requests: {completed - last_completed}")
            
            # Get partial results
            if batch.output_file_id:
                file_response = client.files.content(batch.output_file_id)
                output_content = file_response.text
                
                # Process new results
                newly_processed = process_partial_results(output_content, original_data, processed_ids)
                processed_ids.update(newly_processed)
                
                print(f"Processed {len(newly_processed)} new results. Total processed: {len(processed_ids)}")
                
                # Save periodically
                current_time = time.time()
                if current_time - last_save_time > save_interval:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    interim_file = f"{os.path.splitext(output_file)[0]}_interim_{timestamp}.json"
                    save_updated_data(original_data, interim_file)
                    print(f"Interim save: {interim_file}")
                    last_save_time = current_time
            
            last_completed = completed
        
        # Check if batch is complete or if we've reached a timeout
        if batch.status in ['completed', 'failed', 'expired', 'cancelled']:
            # Final processing to ensure we have everything
            if batch.output_file_id:
                file_response = client.files.content(batch.output_file_id)
                output_content = file_response.text
                process_partial_results(output_content, original_data, processed_ids)
            
            print(f"Batch final status: {batch.status}")
            print(f"Total requests processed: {len(processed_ids)}/{total}")
            
            # If not all completed, log the missing ones
            if len(processed_ids) < total:
                all_ids = set(str(p['id']) for p in original_data['prompts'])
                missing_ids = all_ids - processed_ids
                print(f"Missing {len(missing_ids)} results. IDs: {missing_ids}")
            
            return batch
        
        # Apply a timeout option - if stuck for too long (e.g., 30 minutes with no progress)
        if completed < total and completed == last_completed:
            # This could be implemented to force exit after a certain time without progress
            pass
        
        time.sleep(check_interval)

# 6. Process partial batch results
def process_partial_results(output_content, original_data, processed_ids):
    newly_processed = set()
    
    # Process each line in the output
    for line in output_content.strip().split('\n'):
        try:
            result = json.loads(line)
            custom_id = result['custom_id']
            
            # Skip if we've already processed this ID
            if custom_id in processed_ids:
                continue
            
            if result['error'] is not None:
                print(f"Error for prompt {custom_id}: {result['error']}")
                # We still mark it as processed even if there's an error
                newly_processed.add(custom_id)
                continue
            
            # Extract the category from the assistant's response
            response_body = result['response']['body']
            category = response_body['choices'][0]['message']['content'].strip()
            
            # Normalize the category (in case of slight variations)
            category = next((c for c in CATEGORIES if category.lower() in c.lower()), category)
            
            # Update the original data
            for prompt in original_data['prompts']:
                if str(prompt['id']) == custom_id:
                    prompt['category'] = category
                    break
            
            newly_processed.add(custom_id)
        except Exception as e:
            print(f"Error processing result: {e}")
    
    return newly_processed

# 7. Process complete batch results (can be used as a final check)
def process_batch_results(batch, original_data, processed_ids=None):
    if processed_ids is None:
        processed_ids = set()
        
    # Get the output file
    output_file_id = batch.output_file_id
    if not output_file_id:
        print("No output file available")
        return original_data
    
    # Download the output
    file_response = client.files.content(output_file_id)
    output_content = file_response.text
    
    # Process all results to make sure we didn't miss anything
    process_partial_results(output_content, original_data, processed_ids)
    
    return original_data

# 8. Save the updated data
def save_updated_data(data, output_file):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Updated data saved to {output_file}")

# Function to resume from a failed batch
def resume_from_existing(input_file, existing_file, batch_input_file, output_file):
    # Load original data
    original_data = read_json_file(input_file)
    
    # Load existing partially processed data
    if os.path.exists(existing_file):
        existing_data = read_json_file(existing_file)
        
        # Create a mapping of existing categories
        id_to_category = {}
        for prompt in existing_data['prompts']:
            if 'category' in prompt and prompt['category']:
                id_to_category[str(prompt['id'])] = prompt['category']
        
        # Update original data with existing categories
        for prompt in original_data['prompts']:
            prompt_id = str(prompt['id'])
            if prompt_id in id_to_category:
                prompt['category'] = id_to_category[prompt_id]
    
    # Create a new batch for remaining uncategorized items
    create_batch_input(original_data, batch_input_file)
    
    # Count how many prompts need processing
    remaining = sum(1 for prompt in original_data['prompts'] if 'category' not in prompt or not prompt['category'])
    print(f"Resuming with {remaining} prompts left to process")
    
    if remaining == 0:
        print("All prompts already processed!")
        save_updated_data(original_data, output_file)
        return
    
    # Continue with normal batch processing
    file_id = upload_batch_file(batch_input_file)
    print(f"File uploaded with ID: {file_id}")
    
    batch_id = create_batch_job(file_id)
    print(f"Batch job created with ID: {batch_id}")
    
    processed_ids = set(str(prompt['id']) for prompt in original_data['prompts'] 
                    if 'category' in prompt and prompt['category'])
    print(f"Already processed {len(processed_ids)} prompts")
    
    batch = wait_for_batch_completion(batch_id, original_data, output_file)
    
    # Save final results
    save_updated_data(original_data, output_file)
    
    print("Resume process complete!")

# Main process
def main():
    # File paths
    input_file = "data/harmful/civitai/civitai_nsfw_prompts_2000_translate.json"
    batch_input_file = "batch_classification_input.jsonl"
    output_file = "data/harmful/civitai/civitai_nsfw_prompts_2000_classified.json"
    
    # Check if we have an existing interim file to resume from
    interim_files = [f for f in os.listdir("data/harmful/civitai/") 
                    if f.startswith("civitai_nsfw_prompts_2000_classified_interim_")]
    
    if interim_files:
        # Get the most recent interim file
        latest_interim = sorted(interim_files)[-1]
        existing_file = os.path.join("data/harmful/civitai/", latest_interim)
        print(f"Found interim file: {existing_file}")
        
        # Ask if user wants to resume
        response = input("Do you want to resume from the interim file? (y/n): ")
        if response.lower() == 'y':
            resume_from_existing(input_file, existing_file, batch_input_file, output_file)
            return
    
    # Standard process
    print("Starting new classification process...")
    
    # 1. Read data
    print("Reading input data...")
    data = read_json_file(input_file)
    
    # 2. Create batch input
    print("Creating batch input file...")
    create_batch_input(data, batch_input_file)
    
    # 3. Upload batch file
    print("Uploading batch file...")
    file_id = upload_batch_file(batch_input_file)
    print(f"File uploaded with ID: {file_id}")
    
    # 4. Create batch job
    print("Creating batch job...")
    batch_id = create_batch_job(file_id)
    print(f"Batch job created with ID: {batch_id}")
    
    # 5. Wait for completion with incremental processing
    print("Waiting for batch job to complete with incremental saving...")
    batch = wait_for_batch_completion(batch_id, data, output_file)
    
    # 6. Final save
    save_updated_data(data, output_file)
    
    print("Classification process complete!")

# For cases where a batch is already running
def monitor_existing_batch(batch_id, input_file, output_file):
    print(f"Monitoring existing batch {batch_id}...")
    
    # Load original data
    data = read_json_file(input_file)
    
    # Monitor the batch with incremental processing
    batch = wait_for_batch_completion(batch_id, data, output_file)
    
    # Final save
    save_updated_data(data, output_file)
    
    print("Monitoring complete!")

if __name__ == "__main__":
    # If you have an existing batch that's already running
    # Uncomment and modify these lines:
    # existing_batch_id = "your-batch-id"
    # input_file = "data/harmful/civitai/civitai_nsfw_prompts_2000_translate.json"
    # output_file = "data/harmful/civitai/civitai_nsfw_prompts_2000_classified.json"
    # monitor_existing_batch(existing_batch_id, input_file, output_file)
    
    # Otherwise, run the standard process
    main()