import yaml
import json
import argparse
from pathlib import Path
import sys
import os

# Import the detector classes
from jailbreak_diffusion.judger.pre_checker import OpenAITextDetector, AzureTextDetector, GoogleTextModerator, GPTChecker, LlamaGuardChecker, NSFW_text_classifier_Checker, NSFW_word_match_Checker, distilbert_nsfw_text_checker, distilroberta_nsfw_text_checker, NvidiaAegisChecker


def load_misclassified_samples(json_path):
    """Load misclassified samples from the given JSON file"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading misclassified samples: {e}")
        return []


def initialize_detector(detector_name, config_path):
    """Initialize a detector based on its name and configuration from a config file"""
    try:
        # Load the configuration file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        detector_config = config.get("detectors", {}).get(detector_name, {})
        
        # Initialize detector mapping
        detector_mapping = {
            "openai_text_moderation": OpenAITextDetector,
            "NSFW_text_classifier": NSFW_text_classifier_Checker,
            "NSFW_word_match": NSFW_word_match_Checker,
            "distilbert_nsfw_text_checker": distilbert_nsfw_text_checker,
            "distilroberta_nsfw_text_checker": distilroberta_nsfw_text_checker,
            "gpt_4o_mini": GPTChecker,
            "llama_guard": LlamaGuardChecker,
            "azure_text_moderation": AzureTextDetector,
            "google_text_checker": GoogleTextModerator,
            "nvidia_aegis": NvidiaAegisChecker
        }
        
        if detector_name not in detector_mapping:
            print(f"Error: Unknown detector '{detector_name}'")
            return None
        
        # Handle detectors that don't need parameters (empty dict)
        if not detector_config:
            detector_config = {}
        
        detector = detector_mapping[detector_name](**detector_config)
        print(f"Successfully initialized {detector_name} detector")
        return detector
        
    except Exception as e:
        print(f"Error initializing detector {detector_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def reprocess_samples(detector, samples, batch_size=0):
    """Reprocess samples with the detector and compare with original predictions
    
    Args:
        detector: The detector object to use
        samples: List of sample dictionaries to process
        batch_size: Size of each batch (0 means process all at once)
    """
    results = []
    
    # Extract all text and corresponding metadata for batch processing
    all_texts = []
    valid_samples = []
    
    for i, sample in enumerate(samples):
        text = sample.get("text", "")
        
        if not text:
            print(f"Warning: Empty text in sample {i+1}")
            continue
        
        all_texts.append(text)
        valid_samples.append(sample)
    
    if not valid_samples:
        print("No valid samples to process.")
        return []
    
    try:
        # Process in batches or all at once based on batch_size parameter
        if batch_size <= 0:
            # Process all texts in a single batch
            print(f"Processing all {len(all_texts)} samples in a single batch...")
            new_predictions = detector.check(all_texts, return_scores=True)
            
            if len(new_predictions) != len(all_texts):
                print(f"Warning: Number of predictions ({len(new_predictions)}) doesn't match number of samples ({len(all_texts)})")
                # Adjust the shorter list to match the longer one
                if len(new_predictions) < len(all_texts):
                    print("Padding predictions list...")
                    while len(new_predictions) < len(all_texts):
                        new_predictions.append({"flagged": False, "score": 0.0})
                else:
                    print("Truncating predictions list...")
                    new_predictions = new_predictions[:len(all_texts)]
        else:
            # Process in batches
            print(f"Processing {len(all_texts)} samples in batches of {batch_size}...")
            new_predictions = []
            
            for i in range(0, len(all_texts), batch_size):
                batch_texts = all_texts[i:i+batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(all_texts) + batch_size - 1)//batch_size} ({len(batch_texts)} samples)")
                
                batch_predictions = detector.check(batch_texts, return_scores=True)
                
                if len(batch_predictions) != len(batch_texts):
                    print(f"Warning: Batch predictions count ({len(batch_predictions)}) doesn't match batch size ({len(batch_texts)})")
                    # Handle mismatch by padding or truncating
                    if len(batch_predictions) < len(batch_texts):
                        while len(batch_predictions) < len(batch_texts):
                            batch_predictions.append({"flagged": False, "score": 0.0})
                    else:
                        batch_predictions = batch_predictions[:len(batch_texts)]
                
                new_predictions.extend(batch_predictions)
                
                print(f"Completed batch {i//batch_size + 1}, total predictions so far: {len(new_predictions)}")
        
        # Process each prediction
        for i, (sample, new_prediction) in enumerate(zip(valid_samples, new_predictions)):
            original_label = sample.get("predicted_label", "")
            original_score = sample.get("prediction_score", 0.0)
            
            # Extract binary prediction and confidence score
            if isinstance(new_prediction, bool):
                new_binary_pred = new_prediction
                new_score = 1.0 if new_prediction else 0.0
            elif isinstance(new_prediction, dict):
                if "flagged" in new_prediction:
                    new_binary_pred = new_prediction["flagged"]
                    new_score = new_prediction.get("score", 1.0 if new_binary_pred else 0.0)
                elif "harmful" in new_prediction:
                    new_binary_pred = new_prediction["harmful"]
                    new_score = new_prediction.get("score", 1.0 if new_binary_pred else 0.0)
                elif "score" in new_prediction:
                    new_score = float(new_prediction["score"])
                    new_binary_pred = new_score >= 0.5
                else:
                    print(f"Warning: Unknown dict format in prediction: {new_prediction.keys()}")
                    new_binary_pred = False
                    new_score = 0.0
            else:
                # Handle any other type of prediction
                try:
                    new_score = float(new_prediction)
                    new_binary_pred = new_score >= 0.5
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert prediction to score: {new_prediction}")
                    new_binary_pred = False
                    new_score = 0.0
            
            new_label = "harmful" if new_binary_pred else "benign"
            
            # Determine if the result has changed
            result_changed = new_label != original_label
            score_diff = new_score - original_score
            
            # Record results
            results.append({
                "id": sample.get("id", ""),
                "text": sample.get("text", ""),
                "actual_label": sample.get("actual_label", ""),
                "original_prediction": {
                    "label": original_label,
                    "score": original_score
                },
                "new_prediction": {
                    "label": new_label,
                    "score": new_score,
                    "raw": new_prediction
                },
                "result_changed": result_changed,
                "score_difference": score_diff,
                "category": sample.get("category", []),
                "source": sample.get("source", "")
            })
            
            # Print progress periodically
            if (i + 1) % 100 == 0:
                print(f"Processed {i+1}/{len(valid_samples)} results")
                
    except Exception as e:
        print(f"Error during batch processing: {e}")
        import traceback
        traceback.print_exc()
    
    return results


def save_results(results, output_path):
    """Save reprocessing results to a JSON file"""
    # Calculate summary statistics
    total_samples = len(results)
    changed_predictions = sum(1 for r in results if r["result_changed"])
    
    # Create summary
    summary = {
        "total_samples": total_samples,
        "changed_predictions": changed_predictions,
        "changed_percentage": (changed_predictions / total_samples * 100) if total_samples > 0 else 0,
        "samples": results
    }
    
    # Save to file
    try:
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error saving results: {e}")


def main():
    parser = argparse.ArgumentParser(description="Reprocess misclassified samples with a text detector")
    parser.add_argument(
        "--config", 
        required=True,
        help="Path to evaluation config file"
    )
    parser.add_argument(
        "--detector",
        required=True,
        help="Detector name to use for reprocessing"
    )
    parser.add_argument(
        "--misclassified_path",
        required=True, 
        help="Path to the misclassified samples JSON file"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save reprocessing results (default: will use input filename with _reprocessed suffix)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Batch size for processing (0 means process all at once)"
    )
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output is None:
        misclassified_path = Path(args.misclassified_path)
        args.output = str(misclassified_path.parent / f"{misclassified_path.stem}_reprocessed.json")
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load misclassified samples
    samples = load_misclassified_samples(args.misclassified_path)
    if not samples:
        print("No samples to process. Exiting.")
        return
    
    print(f"Loaded {len(samples)} misclassified samples")
    
    # Initialize detector
    detector = initialize_detector(args.detector, args.config)
    if detector is None:
        print("Failed to initialize detector. Exiting.")
        return
    
    # Reprocess samples
    print(f"Reprocessing samples with {args.detector}...")
    results = reprocess_samples(detector, samples, batch_size=args.batch_size)
    
    # Save results
    save_results(results, args.output)
    
    # Print summary
    total_samples = len(results)
    changed_predictions = sum(1 for r in results if r["result_changed"])
    changed_percentage = (changed_predictions / total_samples * 100) if total_samples > 0 else 0
    
    print("\nReprocessing Summary:")
    print(f"Total samples: {total_samples}")
    print(f"Changed predictions: {changed_predictions} ({changed_percentage:.2f}%)")
    
    # If there are changed predictions, print some examples
    if changed_predictions > 0:
        print("\nExamples of samples with changed predictions:")
        
        # Get first few changed samples (up to 5)
        changed_samples = [r for r in results if r["result_changed"]][:5]
        
        for i, sample in enumerate(changed_samples):
            print(f"\nSample {i+1}:")
            print(f"  ID: {sample['id']}")
            print(f"  Text: {sample['text'][:100]}..." if len(sample['text']) > 100 else f"  Text: {sample['text']}")
            print(f"  Actual Label: {sample['actual_label']}")
            print(f"  Original Prediction: {sample['original_prediction']['label']} (score: {sample['original_prediction']['score']:.4f})")
            print(f"  New Prediction: {sample['new_prediction']['label']} (score: {sample['new_prediction']['score']:.4f})")
            print(f"  Score Difference: {sample['score_difference']:.4f}")


if __name__ == "__main__":
    main()
    
# PYTHONPATH=/home/jin509/jailbreak_diffusion_benchmark/JailbreakDiffusionBench python evaluation_text_detector/reprocess_misclassified.py --config evaluation_text_detector/config/gpt_4o_mini.yaml --detector gpt_4o_mini --misclassified_path evaluation_text_detector/results/gpt_4o_mini/text_checker_eval/text_checker_eval_gpt_4o_mini_misclassified.json