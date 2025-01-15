# import json
# import os
# from collections.abc import Callable, Iterable
# from datetime import date, datetime
# from typing import Any, Dict, List, Optional

# from enum import Enum

# import torch

# from .artifact import AttackArtifact, AttackParameters, JailbreakInfo
# # from .judger import DiffusionJailbreakJudge
# from .config import API_KEYS, LOGS_PATH, NUM_BEHAVIORS
# from .defenses import DEFENSES

# from .diffusion_model.core.factory import DiffusionFactory
# from .diffusion_model.core.model_types import ModelType, ModelArchitecture, ModelProvider
# from .diffusion_model.core.outputs import GenerationOutput


# SUBMISSIONS_PATH = "submissions"

# # Model configurations based on factory.py
# MODEL_REGISTRY = {
#     "stable-diffusion-3": ModelProvider.LOCAL,
#     "stable-diffusion-xl-refiner": ModelProvider.LOCAL,
#     "stable-diffusion-safe": ModelProvider.LOCAL,
#     "cogview3": ModelProvider.LOCAL,
#     "min-dalle": ModelProvider.LOCAL,
#     "proteus-rundiffusion": ModelProvider.LOCAL,
#     "dalle-3": ModelProvider.OPENAI,
#     "dalle-2": ModelProvider.OPENAI,
#     "sdxl": ModelProvider.STABILITY,
#     "sdxl-turbo": ModelProvider.STABILITY,
#     "leonardo-sdxl": ModelProvider.LEONARDO,
#     "leonardo-creative": ModelProvider.LEONARDO,
# }

# def check_file_paths(model_name: str) -> str:
#     """Check if the file path for the given model name exists."""
#     model_name = model_name.lower()
#     if model_name not in MODEL_REGISTRY:
#         raise ValueError(f"Invalid model name: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")
#     file_path = os.path.join(SUBMISSIONS_PATH, f"{model_name}.json")
#     if os.path.exists(file_path):
#         raise ValueError(f"{model_name.capitalize()} submission file already exists: {file_path}")
#     return file_path


# def check_eval_logs_empty(submitted_models: Iterable[str]):
#     """Check if evaluation logs are empty for all submitted models."""
#     nonempty_logs = []
#     for model in submitted_models:
#         log_evals_path = os.path.join(LOGS_PATH, "eval", model)
#         if os.path.exists(log_evals_path):
#             if os.listdir(log_evals_path):
#                 nonempty_logs.append(log_evals_path)
#         else:
#             os.makedirs(log_evals_path, exist_ok=True)
#     if nonempty_logs:
#         raise ValueError(f"Warning, {nonempty_logs} are not empty. Please delete them before running the evaluation.")


# def evaluate_prompts(
#     prompts_dict: Dict[str, Dict[str, str]],
#     generation_type: ModelType = ModelType.TEXT_TO_IMAGE,
#     save_dir: str = "outputs",
#     device: str = "cuda",
#     defense: Optional[str] = None,
#     **kwargs
# ) -> Dict[str, Any]:
#     """Generate images for each prompt and classify them as jailbroken or not.

#     Args:
#         prompts_dict: A nested dictionary, model_name -> behavior -> prompt
#         generation_type: Type of generation task (text2img, img2img, etc)
#         save_dir: Directory to save generated images
#         device: Device to run models on ('cuda' or 'cpu')
#         defense: Optional defense mechanism to apply
#         **kwargs: Additional generation parameters
#     Returns:
#         A nested dictionary containing evaluation results
#     """
#     submitted_models = set(prompts_dict.keys())
#     if not all(len(v) == NUM_BEHAVIORS for v in prompts_dict.values()):
#         raise ValueError("Number of behaviors must match NUM_BEHAVIORS for all models.")

#     # Validate models
#     for model_name in submitted_models:
#         if model_name not in MODEL_REGISTRY:
#             raise ValueError(f"Unknown model: {model_name}")

#     # Check logs
#     check_eval_logs_empty(submitted_models)

#     if defense is not None and defense not in DEFENSES:
#         raise ValueError(f"Invalid defense: {defense}. Should be one of {DEFENSES}")

#     evaluation = {}
#     for model_name, behaviors_dict in prompts_dict.items():
#         provider = MODEL_REGISTRY[model_name]
        
#         # Get API key for cloud providers
#         api_key = None
#         if provider != ModelProvider.LOCAL:
#             api_key = os.environ.get(API_KEYS[model_name])
#             if not api_key:
#                 raise ValueError(f"API key required for {model_name}")

#         # Initialize diffusion model
#         diffusion = DiffusionFactory(
#             model_name=model_name,
#             api_key=api_key,
#             save_dir=save_dir,
#             device=device
#         )

#         model_dict = generate_full_model_dict(
#             model_name=model_name,
#             diffusion=diffusion,
#             behavior_prompt_dict=behaviors_dict,
#             generation_type=generation_type,
#             defense=defense,
#             **kwargs
#         )
#         evaluation[model_name] = model_dict

#         # Clean up CUDA memory
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     evaluation["details"] = {
#         "generation_type": generation_type,
#         "evaluation_date": datetime.now().date(),
#         "defense": defense,
#         "generation_params": kwargs
#     }

#     os.makedirs(SUBMISSIONS_PATH, exist_ok=True)
#     evaluation_path = os.path.join(SUBMISSIONS_PATH, "evaluation.json")
#     with open(evaluation_path, "w") as f:
#         json.dump(evaluation, f, indent=4, cls=CustomArtifactEncoder)
#         print(f"Evaluation results saved at {evaluation_path}.")
#     return evaluation


# def generate_full_model_dict(
#     model_name: str,
#     diffusion: DiffusionFactory,
#     behavior_prompt_dict: Dict[str, str],
#     generation_type: ModelType,
#     defense: Optional[str] = None,
#     **kwargs
# ) -> Dict:
#     """Generate images and evaluate them for a single model."""
#     none_behaviors = [behavior for behavior, prompt in behavior_prompt_dict.items() if prompt is None]
#     valid_behaviors = [behavior for behavior, prompt in behavior_prompt_dict.items() if prompt is not None]
#     valid_prompts = [behavior_prompt_dict[beh] for beh in valid_behaviors]

#     print(f"Generating images for {model_name}")
#     outputs: List[Optional[GenerationOutput]] = []
    
#     for prompt in valid_prompts:
#         try:
#             if defense:
#                 prompt = apply_defense(prompt, defense)
            
#             # Handle different generation types
#             if generation_type == ModelType.TEXT_TO_IMAGE:
#                 output = diffusion.generate([prompt], **kwargs)
#             elif generation_type == ModelType.IMAGE_TO_IMAGE:
#                 if "input_image" not in kwargs:
#                     raise ValueError("input_image required for image-to-image generation")
#                 output = diffusion.generate([prompt], **kwargs)
#             elif generation_type == ModelType.INPAINTING:
#                 if "input_image" not in kwargs or "mask" not in kwargs:
#                     raise ValueError("input_image and mask required for inpainting")
#                 output = diffusion.generate([prompt], **kwargs)
#             else:
#                 raise ValueError(f"Unsupported generation type: {generation_type}")
                
#             outputs.append(output)
            
#         except Exception as e:
#             print(f"Error generating image for prompt: {prompt[:50]}... Error: {str(e)}")
#             outputs.append(None)

#     # Log successful generations
#     for i, behavior in enumerate(valid_behaviors):
#         output = outputs[i]
#         if output:
#             diffusion.log_generation(
#                 behavior=behavior,
#                 prompt=valid_prompts[i],
#                 output=output,
#                 phase="eval"
#             )

#     print(f"Classifying images for {model_name}")
#     classifier = DiffusionJailbreakJudge()
    
#     # Filter out failed generations
#     valid_outputs = [out for out in outputs if out is not None]
#     valid_prompts_filtered = [p for i, p in enumerate(valid_prompts) if outputs[i] is not None]
    
#     classifications = classifier(valid_prompts_filtered, valid_outputs)
    
#     # Reinsert failed classifications
#     full_classifications = []
#     valid_idx = 0
#     for output in outputs:
#         if output is None:
#             full_classifications.append(False)
#         else:
#             full_classifications.append(classifications[valid_idx])
#             valid_idx += 1

#     print(
#         f"Number of jailbreaks {model_name}: {sum(full_classifications)}/{NUM_BEHAVIORS} "
#         f"(Total of {len(valid_behaviors)} submitted prompts)"
#     )

#     model_dict = {}
#     for i in range(len(valid_behaviors)):
#         output = outputs[i]
#         model_dict[valid_behaviors[i]] = {
#             "prompt": valid_prompts[i],
#             "images": output.images if output else None,
#             "metadata": output.metadata if output else None,
#             "jailbroken": full_classifications[i],
#             "generation_type": generation_type
#         }
    
#     for behavior in none_behaviors:
#         model_dict[behavior] = {
#             "prompt": None,
#             "images": None,
#             "metadata": None,
#             "jailbroken": False,
#             "generation_type": generation_type
#         }

#     return model_dict


# def _compute_model_summary(
#     model_name: str,
#     model_eval: Dict,
#     log_dict: Dict,
#     model_default_values: Optional[Dict[str, int]] = None
# ) -> Dict:
#     """Compute summary statistics for a model's evaluation results."""
#     behaviors_df = jbb.read_dataset().as_dataframe()

#     total_num_jailbreaks = 0
#     num_submitted_prompts = 0
#     total_generations = 0
#     total_computation_time = 0
#     successful_generations = 0

#     output: Dict[str, Any] = {}
#     jailbreaks_list = []

#     for index, row in enumerate(behaviors_df.iterrows()):
#         row_data = row[1]
#         goal: str = row_data["Goal"]
#         behavior: str = row_data["Behavior"]
#         category: str = row_data["Category"]

#         if behavior not in model_eval:
#             raise ValueError(f"Behavior {behavior} not found in evaluation for {model_name}")

#         eval_behavior_dict = model_eval[behavior]
#         prompt = eval_behavior_dict["prompt"]
#         images = eval_behavior_dict["images"]
#         metadata = eval_behavior_dict["metadata"]
#         generation_type = eval_behavior_dict["generation_type"]
#         classification = eval_behavior_dict["jailbroken"]

#         num_submitted_prompts += prompt is not None
#         total_num_jailbreaks += classification
#         successful_generations += images is not None

#         generations = 0
#         compute_time = 0
#         first_jailbreak_gen = None

#         # Parse generation logs
#         if log_dict and behavior in log_dict:
#             log_behavior_dict = log_dict[behavior]
#             for timestamp, gen_logs in log_behavior_dict:
#                 for gen in gen_logs:
#                     generations += 1
#                     compute_time += gen.get("computation_time", 0)
#                     if images and classification and gen["images"] == images:
#                         first_jailbreak_gen = generations

#         total_generations += generations
#         total_computation_time += compute_time

#         jb_info = JailbreakInfo(
#             index=index,
#             goal=goal,
#             category=category,
#             behavior=behavior,
#             prompt=prompt,
#             images=images,
#             metadata=metadata,
#             generations_to_jailbreak=map_zero(first_jailbreak_gen),
#             number_of_generations=map_zero(generations),
#             computation_time=map_zero(compute_time),
#             generation_type=generation_type,
#             jailbroken=classification
#         )
#         jailbreaks_list.append(jb_info)

#     model_summary_dict = {
#         "total_number_of_jailbreaks": total_num_jailbreaks,
#         "attack_success_rate": total_num_jailbreaks / NUM_BEHAVIORS,
#         "number_of_submitted_prompts": num_submitted_prompts,
#         "successful_generations": successful_generations,
#         "total_generations": map_zero(total_generations),
#         "total_computation_time": map_zero(total_computation_time)
#     }

#     output["model_summary"] = model_summary_dict
#     output["jailbreaks"] = jailbreaks_list
#     return output


# # def apply_defense(prompt: str, defense: str) -> str:
# #     """Apply a defense transformation to the prompt."""
# #     if defense == "input_filtering":
# #         return apply_input_filtering(prompt)
# #     elif defense == "adversarial_training":
# #         return apply_adversarial_training(prompt)
# #     elif defense == "prompt_sanitization":
# #         return apply_prompt_sanitization(prompt)
# #     else:
# #         raise ValueError(f"Unknown defense mechanism: {defense}")


# def map_zero(x: Any) -> Any:
#     """Map zero to None for integer values."""
#     if isinstance(x, int) and x == 0:
#         return None
#     return x


# class CustomArtifactEncoder(json.JSONEncoder):
#     """Custom JSON encoder for artifact types."""
#     def default(self, o: Any) -> Any:
#         if isinstance(o, (AttackParameters, JailbreakInfo, AttackArtifact)):
#             return o.model_dump()
#         elif isinstance(o, (datetime, date)):
#             return o.isoformat()
#         elif isinstance(o, ModelType):
#             return o.value
#         else:
#             return super().default(o)