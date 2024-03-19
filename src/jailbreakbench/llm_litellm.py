import litellm
from tqdm import tqdm


from jailbreakbench.config import (
    LITELLM_CUSTOM_PROMPTS, LITELLM_MODEL_NAMES
)
from jailbreakbench.llm_wrapper import LLM, LLMOutput
import jailbreakbench.defenses as defenses 

litellm.drop_params = True


class LLMLiteLLM(LLM):
    _litellm_model_name: str
    _api_key: str
    system_prompt: str

    def __init__(self, model_name: str, api_key: str):
        """Initializes the LLMLiteLLM with the specified model name.
        For Together AI models, api_key is the API key for the Together AI API.
        For OpenAI models, api_key is the API key for the OpenAI API.
        """

        super().__init__(model_name)
        self._api_key = api_key
        self._litellm_model_name = LITELLM_MODEL_NAMES[self.model_name]
        self.generation_params: dict[str, int | float] = {}
        if self.temperature == 0:
            self.generation_params["temperature"] = 0
        else:
            self.generation_params["top_p"] = 0.9
        self.generation_params["max_tokens"] = self.max_n_tokens
        self.generation_params["seed"] = 0
        if self.model_name in LITELLM_CUSTOM_PROMPTS:
            litellm.register_prompt_template(
                model=self._litellm_model_name, roles=LITELLM_CUSTOM_PROMPTS[self.model_name]
            )

    def _build_llm_input(self, prompt: str) -> list[dict[str, str]]:
        chat = []
        if self.system_prompt is not None:
            chat.append({"role": "system", "content": self.system_prompt})
        chat.append({"role": "user", "content": prompt})
        return chat
    
    def _query(self, inputs: list) -> list:
        """Query the LLM with supplied inputs."""

        return litellm.batch_completion(
            model=self._litellm_model_name,
            messages=inputs,
            api_key=self._api_key,
            **self.generation_params,
        )

    def _get_responses(
            self, 
            prompts: list[str], 
            max_new_tokens: int | None = None,
            defense: str | None = None
        ) -> LLMOutput:

        if max_new_tokens is not None:
            self.generation_params["max_tokens"] = max_new_tokens
        llm_inputs = [self._build_llm_input(prompt) for prompt in prompts]

        # Query the LLM; apply defense if applicable
        if defense is None:
            outputs = self._query(llm_inputs)
        else:
            defense = vars(defenses)[defense](self._query)
            outputs = [defense.query(p) for p in tqdm(llm_inputs)]

        # Collect meta data from queries
        responses: list[str] = [
            output["choices"][0]["message"].content.strip() 
            for output in outputs
        ]
        prompt_tokens: list[int] = [
            output["usage"]["prompt_tokens"] 
            for output in outputs
        ]
        completion_tokens: list[int] = [
            output["usage"]["completion_tokens"] 
            for output in outputs
        ]

        return LLMOutput(
            responses=responses, 
            prompt_tokens=prompt_tokens, 
            completion_tokens=completion_tokens
        )
