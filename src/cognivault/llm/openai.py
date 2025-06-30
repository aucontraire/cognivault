import openai
from typing import Any, Iterator, Optional, Callable, Union, cast, List, Dict
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from .llm_interface import LLMInterface, LLMResponse


class OpenAIChatLLM(LLMInterface):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        base_url: Optional[str] = None,
    ):
        """
        Initialize the OpenAIChatLLM.

        Parameters
        ----------
        api_key : str
            Your OpenAI API key.
        model : str, optional
            The model name to use (default is "gpt-4").
        base_url : str, optional
            Optional base URL for custom endpoints.
        """
        self.api_key = api_key
        self.model: Optional[str] = model
        self.base_url: Optional[str] = base_url
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        on_log: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> Union[LLMResponse, Iterator[str]]:
        """
        Generate a response from the OpenAI LLM.

        Parameters
        ----------
        prompt : str
            The prompt to send to the LLM.
        system_prompt : str, optional
            System prompt to provide context/instructions to the LLM.
        stream : bool, optional
            Whether to yield partial output tokens (default is False).
        on_log : Callable[[str], None], optional
            Logging hook to trace internal decisions.
        **kwargs : dict
            Additional OpenAI API parameters such as temperature, max_tokens, etc.

        Returns
        -------
        LLMResponse or Iterator[str]
            The structured response or a stream of tokens.
        """
        if on_log:
            on_log(f"[OpenAIChatLLM] Prompt: {prompt}")
            if system_prompt:
                on_log(f"[OpenAIChatLLM] System Prompt: {system_prompt}")
            on_log(f"[OpenAIChatLLM] Stream: {stream}, Model: {self.model}")

        # Build messages array with optional system prompt
        messages: List[ChatCompletionMessageParam] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            assert isinstance(self.model, str), "model must be a string"
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=stream,
                **kwargs,
            )
        except openai.APIError as e:
            if on_log:
                on_log(f"[OpenAIChatLLM][error] {str(e)}")
            raise

        if stream:

            def token_generator():
                for chunk in response:
                    delta = chunk.choices[0].delta
                    content = delta.content or ""
                    if on_log and content:
                        on_log(f"[OpenAIChatLLM][streaming] {content}")
                    yield content

            return token_generator()

        else:
            response = cast(ChatCompletion, response)
            if (
                not hasattr(response, "choices")
                or not isinstance(response.choices, list)
                or not response.choices
            ):
                raise ValueError("Missing or invalid 'choices' in response")
            choice = response.choices[0]
            if not hasattr(choice, "message") or not hasattr(choice.message, "content"):
                raise ValueError("Missing 'message.content' in the first choice")

            if not hasattr(response, "usage") or not hasattr(
                response.usage, "total_tokens"
            ):
                raise ValueError("Missing 'usage.total_tokens' in response")

            text = choice.message.content or ""
            tokens_used = response.usage.total_tokens if response.usage else None
            finish_reason = choice.finish_reason

            return LLMResponse(
                text=text,
                tokens_used=tokens_used,
                model_name=self.model,
                finish_reason=finish_reason,
            )
