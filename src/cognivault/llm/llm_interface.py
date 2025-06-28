from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional, Callable, Union


class LLMResponse:
    """
    Structured response from an LLM.

    Attributes
    ----------
    text : str
        The generated text content.
    tokens_used : Optional[int]
        Number of tokens consumed, if known.
    model_name : Optional[str]
        Identifier of the model used.
    finish_reason : Optional[str]
        Reason the model stopped generation (e.g., 'length', 'stop').
    """

    def __init__(
        self,
        text: str,
        tokens_used: Optional[int] = None,
        model_name: Optional[str] = None,
        finish_reason: Optional[str] = None,
    ):
        self.text = text
        self.tokens_used = tokens_used
        self.model_name = model_name
        self.finish_reason = finish_reason


class LLMInterface(ABC):
    """
    Abstract base class for large language model (LLM) interfaces.

    This interface defines the standard contract for any LLM implementation,
    including real providers like OpenAI or mock/stub classes used for testing.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        stream: bool = False,
        on_log: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> Union[LLMResponse, Iterator[str]]:
        """
        Generate a completion based on the given prompt.

        Parameters
        ----------
        prompt : str
            The prompt text to send to the LLM.
        stream : bool, optional
            Whether to yield partial output tokens (for streaming).
        on_log : Callable[[str], None], optional
            Callback for logging internal events (e.g., prompt, usage).
        **kwargs : dict
            Additional keyword arguments specific to the implementation
            (e.g., temperature, max_tokens).

        Returns
        -------
        LLMResponse or Iterator[str]
            The structured response from the model or a stream of partial strings.
        """
        pass
