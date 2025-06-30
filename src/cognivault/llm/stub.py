from cognivault.llm.llm_interface import LLMInterface, LLMResponse
from typing import Iterator, Optional, Callable, Any, Union


class StubLLM(LLMInterface):
    """
    A stubbed LLM class used for testing without incurring API costs or making external calls.
    """

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
        Generate a mock response for a given prompt. This simulates the behavior of a real LLM.

        Parameters
        ----------
        prompt : str
            The input text prompt to simulate a response for.
        system_prompt : str, optional
            System prompt to provide context/instructions. Included in mock response.
        stream : bool, optional
            If True, simulate streaming output. Ignored in this stub.
        on_log : Callable[[str], None], optional
            Optional logging callback for tracing messages. Ignored in this stub.
        **kwargs : dict
            Additional keyword arguments (ignored in this stub).

        Returns
        -------
        LLMResponse
            A canned response object.
        """
        response_text = f"[STUB RESPONSE] You asked: {prompt}"
        if system_prompt:
            response_text = (
                f"[STUB RESPONSE] System: {system_prompt[:50]}... | User: {prompt}"
            )

        return LLMResponse(
            text=response_text,
            tokens_used=0,
            model_name="stub-llm",
            finish_reason="stop",
        )
