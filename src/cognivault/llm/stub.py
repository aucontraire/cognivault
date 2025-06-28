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
        return LLMResponse(
            text=f"[STUB RESPONSE] You asked: {prompt}",
            tokens_used=0,
            model_name="stub-llm",
            finish_reason="stop",
        )
