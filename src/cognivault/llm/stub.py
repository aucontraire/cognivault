class StubLLM:
    """
    A stubbed LLM class used for testing without incurring API costs or making external calls.
    """

    def generate(self, prompt: str) -> str:
        """
        Generate a mock response for a given prompt. This simulates the behavior of a real LLM.

        Parameters
        ----------
        prompt : str
            The input text prompt to simulate a response for.

        Returns
        -------
        str
            A canned or fake response string.
        """
        return f"[STUB RESPONSE] You asked: {prompt}"
