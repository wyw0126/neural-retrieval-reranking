"""Safe error categories; never include provider bodies or API credentials."""
class LLMError(RuntimeError):
    pass


class LLMConfigurationError(LLMError):
    pass


class LLMRequestError(LLMError):
    def __init__(self, status_code: int | None = None):
        self.status_code = status_code
        super().__init__(f"LLM request failed (status={status_code})")


class LLMResponseError(LLMError):
    pass
