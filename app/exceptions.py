"""Agent domain errors mapped to HTTP responses."""


class AgentError(Exception):
    """Base agent error. Clients receive `message` only — never stack traces."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        code: str = "AGENT_ERROR",
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.code = code
        super().__init__(message)


class InvalidRequestError(AgentError):
    def __init__(self, message: str = "Invalid request") -> None:
        super().__init__(message, status_code=422, code="INVALID_REQUEST")


class ConfigurationError(AgentError):
    def __init__(self, message: str = "Agent configuration error") -> None:
        super().__init__(message, status_code=500, code="CONFIGURATION_ERROR")


class LLMProviderError(AgentError):
    def __init__(self, message: str = "LLM provider failure") -> None:
        super().__init__(message, status_code=502, code="LLM_PROVIDER_FAILURE")


class LLMTimeoutError(AgentError):
    def __init__(self, message: str = "LLM request timed out") -> None:
        super().__init__(message, status_code=504, code="LLM_TIMEOUT")


class ToolFailureError(AgentError):
    def __init__(self, message: str = "Tool execution failed") -> None:
        super().__init__(message, status_code=502, code="TOOL_FAILURE")


class AgentExecutionError(AgentError):
    def __init__(self, message: str = "Agent execution failed") -> None:
        super().__init__(message, status_code=500, code="AGENT_EXECUTION_FAILURE")


class UnexpectedAgentError(AgentError):
    def __init__(self, message: str = "Unexpected agent error") -> None:
        super().__init__(message, status_code=500, code="UNEXPECTED_ERROR")
