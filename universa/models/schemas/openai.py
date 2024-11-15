from httpx._client import Client

from openai.types.chat import (
    ChatCompletion, 
    ChatCompletionMessageToolCall
)

from ...schema import Schema
from ..message import BaseMessage
from ...utils._types import *


OpenAIResponse = Dict[str, Any]

class OpenAIFunctionChoice(Schema):
    name: str = Field(
        description="The name of the function to call.",
    )

class OpenAIToolChoice(Schema):
    type: str = Field(
        description="The type of tool choice.",
        default="function"
    )
    function: OpenAIFunctionChoice = Field(
        description="The function to call.",
    )

class OpenAIToolCallMessage(BaseMessage):
    tool_call_id: str
    role: Literal["tool"] = "tool"
    name: str = Field(description="Name of the function that has been executed"),
    content: Any = Field(description="Result of the function execution"),
    
class OpenAIOutputMessage(BaseMessage):
    content: Optional[str] = Field(
        description="The contents of the message.",
        default=None
    )

    role: Literal["assistant"] = Field(
        description="The role of the author of this message.",
        default="assistant"
    )

    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = Field(
        description="The tool calls generated by the model, such as function calls.",
        default=None
    )

class OpenAIMessage(BaseMessage):
    content: str = Field(description="The contents of the message.")
    role: Literal["user", "assistant", "system"] = Field(
        description="The role of the author of this message."
    )

class OpenAIInput(Schema):
    """
    OpenAI API input schema. They contain all the configs that OpenAI chat completion accepts.
    """
    messages: List[Union[
        OpenAIMessage, 
        OpenAIOutputMessage,
        OpenAIToolCallMessage
    ]] = Field(description="A list of messages comprising the conversation so far.")
    model: str = Field(description="The model to use.")
    tools: Optional[List[JsonSchemaValue]] = Field(
        description="A list of tools the model may call. Currently, only functions are supported as a tool. Use this to provide a list of functions the model may generate JSON inputs for.", 
        default=None
    )
    tool_choice: Optional[Union[Literal["none", "auto"], OpenAIToolChoice]] = Field(
        description="Controls which (if any) function is called by the model. If not provided, the model will choose a function to call.", 
        default=None
    )
    max_tokens: Optional[int] = Field(
        description="The maximum number of [tokens](/tokenizer) that can be generated in the chat completion.", 
        default=None
    )
    temperature: Optional[float] = Field(
        description="What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.",
        default=None
    )
    top_p: Optional[float] = Field(
        description="An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.",
        default=1.0
    )
    frequency_penalty: Optional[float] = Field(
        description="Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.",
        default=0.0
    )
    presence_penalty: Optional[float] = Field(
        description="Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.",
        default=0.0
    )
    stop: Union[str, List[str]] = Field(
        description="Up to 4 sequences where the API will stop generating further tokens.", 
        default=None
    )
    response_format: Optional[JsonSchemaValue] = Field(
        description="An object specifying the format that the model must output. Compatible with GPT-4 Turbo and gpt-3.5-turbo-1106.",
        default=None
    )
    stream: Optional[bool] = Field(
        description="Whether to stream the response or wait for the model to finish generating the response.",
        default=False
    )
    model_config = ConfigDict(
        protected_namespaces=()
    )

class OpenAIOutput(Schema):
    response: Union[str, ChatCompletion] = Field(
        description="The response from the OpenAI API."
    )

class OpenAIConstructor(Schema):
    api_key: Optional[str] = Field(
        description="The API key for authentication.",
        default=None
    )

    organization: Optional[str] = Field(
        description="The organization associated with the API key.",
        default=None
    )

    base_url: Optional[str] = Field(
        description="The base URL for the API.",
        default=None
    )

    timeout: Optional[float] = Field(
        description="The timeout duration for API requests.",
        default=None
    )

    max_retries: int = Field(
        description="The maximum number of retries for failed requests.",
        default=2
    )

    default_headers: Optional[Mapping[str, str]] = Field(
        description="The default headers to include in API requests.",
        default=None
    )

    http_client: Optional[Client] = Field(
        description="The HTTP client to use for making requests.",
        default=None
    )

    _strict_response_validation: bool = False
    model_config = ConfigDict(arbitrary_types_allowed=True)


class OpenAIRequestError(Exception):
    """
    Basic exception class for OpenAI request error.
    """
    def __init__(self, message: str) -> None:
        """
        Initialize the exception with a given message.

        Args:
            * `message` (`str`): Message for the exception.
        """
        super().__init__(message)
        