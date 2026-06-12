import os
import time
from typing import TYPE_CHECKING, Any, AsyncGenerator, Iterator, Optional, Type

from pydantic import BaseModel
from theflow.utils.modules import import_dotted_string

from kotaemon.base import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    LLMInterface,
    Param,
    StructuredOutputLLMInterface,
)
from kotaemon.utils.rag_debug import rag_log

from .base import ChatLLM

if TYPE_CHECKING:
    from openai.types.chat.chat_completion_message_param import (
        ChatCompletionMessageParam,
    )


class BaseChatOpenAI(ChatLLM):
    """Base interface for OpenAI chat model, using the openai library

    This class exposes the parameters in resources.Chat. To subclass this class:

        - Implement the `prepare_client` method to return the OpenAI client
        - Implement the `openai_response` method to return the OpenAI response
        - Implement the params relate to the OpenAI client
    """

    _dependencies = ["openai"]
    _capabilities = ["chat", "text"]  # consider as mixin

    api_key: str = Param(help="API key", required=True)
    timeout: Optional[float] = Param(None, help="Timeout for the API request")
    max_retries: Optional[int] = Param(
        None, help="Maximum number of retries for the API request"
    )

    temperature: Optional[float] = Param(
        None,
        help=(
            "Number between 0 and 2 that controls the randomness of the generated "
            "tokens. Lower values make the model more deterministic, while higher "
            "values make the model more random."
        ),
    )
    max_tokens: Optional[int] = Param(
        None,
        help=(
            "Maximum number of tokens to generate. The total length of input tokens "
            "and generated tokens is limited by the model's context length."
        ),
    )
    n: int = Param(
        1,
        help=(
            "Number of completions to generate. The API will generate n completion "
            "for each prompt."
        ),
    )
    stop: Optional[str | list[str]] = Param(
        None,
        help=(
            "Stop sequence. If a stop sequence is detected, generation will stop "
            "at that point. If not specified, generation will continue until the "
            "maximum token length is reached."
        ),
    )
    frequency_penalty: Optional[float] = Param(
        None,
        help=(
            "Number between -2.0 and 2.0. Positive values penalize new tokens "
            "based on their existing frequency in the text so far, decrearsing the "
            "model's likelihood of repeating the same text."
        ),
    )
    presence_penalty: Optional[float] = Param(
        None,
        help=(
            "Number between -2.0 and 2.0. Positive values penalize new tokens "
            "based on their existing presence in the text so far, decrearsing the "
            "model's likelihood of repeating the same text."
        ),
    )
    tool_choice: Optional[str] = Param(
        None,
        help=(
            "Choice of tool to use for the completion. Available choices are: "
            "auto, default."
        ),
    )
    tools: Optional[list[str]] = Param(
        None,
        help="List of tools to use for the completion.",
    )
    logprobs: Optional[bool] = Param(
        None,
        help=(
            "Include log probabilities on the logprobs most likely tokens, "
            "as well as the chosen token."
        ),
    )
    logit_bias: Optional[dict] = Param(
        None,
        help=(
            "Dictionary of logit bias values to add to the logits of the tokens "
            "in the vocabulary."
        ),
    )
    top_logprobs: Optional[int] = Param(
        None,
        help=(
            "An integer between 0 and 5 specifying the number of most likely tokens "
            "to return at each token position, each with an associated log "
            "probability. `logprobs` must also be set to `true` if this parameter "
            "is used."
        ),
    )
    top_p: Optional[float] = Param(
        None,
        help=(
            "An alternative to sampling with temperature, called nucleus sampling, "
            "where the model considers the results of the token with top_p "
            "probability mass. So 0.1 means that only the tokens comprising the "
            "top 10% probability mass are considered."
        ),
    )
    extra_body: Optional[dict[str, Any]] = Param(
        None,
        help=(
            "Extra provider-specific request body. For Ollama's OpenAI-compatible "
            "endpoint this can carry options such as num_ctx and num_predict."
        ),
    )

    def _is_ollama_endpoint(self) -> bool:
        base_url = str(getattr(self, "base_url", "") or "").lower()
        api_key = str(getattr(self, "api_key", "") or "").lower()
        return "11434" in base_url or api_key == "ollama"

    def _env_int(self, *names: str, default: int) -> int:
        for name in names:
            value = os.environ.get(name)
            if value:
                try:
                    return int(value)
                except ValueError:
                    continue
        return default

    def _ollama_timeout(self) -> Optional[float]:
        if not self._is_ollama_endpoint():
            return self.timeout
        minimum = self._env_int("KH_OLLAMA_TIMEOUT", "OLLAMA_TIMEOUT", default=600)
        try:
            configured = float(self.timeout) if self.timeout is not None else 0.0
        except (TypeError, ValueError):
            configured = 0.0
        return max(configured, float(minimum))

    def _ollama_extra_body(self) -> dict[str, Any] | None:
        if not self._is_ollama_endpoint():
            return self.extra_body

        extra_body = dict(self.extra_body or {})
        options = dict(extra_body.get("options") or {})
        options.setdefault(
            "num_ctx",
            self._env_int("KH_OLLAMA_NUM_CTX", "OLLAMA_NUM_CTX", default=32768),
        )
        options.setdefault(
            "num_predict",
            self._env_int(
                "KH_OLLAMA_NUM_PREDICT",
                "OLLAMA_NUM_PREDICT",
                default=1024,
            ),
        )
        extra_body["options"] = options
        return extra_body

    @Param.auto(depends_on=["max_retries"])
    def max_retries_(self):
        if self.max_retries is None:
            from openai._constants import DEFAULT_MAX_RETRIES

            return DEFAULT_MAX_RETRIES
        return self.max_retries

    def prepare_message(
        self, messages: str | BaseMessage | list[BaseMessage]
    ) -> list["ChatCompletionMessageParam"]:
        """Prepare the message into OpenAI format

        Returns:
            list[dict]: List of messages in OpenAI format
        """
        input_: list[BaseMessage] = []
        output_: list["ChatCompletionMessageParam"] = []

        if isinstance(messages, str):
            input_ = [HumanMessage(content=messages)]
        elif isinstance(messages, BaseMessage):
            input_ = [messages]
        else:
            input_ = messages

        for message in input_:
            output_.append(message.to_openai_format())

        return output_

    def _debug_message_summary(self, messages: list[dict]) -> list[dict[str, Any]]:
        summary: list[dict[str, Any]] = []
        for idx, message in enumerate(messages):
            content = message.get("content", "")
            if isinstance(content, list):
                text = "\n".join(
                    str(part.get("text", "")) if isinstance(part, dict) else str(part)
                    for part in content
                )
            else:
                text = str(content or "")
            summary.append(
                {
                    "index": idx,
                    "role": message.get("role"),
                    "content_chars": len(text),
                    "content_preview": text[:1200],
                }
            )
        return summary

    def _debug_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in params.items()
            if key not in {"messages", "tools"}
        }

    def prepare_output(self, resp: dict) -> LLMInterface:
        """Convert the OpenAI response into LLMInterface"""
        additional_kwargs = {}
        first_message = resp["choices"][0]["message"]
        if "tool_calls" in first_message:
            additional_kwargs["tool_calls"] = first_message["tool_calls"]
        for reasoning_key in ("reasoning_content", "reasoning"):
            if first_message.get(reasoning_key):
                additional_kwargs[reasoning_key] = first_message[reasoning_key]

        if resp["choices"][0].get("logprobs") is None:
            logprobs = []
        else:
            all_logprobs = resp["choices"][0]["logprobs"].get("content")
            logprobs = (
                [logprob["logprob"] for logprob in all_logprobs] if all_logprobs else []
            )

        def message_content(message: dict) -> str:
            content = message.get("content") or ""
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        parts.append(str(part.get("text") or ""))
                    else:
                        parts.append(str(part))
                return "".join(parts)
            return str(content)

        usage = resp.get("usage") or {}
        output = LLMInterface(
            candidates=[message_content(_["message"]) for _ in resp["choices"]],
            content=message_content(first_message),
            total_tokens=usage.get("total_tokens", -1),
            prompt_tokens=usage.get("prompt_tokens", -1),
            completion_tokens=usage.get("completion_tokens", -1),
            additional_kwargs=additional_kwargs,
            messages=[
                AIMessage(content=message_content(_["message"]))
                for _ in resp["choices"]
            ],
            logprobs=logprobs,
        )

        return output

    def prepare_client(self, async_version: bool = False):
        """Get the OpenAI client

        Args:
            async_version (bool): Whether to get the async version of the client
        """
        raise NotImplementedError

    def openai_response(self, client, **kwargs):
        """Get the openai response"""
        raise NotImplementedError

    async def aopenai_response(self, client, **kwargs):
        """Get the openai response"""
        raise NotImplementedError

    def invoke(
        self, messages: str | BaseMessage | list[BaseMessage], *args, **kwargs
    ) -> LLMInterface:
        client = self.prepare_client(async_version=False)
        input_messages = self.prepare_message(messages)
        request_id = f"llm-{time.time_ns()}"
        rag_log(
            "llm.invoke.start",
            request_id=request_id,
            llm_class=self.__class__.__name__,
            model=getattr(self, "model", None),
            messages=self._debug_message_summary(input_messages),
            kwargs=kwargs,
        )
        try:
            resp = self.openai_response(
                client, messages=input_messages, stream=False, **kwargs
            ).dict()
            output = self.prepare_output(resp)
            first_choice = (resp.get("choices") or [{}])[0]
            first_message = first_choice.get("message") or {}
            rag_log(
                "llm.invoke.end",
                request_id=request_id,
                model=getattr(self, "model", None),
                finish_reason=first_choice.get("finish_reason"),
                usage=resp.get("usage"),
                content_chars=len(output.text or output.content or ""),
                content_preview=(output.text or output.content or "")[:1200],
                reasoning_chars=len(
                    str(
                        first_message.get("reasoning_content")
                        or first_message.get("reasoning")
                        or ""
                    )
                ),
            )
            return output
        except Exception as exc:
            rag_log(
                "llm.invoke.error",
                request_id=request_id,
                model=getattr(self, "model", None),
                error=repr(exc),
            )
            raise

    async def ainvoke(
        self, messages: str | BaseMessage | list[BaseMessage], *args, **kwargs
    ) -> LLMInterface:
        client = self.prepare_client(async_version=True)
        input_messages = self.prepare_message(messages)
        resp = (
            await self.aopenai_response(
                client, messages=input_messages, stream=False, **kwargs
            )
        ).dict()

        return self.prepare_output(resp)

    def stream(
        self, messages: str | BaseMessage | list[BaseMessage], *args, **kwargs
    ) -> Iterator[LLMInterface]:
        client = self.prepare_client(async_version=False)
        input_messages = self.prepare_message(messages)
        request_id = f"llm-stream-{time.time_ns()}"
        rag_log(
            "llm.stream.start",
            request_id=request_id,
            llm_class=self.__class__.__name__,
            model=getattr(self, "model", None),
            messages=self._debug_message_summary(input_messages),
            kwargs=kwargs,
        )
        try:
            resp = self.openai_response(
                client, messages=input_messages, stream=True, **kwargs
            )
        except Exception as exc:
            rag_log(
                "llm.stream.request_error",
                request_id=request_id,
                model=getattr(self, "model", None),
                error=repr(exc),
            )
            raise

        chunks_seen = 0
        content_chunks = 0
        reasoning_chunks = 0
        content_chars = 0
        reasoning_chars = 0
        last_finish_reason = None
        for c in resp:
            chunk = c.dict()
            chunks_seen += 1
            if not chunk["choices"]:
                rag_log("llm.stream.empty_choices", request_id=request_id, chunk=chunk)
                continue
            delta = chunk["choices"][0]["delta"]
            content = delta.get("content")
            reasoning_content = delta.get("reasoning_content") or delta.get("reasoning")
            finish_reason = chunk["choices"][0].get("finish_reason")
            if finish_reason:
                last_finish_reason = finish_reason
            if content is not None:
                content_chunks += 1
                content_chars += len(content or "")
                rag_log(
                    "llm.stream.content_chunk",
                    request_id=request_id,
                    chunk_index=chunks_seen,
                    content_chars=len(content or ""),
                    content_preview=(content or "")[:500],
                    finish_reason=finish_reason,
                )
                if chunk["choices"][0].get("logprobs") is None:
                    logprobs = []
                else:
                    logprobs = [
                        logprob["logprob"]
                        for logprob in chunk["choices"][0]["logprobs"].get(
                            "content", []
                        )
                    ]

                yield LLMInterface(
                    content=content,
                    logprobs=logprobs,
                    additional_kwargs={
                        "finish_reason": chunk["choices"][0].get("finish_reason")
                    },
                )
            elif reasoning_content:
                reasoning_chunks += 1
                reasoning_chars += len(str(reasoning_content))
                rag_log(
                    "llm.stream.reasoning_chunk",
                    request_id=request_id,
                    chunk_index=chunks_seen,
                    reasoning_chars=len(str(reasoning_content)),
                    reasoning_preview=str(reasoning_content)[:500],
                    finish_reason=finish_reason,
                )
                yield LLMInterface(
                    content="",
                    additional_kwargs={"reasoning_content": reasoning_content},
                )
            else:
                rag_log(
                    "llm.stream.non_content_chunk",
                    request_id=request_id,
                    chunk_index=chunks_seen,
                    chunk=chunk,
                    finish_reason=finish_reason,
                )
        rag_log(
            "llm.stream.end",
            request_id=request_id,
            model=getattr(self, "model", None),
            chunks_seen=chunks_seen,
            content_chunks=content_chunks,
            reasoning_chunks=reasoning_chunks,
            content_chars=content_chars,
            reasoning_chars=reasoning_chars,
            finish_reason=last_finish_reason,
        )

    async def astream(
        self, messages: str | BaseMessage | list[BaseMessage], *args, **kwargs
    ) -> AsyncGenerator[LLMInterface, None]:
        client = self.prepare_client(async_version=True)
        input_messages = self.prepare_message(messages)
        resp = self.openai_response(
            client, messages=input_messages, stream=True, **kwargs
        )

        async for chunk in resp:
            if not chunk.choices:
                continue
            if chunk.choices[0].delta.content is not None:
                yield LLMInterface(content=chunk.choices[0].delta.content)


class ChatOpenAI(BaseChatOpenAI):
    """OpenAI chat model"""

    base_url: Optional[str] = Param(None, help="OpenAI base URL")
    organization: Optional[str] = Param(None, help="OpenAI organization")
    model: str = Param(help="OpenAI model", required=True)

    def prepare_client(self, async_version: bool = False):
        """Get the OpenAI client

        Args:
            async_version (bool): Whether to get the async version of the client
        """
        params = {
            "api_key": self.api_key,
            "organization": self.organization,
            "base_url": self.base_url,
            "timeout": self._ollama_timeout(),
            "max_retries": self.max_retries_,
        }
        if async_version:
            from openai import AsyncOpenAI

            return AsyncOpenAI(**params)

        from openai import OpenAI

        return OpenAI(**params)

    def prepare_params(self, **kwargs):
        if "tools_pydantic" in kwargs:
            kwargs.pop("tools_pydantic")

        params_ = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "n": self.n,
            "stop": self.stop,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "tool_choice": self.tool_choice,
            "tools": self.tools,
            "logprobs": self.logprobs,
            "logit_bias": self.logit_bias,
            "top_logprobs": self.top_logprobs,
            "top_p": self.top_p,
            "extra_body": self._ollama_extra_body(),
        }
        if self._is_ollama_endpoint() and params_["max_tokens"] is None:
            params_["max_tokens"] = self._env_int(
                "KH_OLLAMA_MAX_TOKENS",
                "OLLAMA_MAX_TOKENS",
                "KH_OLLAMA_NUM_PREDICT",
                "OLLAMA_NUM_PREDICT",
                default=1024,
            )
        params = {k: v for k, v in params_.items() if v is not None}
        params.update(kwargs)

        return params

    def openai_response(self, client, **kwargs):
        """Get the openai response"""
        params = self.prepare_params(**kwargs)
        rag_log(
            "llm.openai.request_params",
            model=self.model,
            params=self._debug_params(params),
            messages=self._debug_message_summary(params.get("messages") or []),
        )
        return client.chat.completions.create(**params)

    async def aopenai_response(self, client, **kwargs):
        params = self.prepare_params(**kwargs)
        return await client.chat.completions.create(**params)


class StructuredOutputChatOpenAI(ChatOpenAI):
    """OpenAI chat model that returns structured output"""

    response_schema: Type[BaseModel] = Param(
        help="class that subclasses pydantics BaseModel", required=True
    )

    def prepare_output(self, resp: dict) -> StructuredOutputLLMInterface:
        """Convert the OpenAI response into StructuredOutputLLMInterface"""
        additional_kwargs = {}

        if "tool_calls" in resp["choices"][0]["message"]:
            additional_kwargs["tool_calls"] = resp["choices"][0]["message"][
                "tool_calls"
            ]

        if resp["choices"][0].get("logprobs") is None:
            logprobs = []
        else:
            all_logprobs = resp["choices"][0]["logprobs"].get("content")
            logprobs = (
                [logprob["logprob"] for logprob in all_logprobs] if all_logprobs else []
            )

        output = StructuredOutputLLMInterface(
            parsed=resp["choices"][0]["message"]["parsed"],
            candidates=[(_["message"]["content"] or "") for _ in resp["choices"]],
            content=resp["choices"][0]["message"]["content"] or "",
            total_tokens=resp["usage"]["total_tokens"],
            prompt_tokens=resp["usage"]["prompt_tokens"],
            completion_tokens=resp["usage"]["completion_tokens"],
            messages=[
                AIMessage(content=(_["message"]["content"]) or "")
                for _ in resp["choices"]
            ],
            additional_kwargs=additional_kwargs,
            logprobs=logprobs,
        )

        return output

    def prepare_params(self, **kwargs):
        if "tools_pydantic" in kwargs:
            kwargs.pop("tools_pydantic")

        params_ = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "n": self.n,
            "stop": self.stop,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "tool_choice": self.tool_choice,
            "tools": self.tools,
            "logprobs": self.logprobs,
            "logit_bias": self.logit_bias,
            "top_logprobs": self.top_logprobs,
            "top_p": self.top_p,
            "response_format": self.response_schema,
        }
        params = {k: v for k, v in params_.items() if v is not None}
        params.update(kwargs)

        # doesn't do streaming
        params.pop("stream")

        return params

    def openai_response(self, client, **kwargs):
        """Get the openai response"""
        params = self.prepare_params(**kwargs)

        return client.beta.chat.completions.parse(**params)

    async def aopenai_response(self, client, **kwargs):
        """Get the openai response"""
        params = self.prepare_params(**kwargs)

        return await client.beta.chat.completions.parse(**params)


class AzureChatOpenAI(BaseChatOpenAI):
    """OpenAI chat model provided by Microsoft Azure"""

    azure_endpoint: str = Param(
        help=(
            "HTTPS endpoint for the Azure OpenAI model. The azure_endpoint, "
            "azure_deployment, and api_version parameters are used to construct "
            "the full URL for the Azure OpenAI model."
        ),
        required=True,
    )
    azure_deployment: str = Param(help="Azure deployment name", required=True)
    api_version: str = Param(help="Azure model version", required=True)
    azure_ad_token: Optional[str] = Param(None, help="Azure AD token")
    azure_ad_token_provider: Optional[str] = Param(None, help="Azure AD token provider")

    @Param.auto(depends_on=["azure_ad_token_provider"])
    def azure_ad_token_provider_(self):
        if isinstance(self.azure_ad_token_provider, str):
            return import_dotted_string(self.azure_ad_token_provider, safe=False)

    def prepare_client(self, async_version: bool = False):
        """Get the OpenAI client

        Args:
            async_version (bool): Whether to get the async version of the client
        """
        params = {
            "azure_endpoint": self.azure_endpoint,
            "api_version": self.api_version,
            "api_key": self.api_key,
            "azure_ad_token": self.azure_ad_token,
            "azure_ad_token_provider": self.azure_ad_token_provider_,
            "timeout": self.timeout,
            "max_retries": self.max_retries_,
        }
        if async_version:
            from openai import AsyncAzureOpenAI

            return AsyncAzureOpenAI(**params)

        from openai import AzureOpenAI

        return AzureOpenAI(**params)

    def prepare_params(self, **kwargs):
        if "tools_pydantic" in kwargs:
            kwargs.pop("tools_pydantic")

        params_ = {
            "model": self.azure_deployment,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "n": self.n,
            "stop": self.stop,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "tool_choice": self.tool_choice,
            "tools": self.tools,
            "logprobs": self.logprobs,
            "logit_bias": self.logit_bias,
            "top_logprobs": self.top_logprobs,
            "top_p": self.top_p,
        }
        params = {k: v for k, v in params_.items() if v is not None}
        params.update(kwargs)

        return params

    def openai_response(self, client, **kwargs):
        """Get the openai response"""
        params = self.prepare_params(**kwargs)
        return client.chat.completions.create(**params)

    async def aopenai_response(self, client, **kwargs):
        params = self.prepare_params(**kwargs)
        return await client.chat.completions.create(**params)
