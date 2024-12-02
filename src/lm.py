from __future__ import annotations

import threading
from typing import Any, Optional
from dspy import LM
from openai import OpenAI


# Custom LM
class OpenAILM(LM):
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        support_structured_response: bool = False,
        **kwargs,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.provider = "default"
        self.system_prompt = system_prompt
        self.model = model
        self.support_structured_response = support_structured_response

        self.kwargs = {
            "temperature": 0.0,
            "max_tokens": 150,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
            **kwargs,
        }

        self._token_usage_lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.history: list[dict[str, Any]] = []

    def basic_request(self, prompt: str, **kwargs):
        messages = [{"role": "user", "content": prompt}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        if "response_format" in kwargs:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=kwargs["response_format"],
            )
        elif "response_format_default" in kwargs:  # for recursive_parsing
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format=kwargs["response_format_default"],
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, **{**self.kwargs, **kwargs}
            )

        self.history.append(
            {
                "prompt": prompt,
                "response": response,
                "kwargs": {**self.kwargs, **kwargs},
            }
        )
        return response

    def log_usage(self, response):
        usage_data = response.usage
        if usage_data:
            with self._token_usage_lock:
                self.prompt_tokens += usage_data.prompt_tokens
                self.completion_tokens += usage_data.completion_tokens

    def get_usage_and_reset(self):
        usage = {
            self.model: {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            }
        }
        self.prompt_tokens = 0
        self.completion_tokens = 0
        return usage

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> "list[str]":
        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        response = self.basic_request(prompt, **kwargs)
        self.log_usage(response)

        choices = response.choices
        completed_choices = [c for c in choices if c.finish_reason != "length"]

        if only_completed and len(completed_choices):
            choices = completed_choices

        completions = [c.message.content for c in choices]

        return completions
