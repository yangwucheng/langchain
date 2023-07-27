"""Azure OpenAI chat wrapper."""
from __future__ import annotations

import logging
from typing import Any, Dict, Mapping

from pydantic import root_validator

from langchain.chat_models.openai import ChatOpenAI
from langchain.schema import ChatResult
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class SageGPTChatOpenAI(ChatOpenAI):
    model_name: str = "sagegpt-1.0"
    openai_api_type: str = "openai"
    openai_api_base: str = ""
    openai_api_version: str = ""
    openai_api_key: str = ""
    openai_organization: str = ""
    openai_proxy: str = ""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["openai_api_key"] = get_from_dict_or_env(
            values,
            "openai_api_key",
            "OPENAI_API_KEY",
            default="1234567890"
        )
        values["openai_api_base"] = get_from_dict_or_env(
            values,
            "openai_api_base",
            "OPENAI_API_BASE",
            default="https://sagegpt-platform.4paradigm.com/chat-engine/api/v1"
        )
        values["openai_api_version"] = get_from_dict_or_env(
            values,
            "openai_api_version",
            "OPENAI_API_VERSION",
            default="1.0"
        )
        values["openai_api_type"] = get_from_dict_or_env(
            values,
            "openai_api_type",
            "OPENAI_API_TYPE",
            default="openai"
        )
        values["openai_organization"] = get_from_dict_or_env(
            values,
            "openai_organization",
            "OPENAI_ORGANIZATION",
            default="",
        )
        values["openai_proxy"] = get_from_dict_or_env(
            values,
            "openai_proxy",
            "OPENAI_PROXY",
            default="",
        )
        try:
            import openai

        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        try:
            values["client"] = openai.ChatCompletion
        except AttributeError:
            raise ValueError(
                "`openai` has no `ChatCompletion` attribute, this is likely "
                "due to an old version of the openai package. Try upgrading it "
                "with `pip install --upgrade openai`."
            )
        if values["n"] < 1:
            raise ValueError("n must be at least 1.")
        if values["n"] > 1 and values["streaming"]:
            raise ValueError("n must be 1 when streaming.")
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        return {
            **super()._default_params,
            "model": self.model_name,
        }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**self._default_params}

    @property
    def _client_params(self) -> Dict[str, Any]:
        """Get the config params used for the openai client."""
        openai_creds = {
            "api_type": self.openai_api_type,
            "api_version": self.openai_api_version,
        }
        return {**super()._client_params, **openai_creds}

    @property
    def _llm_type(self) -> str:
        return "sagegpt-openai-chat"

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        response["usage"] = {}
        return super()._create_chat_result(response)
