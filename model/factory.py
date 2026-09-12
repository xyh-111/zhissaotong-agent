from abc import ABC, abstractmethod
from typing import Optional
import os

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from utils.config_handler import rag_conf

# DashScope 的 OpenAI 兼容接口地址（仅对话模型走此接口）
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 兼容接口复用 DashScope 的 API Key（与原生 DashScope 接口是同一把 key）
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")


class DashScopeTextEmbeddings(DashScopeEmbeddings):
    """支持自定义批大小的 DashScope 向量模型封装。

    部分新向量模型（如 qwen3.7-text-embedding）单次请求最多 20 条，
    而原生 DashScopeEmbeddings 对未知模型默认按 25 条一批发送，会触发
    「batch size should not be larger than 20」错误，故这里手动分批。
    """

    def __init__(self, model: str, batch_size: int = 20, **kwargs):
        super().__init__(model=model, **kwargs)
        self._embed_batch_size = batch_size

    def embed_documents(self, texts):
        result = []
        for i in range(0, len(texts), self._embed_batch_size):
            result.extend(super().embed_documents(texts[i:i + self._embed_batch_size]))
        return result


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def __init__(self, model_name: str = None, temperature: float = None):
        # 未指定模型名时回退到业务模型；temperature 为 None 时不注入（保持默认）
        self.model_name = model_name or rag_conf["chat_model_name"]
        self.temperature = temperature

    def generator(self) -> BaseChatModel:
        kwargs = {
            "model": self.model_name,
            "base_url": DASHSCOPE_BASE_URL,
            "api_key": DASHSCOPE_API_KEY,
        }
        if self.temperature is not None:
            # ChatOpenAI 直接支持 temperature 字段，0 保证输出可复现
            kwargs["temperature"] = self.temperature
        return ChatOpenAI(**kwargs)


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Embeddings:
        # 向量模型走 DashScope 原生接口：
        # 兼容接口的 /embeddings 对 embedding 模型存在输入格式缺陷（报 input.contents 错误），
        # 而原生 TextEmbedding 才能正确支持 qwen3.7-text-embedding 等新向量模型；
        # 另需用自定义封装类把单次批大小控制在 20 条以内
        return DashScopeTextEmbeddings(model=rag_conf["embeddings_model_name"])


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()
