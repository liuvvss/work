import json

import numpy as np
import redis
from typing import Optional, List, Union, Any, Dict
import Levenshtein # pip install Levenshtein

class SemanticMessageHistory:
    def __init__(
            self,
            name: str, # 对话的名字，类似session id
            ttl: int=3600*24,
            redis_url: str = "localhost",
            redis_port: int = 6379,
            redis_password: str = None,
    ):
        self.name = name
        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password
        )
        self.ttl = ttl

    def get_history(self):
        history = self.redis.get(f"semantic_history:{self.name}")
        if not history:
            return []
        else:
            return json.loads(history)

    def add_message(self, message: Dict[Any, Any]):
        history = self.get_history()
        history.extend(message)
        self.redis.setex(f"semantic_history:{self.name}", self.ttl, json.dumps(history))

    def get_recent(self, role: Optional[Union[str, List[str]]] = None, top_k=10):
        history = self.get_history()
        if role:
            selected_history = [message for message in history if message.get("role", "") == role]
        else:
            selected_history = history

        if top_k:
            selected_history = selected_history[-top_k:]

        return selected_history

    def get_relevant(self, content: str, top_k=10):
        history = self.get_history()
        selected_history = [message for message in history if content in message.get("content", "")]
        if not selected_history:
            return []
        selected_history = sorted(selected_history, key=lambda message: Levenshtein.ratio(message.get("content", ""), content), reverse=True)
        if top_k:
            selected_history = selected_history[:top_k]
        return selected_history

    def delete_history(self, top_k=10):
        history = self.get_history()
        history = history[-top_k:]
        self.redis.setex(f"semantic_history:{self.name}", self.ttl, json.dumps(history))

    def clear_history(self):
        return self.redis.delete(f"semantic_history:{self.name}")


if __name__ == "__main__":
    history = SemanticMessageHistory(
        name="my-session",
        redis_url="localhost",
    )
    history.clear_history()
    history.add_message([
        {"role": "user", "content": "hello, how are you?"},
        {"role": "llm", "content": "I'm doing fine, thanks."},
        {"role": "user", "content": "what is the weather going to be today?"},
        {"role": "llm", "content": "I don't know", "metadata": {"model": "gpt-4"}},
        {"role": "user", "content": "what is the weather going to be today?"},
    ])

    print("get_history", history.get_history())
    print("get_recent topk=1", history.get_recent(top_k=1))
    print("get_recent role=user", history.get_recent(role="user", top_k=1))

    print("\nget_relevant today", history.get_relevant("today",top_k=1))
    print("get_relevant today", history.get_relevant("thanks",top_k=1))
    #
    # history.clear_history()
    # history.add_message([
    #     {"role": "user", "content": "hello, how are you?"},
    #     {"role": "llm", "content": "I'm doing fine, thanks."},
    #     {"role": "user", "content": "what is the weather going to be today?"},
    # ])
    # history.gen()


