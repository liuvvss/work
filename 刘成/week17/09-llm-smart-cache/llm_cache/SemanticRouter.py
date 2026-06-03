import os
import json
import numpy as np
import faiss
import redis
from typing import Callable, Optional, List, Union, Any, Dict


class SemanticRouter:
    def __init__(
        self,
        name: str,
        embedding_method: Callable[[Union[str, List[str]]], Any],
        ttl: int = 3600 * 24,
        redis_url: str = "localhost",
        redis_port: int = 6379,
        redis_password: str = None,
        distance_threshold: float = 0.3,
    ):
        self.name = name
        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password
        )
        self.ttl = ttl
        self.distance_threshold = distance_threshold
        self.embedding_method = embedding_method

        # FAISS 索引
        if os.path.exists(f"{self.name}.index"):
            self.index = faiss.read_index(f"{self.name}.index")
        else:
            self.index = None

        # 从 Redis 恢复 routes 数据
        self._load_routes()

        # 路由结果缓存（内存缓存，避免重复计算）
        self._route_cache = {}

    def _load_routes(self):
        """从 Redis 加载路由数据"""
        routes_data = self.redis.get(f"router:{self.name}:routes")
        if routes_data:
            self.routes = json.loads(routes_data)
        else:
            self.routes = []

    def _save_routes(self):
        """保存路由数据到 Redis"""
        self.redis.setex(f"router:{self.name}:routes", self.ttl, json.dumps(self.routes))

    def add_route(self, references: List[str], name: str, metadata: dict = None):
        """添加一条路由规则

        Args:
            references: 参考问题列表，用于训练路由
            name: 路由目标名称
            metadata: 附加元数据
        """
        embeddings = self.embedding_method(references)

        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])

        self.index.add(embeddings)
        faiss.write_index(self.index, f"{self.name}.index")

        self.routes.append({
            "name": name,
            "references": references,
            "metadata": metadata or {},
        })
        self._save_routes()

        # 记录每个 reference 对应的 route 索引（用于后续追溯）
        for ref in references:
            self.redis.lpush(f"router:{self.name}:refs", ref)

    def route(self, question: str) -> Optional[Dict]:
        """将问题路由到对应的 route

        Args:
            question: 用户问题

        Returns:
            匹配成功返回 {"name": ..., "metadata": ...}，否则返回 None
        """
        # 检查内存缓存
        if question in self._route_cache:
            return self._route_cache[question]

        if self.index is None:
            return None

        embedding = self.embedding_method(question)
        dis, ind = self.index.search(embedding, k=1)

        if dis[0][0] > self.distance_threshold:
            self._route_cache[question] = None
            return None

        route = self.routes[ind[0][0]]
        result = {"name": route["name"], "metadata": route["metadata"]}
        self._route_cache[question] = result
        return result

    def clear(self):
        """清除所有路由数据"""
        self.redis.delete(f"router:{self.name}:routes")
        self.redis.delete(f"router:{self.name}:refs")
        if os.path.exists(f"{self.name}.index"):
            os.unlink(f"{self.name}.index")
        self.index = None
        self.routes = []
        self._route_cache = {}


if __name__ == "__main__":
    import pytest

    # 本地测试
    def mock_embedding(texts):
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for text in texts:
            vec = np.ones(768) * (hash(text) % 100) / 100.0
            vectors.append(vec)
        return np.array(vectors)

    router = SemanticRouter(
        name="demo_router",
        embedding_method=mock_embedding,
        distance_threshold=0.5,
    )

    router.clear()

    router.add_route(
        references=["Hi, good morning", "Hello"],
        name="greeting",
        metadata={"type": "greeting"}
    )

    router.add_route(
        references=["How to refund?", "退货流程"],
        name="refund",
        metadata={"type": "refund"}
    )

    print("Route 'Hi, good morning':", router.route("Hi, good morning"))
    print("Route 'Hello there':", router.route("Hello there"))
    print("Route 'What's the weather':", router.route("What's the weather"))

    router.clear()
    print("After clear, route:", router.route("Hi, good morning"))