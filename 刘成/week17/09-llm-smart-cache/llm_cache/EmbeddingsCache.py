import numpy as np
import redis
from typing import Optional, List, Union
import hashlib

class EmbeddingsCache:
    def __init__(
            self,
            name: str, ttl: int=3600*24,
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

    def store(self, text: Union[List[str], str], embedding: np.ndarray):
        if isinstance(text, str):
            text = [text]

        try:
            with self.redis.pipeline() as pipe:
                for i, t in enumerate(text):
                    t_code = hashlib.md5(t.encode()).hexdigest() # 文本的哈希值作为 key
                    key = f"{self.name}:{t_code}"
                    value = embedding.tobytes() if embedding.ndim == 1 else embedding[i].tobytes()
                    pipe.setex(key, self.ttl, value)

                return pipe.execute()
        except:
            return -1

    def delete(self, text: Union[List[str], str]):
        if isinstance(text, str):
            text = [text]

        try:
            key_list = []
            for t in text:  # 修正：不需要索引 i
                t_code = hashlib.md5(t.encode()).hexdigest()
                key_list.append(f"{self.name}:{t_code}")  # 使用 append 更清晰

            return self.redis.delete(*key_list)
        except Exception as e:
            print(f"Delete error: {e}")
            return -1

    def call(self, text: Union[List[str], str]):
        if isinstance(text, str):
            text = [text]

        try:
            key_list = []
            for i, t in enumerate(text):
                t_code = hashlib.md5(t.encode()).hexdigest()
                key_list += [f"{self.name}:{t_code}"]

            results = self.redis.mget(key_list)

            if not results:
                return None

            embeddings = []
            for result in results:
                if result is None:
                    embeddings.append(None)
                else:
                    embedding = np.frombuffer(result, dtype=np.float32) # embedding编码结果 转换为 numpy 数组
                    embeddings.append(embedding)

            return embeddings

        except:
            print("Error")
            return None

if __name__ == "__main__":
    embed_cache = EmbeddingsCache(
        name="embedding_cache",
        ttl=360,
        redis_url="localhost",
    )

    def get_embedding(text):
        return np.random.rand(768)

    print(embed_cache.store(text="hello world", embedding=get_embedding("hello world")))
    print(embed_cache.call(text="hello world"))
    print(embed_cache.delete(text="hello world"))
