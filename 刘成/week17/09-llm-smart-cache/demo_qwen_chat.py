"""调用 Qwen 模型进行对话，集成 SemanticCache 语义缓存 + SemanticMessageHistory 对话历史"""
import os
from openai import OpenAI
from llm_cache.SemanticMessageHistory import SemanticMessageHistory
from llm_cache.SemanticCache import SemanticCache

# Qwen API 配置
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "sk-8916beb8ce594373890f25d8afc8f81e")

client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def get_embedding(texts):
    """调用 Qwen embedding 模型，将文本转为向量"""
    if isinstance(texts, str):
        texts = [texts]
    response = client.embeddings.create(
        model="text-embedding-v3",
        input=texts,
    )
    vectors = [item.embedding for item in response.data]
    import numpy as np
    return np.array(vectors, dtype=np.float32)


# 全局语义缓存（所有 session 共享，节省重复问题的大模型调用）
semantic_cache = SemanticCache(
    name="qwen_semantic_cache",
    embedding_method=get_embedding,
    ttl=3600 * 24,
    redis_url="localhost",
    distance_threshold=0.15,
)


def chat(session_name: str, user_input: str, clear: bool = False) -> str:
    """单轮对话，自动携带历史记录 + 语义缓存"""
    history = SemanticMessageHistory(
        name=session_name,
        redis_url="localhost",
        ttl=3600,
    )

    # 可选：清空历史
    if clear:
        history.clear_history()

    # ========== SemanticCache: 先查语义缓存 ==========
    cached = semantic_cache.call(user_input)
    if cached:
        cached_answers = [c.decode() if isinstance(c, bytes) else c for c in cached if c]
        if cached_answers:
            print(f"[ SemanticCache 命中 {len(cached_answers)} 条相似结果，直接返回]")
            answer = cached_answers[0]
            # 保存本轮（缓存命中也计入历史）
            history.add_message([
                {"role": "user", "content": user_input},
                {"role": "llm", "content": answer},
            ])
            return answer

    # ========== 未命中：调用 Qwen ==========
    raw_messages = history.get_history()
    messages = [{"role": "user" if m.get("role") == "user" else "assistant", "content": m["content"]}
                for m in raw_messages]

    chat_messages = [{"role": "system", "content": "你是一个有帮助的助手。"}]
    chat_messages.extend(messages)
    chat_messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="qwen-plus",
        messages=chat_messages,
        temperature=0.7,
    )

    assistant_reply = response.choices[0].message.content
    print(f"[ SemanticCache 未命中，调用 Qwen，存入缓存]")

    # 保存本轮对话到历史
    history.add_message([
        {"role": "user", "content": user_input},
        {"role": "llm", "content": assistant_reply},
    ])

    # 存入语义缓存（相同/相似问题下次直接命中）
    semantic_cache.store(user_input, assistant_reply)

    return assistant_reply


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen 对话测试（支持语义缓存）")
    parser.add_argument("--session", "-s", default="default", help="Session 名称")
    parser.add_argument("--clear", "-c", action="store_true", help="清空历史后开始")
    parser.add_argument("--input", "-i", default=None, help="直接传入对话内容")
    args = parser.parse_args()

    if args.input:
        reply = chat(args.session, args.input, clear=args.clear)
        print(f"Qwen: {reply}")
    else:
        session = args.session
        print(f"=== 开始对话 (Session: {session})，输入 q 退出 ===")
        print(f"=== 相同/相似问题会自动命中 SemanticCache，无需重复调用 Qwen ===\n")
        if args.clear:
            SemanticMessageHistory(name=session, redis_url="localhost").clear_history()
            print("(已清空历史)\n")

        while True:
            try:
                user_input = input("\n你: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n退出")
                break
            if user_input.lower() in ("q", "quit", "exit"):
                break
            if not user_input:
                continue

            reply = chat(session, user_input)
            print(f"Qwen: {reply}")