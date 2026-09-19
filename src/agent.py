from __future__ import annotations

from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)
        if not results:
            return "Không tìm thấy thông tin phù hợp trong cơ sở tri thức để trả lời câu hỏi."

        context_blocks: list[str] = []
        for i, item in enumerate(results, start=1):
            source = item["metadata"].get("source") or item["metadata"].get("doc_id") or item.get("id") or "tài liệu"
            context_blocks.append(f"[{i}] (Nguồn: {source}):\n{item['content']}")

        context_text = "\n\n".join(context_blocks)
        prompt = (
            "Bạn là trợ lý giải đáp thông tin chính sách và quy định đại học. Dựa vào các đoạn ngữ cảnh sau đây để trả lời câu hỏi.\n"
            "Yêu cầu:\n"
            "- Chỉ sử dụng thông tin trong ngữ cảnh được cung cấp, tuyệt đối không tự suy đoán.\n"
            "- Trích dẫn rõ số thứ tự nguồn [1], [2] tương ứng với thông tin bạn sử dụng.\n"
            "- Nếu ngữ cảnh không có thông tin để trả lời, hãy nêu rõ là tài liệu không đề cập.\n\n"
            f"Ngữ cảnh trích xuất:\n{context_text}\n\n"
            f"Câu hỏi: {question}\n\n"
            "Câu trả lời:"
        )

        return self.llm_fn(prompt)
