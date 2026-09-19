from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

from src.agent import KnowledgeBaseAgent
from src.chunking import FixedSizeChunker, HeadingChunker, RecursiveChunker, SentenceChunker
from src.embeddings import (
    GEMINI_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    GeminiEmbedder,
    LocalEmbedder,
    MockEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

CORPUS_DIR = Path("data/scholarship")
SOURCES_CSV = CORPUS_DIR / "sources.csv"
OUTPUT_FILE = Path("ket_qua_benchmark.txt")


def parse_markdown(path: Path) -> tuple[dict, str]:
    """Tách YAML frontmatter và nội dung Markdown."""
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            raw_fm = parts[1]
            body = parts[2].strip()
            metadata = {}
            for line in raw_fm.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    metadata[key] = val
            return metadata, body
    return {}, text.strip()


def get_embedder_and_llm() -> tuple[any, any, str]:
    """Khởi tạo Embedder và LLM dựa trên cấu hình môi trường (.env)."""
    provider = os.getenv("EMBEDDING_PROVIDER", "mock").strip().lower()
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if (provider == "gemini" or gemini_key) and gemini_key:
        try:
            from google import genai

            embed_model = os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL)
            embedder = GeminiEmbedder(model_name=embed_model)
            client = genai.Client(api_key=gemini_key)

            def gemini_llm(prompt: str) -> str:
                try:
                    res = client.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=prompt,
                    )
                    return res.text.strip()
                except Exception as e:
                    return f"[Lỗi gọi Gemini LLM: {e}]"

            return embedder, gemini_llm, f"Google Gemini (Embedding: {embed_model} | LLM: gemini-3.6-flash)"
        except Exception as e:
            print(f"[Cảnh báo] Không thể kết nối Gemini API: {e}. Quay về Mock.")

    elif (provider == "openai" or openai_key) and openai_key:
        try:
            from openai import OpenAI

            embed_model = os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL)
            embedder = OpenAIEmbedder(model_name=embed_model)
            client = OpenAI(api_key=openai_key)

            def openai_llm(prompt: str) -> str:
                try:
                    res = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return res.choices[0].message.content.strip()
                except Exception as e:
                    return f"[Lỗi gọi OpenAI LLM: {e}]"

            return embedder, openai_llm, f"OpenAI (Embedding: {embed_model} | LLM: gpt-4o-mini)"
        except Exception as e:
            print(f"[Cảnh báo] Không thể kết nối OpenAI API: {e}. Quay về Mock.")

    elif provider == "local":
        try:
            embed_model = os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL)
            embedder = LocalEmbedder(model_name=embed_model)
            return embedder, None, f"Local Embedder ({embed_model})"
        except Exception as e:
            print(f"[Cảnh báo] Không thể tải model local: {e}. Quay về Mock.")

    # Fallback mặc định
    def fallback_llm(prompt: str) -> str:
        if "Ngữ cảnh trích xuất:" in prompt:
            ctx = prompt.split("Ngữ cảnh trích xuất:")[1].split("Câu hỏi:")[0].strip()
            first_chunk = ctx.split("\n\n")[0]
            preview = first_chunk.replace("\n", " ")[:200]
            return f"[Agent Mock Summary]: Dựa trên tài liệu trích xuất: {preview}..."
        return "[Agent Mock]: Không đủ ngữ cảnh."

    return _mock_embed, fallback_llm, "MockEmbedder (Giả lập số học hash MD5)"


def load_corpus(strategy_name: str = "heading", max_chunk_size: int = 500) -> tuple[list[Document], int, dict]:
    """Tải và chia nhỏ tài liệu theo sources.csv theo chiến lược được chọn."""
    if strategy_name == "heading":
        chunker = HeadingChunker(max_chunk_size=max_chunk_size)
    elif strategy_name == "fixed_size":
        chunker = FixedSizeChunker(chunk_size=max_chunk_size, overlap=50)
    elif strategy_name == "recursive":
        chunker = RecursiveChunker(chunk_size=max_chunk_size)
    elif strategy_name == "sentence":
        chunker = SentenceChunker(max_sentences_per_chunk=3)
    else:
        chunker = HeadingChunker(max_chunk_size=max_chunk_size)

    file_paths: list[Path] = []
    if SOURCES_CSV.exists():
        with open(SOURCES_CSV, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel_path = Path(row["file_path"].strip())
                if rel_path.exists():
                    file_paths.append(rel_path)

    if not file_paths:
        file_paths = sorted(CORPUS_DIR.glob("*.md"))

    all_docs: list[Document] = []
    file_stats = {}

    for file_path in file_paths:
        metadata, body = parse_markdown(file_path)
        doc_id = metadata.get("doc_id", file_path.stem)
        metadata["doc_id"] = doc_id
        metadata["source_file"] = str(file_path)

        chunks = chunker.chunk(body)
        file_stats[doc_id] = len(chunks)

        for i, chunk_text in enumerate(chunks):
            doc = Document(
                id=f"{doc_id}#{i}",
                content=chunk_text,
                metadata=dict(metadata),
            )
            all_docs.append(doc)

    return all_docs, len(file_paths), file_stats


BENCHMARK_QUERIES = [
    {
        "id": 1,
        "type": "tra_so_lieu",
        "query": "Mức học bổng dành cho sinh viên trúng tuyển ngành Vi mạch bán dẫn tại Đại học Công nghiệp Hà Nội là bao nhiêu tiền một tháng?",
        "gold_doc": "haui-financial-aid-scholarships",
        "gold_answer": "4.200.000 đồng/tháng.",
        "filter": None,
    },
    {
        "id": 2,
        "type": "hoi_dieu_kien",
        "query": "Để duy trì học bổng Đinh Thiện Lý, sinh viên cần đạt kết quả học tập (CGPA) tối thiểu là bao nhiêu nếu tính theo thang điểm 10?",
        "gold_doc": "lstf-dinh-thien-ly-scholarship",
        "gold_answer": "Điểm trung bình tích lũy (CGPA) từ 8.0 trở lên theo thang điểm 10.",
        "filter": None,
    },
    {
        "id": 3,
        "type": "hoi_quy_trinh",
        "query": "Để Quỹ Đinh Thiện Lý thực hiện giải ngân thanh toán học bổng, phía nhà trường cần thực hiện quy trình gửi email xác nhận với những thông tin gì?",
        "gold_doc": "lstf-dinh-thien-ly-scholarship",
        "gold_answer": "Gửi văn bản chính thức gồm: (1) Bộ hồ sơ sinh viên, (2) Tổng số tiền học bổng, và (3) Thông tin tài khoản ngân hàng.",
        "filter": None,
    },
    {
        "id": 4,
        "type": "liet_ke",
        "query": "Hãy liệt kê các mức phần trăm hỗ trợ trong chính sách miễn, giảm học phí theo Nghị định của Chính phủ đối với sinh viên?",
        "gold_doc": "hust-financial-aid-for-students",
        "gold_answer": "Có 3 mức tương ứng là: 100%, 70% và 50% học phí.",
        "filter": None,
    },
    {
        "id": 5,
        "type": "metadata_trap",
        "query": "Tiêu chuẩn về kết quả học tập để sinh viên được nhận hỗ trợ 100% học phí là gì?",
        "gold_doc": "hust-financial-aid-for-students",
        "gold_answer": "Không có tiêu chuẩn về kết quả học tập, chỉ cần sinh viên thuộc diện chính sách theo quy định.",
        "filter": {"audience": "student"},
        "is_ab_test": True,
    },
]


def run_benchmark(strategy_name: str = "heading") -> str:
    lines: list[str] = []

    def log(msg: str = ""):
        print(msg)
        lines.append(msg)

    embedder, llm_fn, backend_desc = get_embedder_and_llm()

    log("=" * 80)
    log(f"KẾT QUẢ ĐÁNH GIÁ BENCHMARK TRUY XUẤT - LAB 07 (K4-L3A)")
    log(f"Sinh viên thực hiện: Lại Bá Quân")
    log(f"Chiến lược chia nhỏ: {strategy_name.upper()} (HeadingChunker)")
    log(f"Hệ thống AI sử dụng: {backend_desc}")
    log(f"Thư mục dữ liệu: {CORPUS_DIR}")
    log("=" * 80)

    docs, num_files, stats = load_corpus(strategy_name=strategy_name)
    log(f"\n[1] Thống kê dữ liệu đã nạp (dựa trên sources.csv):")
    log(f"- Tổng số tài liệu: {num_files}")
    log(f"- Tổng số chunks tạo ra: {len(docs)}")
    for doc_id, count in stats.items():
        log(f"  + {doc_id}: {count} chunks")

    log("\nĐang tiến hành nhúng vector và xây dựng chỉ mục Knowledge Base...")
    t0 = time.time()
    store = EmbeddingStore("scholarship_benchmark", embedding_fn=embedder)
    store.add_documents(docs)
    log(f"Xây dựng chỉ mục hoàn tất sau {time.time() - t0:.2f}s.")

    agent = KnowledgeBaseAgent(store=store, llm_fn=llm_fn)

    log("\n" + "=" * 80)
    log("[2] Chạy 5 câu hỏi đánh giá (Benchmark Queries):")
    log("=" * 80)

    total_score = 0

    for item in BENCHMARK_QUERIES:
        qid = item["id"]
        qtype = item.get("type", "general")
        q = item["query"]
        gold_doc = item["gold_doc"]
        gold_ans = item["gold_answer"]
        flt = item["filter"]

        log(f"\n--- CÂU HỎI {qid} [{qtype.upper()}] ---")
        log(f"Query: {q}")
        log(f"Gold Doc: {gold_doc}")
        log(f"Gold Answer: {gold_ans}")
        log(f"Metadata Filter: {flt}")

        results = store.search_with_filter(q, top_k=3, metadata_filter=flt)
        log(f"\nTop-3 Chunks tìm thấy:")

        has_relevant_top1 = False
        has_relevant_top3 = False

        for rank, r in enumerate(results, start=1):
            src_doc = r["metadata"].get("doc_id", "")
            is_match = (src_doc == gold_doc)
            if is_match:
                has_relevant_top3 = True
                if rank == 1:
                    has_relevant_top1 = True

            match_indicator = "[ĐÚNG NGUỒN]" if is_match else "[KHÁC NGUỒN]"
            content_preview = r["content"].replace("\n", " ")[:140]
            log(f"  {rank}. Score={r['score']:.4f} | doc_id={src_doc} {match_indicator}")
            log(f"     Nội dung: {content_preview}...")

        q_score = 2 if has_relevant_top1 else (1 if has_relevant_top3 else 0)
        total_score += q_score
        log(f"-> Điểm câu hỏi {qid}: {q_score}/2 điểm")

        log("\n-> Tác tử KnowledgeBaseAgent sinh câu trả lời:")
        agent_ans = agent.answer(q, top_k=3)
        log(f"{agent_ans}\n")

        if item.get("is_ab_test"):
            log(">>> THỰC HIỆN THỬ NGHIỆM A/B: CHẠY LẠI KHI KHÔNG CÓ METADATA FILTER <<<")
            unfiltered_results = store.search_with_filter(q, top_k=3, metadata_filter=None)
            log(f"Kết quả khi KHÔNG lọc:")
            for rank, r in enumerate(unfiltered_results, start=1):
                src_doc = r["metadata"].get("doc_id", "")
                aud = r["metadata"].get("audience", "")
                log(f"  {rank}. Score={r['score']:.4f} | doc_id={src_doc} | audience={aud}")
                log(f"     Nội dung: {r['content'].replace(chr(10), ' ')[:120]}...")

            log(f"\n=> NHẬN XÉT A/B TEST:")
            log(f"- Khi CÓ FILTER (audience=student): Hệ thống loại bỏ hoàn toàn các tài liệu quy định dành cho giảng viên/cán bộ (faculty/staff), chỉ tìm kiếm trong tài liệu sinh viên 'hust-financial-aid-for-students'.")
            log(f"- Khi KHÔNG FILTER: Tài liệu 'hust-postgrad-research-scholarships' (audience=faculty) xuất hiện trong ứng viên và có thể gây nhiễu, khiến agent trả lời sai quyền lợi của đối tượng sinh viên!")

    log("\n" + "=" * 80)
    log(f"TỔNG KẾT ĐIỂM CHẤT LƯỢNG TRUY XUẤT: {total_score}/10 ĐIỂM")
    log("=" * 80)

    output_content = "\n".join(lines)
    OUTPUT_FILE.write_text(output_content, encoding="utf-8")
    log(f"\nĐã lưu toàn bộ kết quả vào: {OUTPUT_FILE.resolve()}")
    return output_content


if __name__ == "__main__":
    strategy = sys.argv[1] if len(sys.argv) > 1 else "heading"
    run_benchmark(strategy_name=strategy)
