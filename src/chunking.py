from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        # Split on sentence boundaries while keeping sentences intact
        pattern = r"(?<=[.!?])(?:\s+|\n+)"
        raw_sentences = re.split(pattern, text.strip())
        sentences = [s.strip() for s in raw_sentences if s.strip()]

        if not sentences:
            return []

        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            chunk = " ".join(group).strip()
            if chunk:
                chunks.append(chunk)

        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if not current_text:
            return []

        if len(current_text) <= self.chunk_size:
            return [current_text]

        if not remaining_separators:
            # Fallback: slice into chunks of chunk_size
            return [
                current_text[i : i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
            ]

        separator = remaining_separators[0]
        next_separators = remaining_separators[1:]

        if separator == "":
            splits = list(current_text)
        else:
            splits = current_text.split(separator)

        good_splits: list[str] = []
        for piece in splits:
            if not piece and separator != "":
                continue
            if len(piece) > self.chunk_size:
                sub_chunks = self._split(piece, next_separators)
                good_splits.extend(sub_chunks)
            else:
                good_splits.append(piece)

        # Merge consecutive smaller pieces up to chunk_size
        final_chunks: list[str] = []
        accumulated = ""
        for piece in good_splits:
            if not piece:
                continue
            if not accumulated:
                accumulated = piece
            else:
                joiner = separator if separator != "" else ""
                candidate = accumulated + joiner + piece
                if len(candidate) <= self.chunk_size:
                    accumulated = candidate
                else:
                    final_chunks.append(accumulated)
                    accumulated = piece

        if accumulated:
            final_chunks.append(accumulated)

        return final_chunks or [current_text]


class HeadingChunker:
    """
    Domain-specific chunking strategy for university regulations and structured documents.

    Splits text on Markdown headings (#, ##, ###) or regulation clauses (Điều X).
    Preserves section titles by prepending heading context to subchunks if section is split.
    """

    def __init__(self, max_chunk_size: int = 600, min_chunk_size: int = 50) -> None:
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        heading_pattern = re.compile(r"(?m)^(#{1,4}\s+.+|Điều\s+\d+[\.:\s].*)$")
        matches = list(heading_pattern.finditer(text))

        if not matches:
            return RecursiveChunker(chunk_size=self.max_chunk_size).chunk(text)

        sections: list[tuple[str, str]] = []
        if matches[0].start() > 0:
            preamble = text[: matches[0].start()].strip()
            if preamble:
                sections.append(("Giới thiệu", preamble))

        for i, match in enumerate(matches):
            heading = match.group(0).strip()
            start_idx = match.end()
            end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_content = text[start_idx:end_idx].strip()
            full_section = f"{heading}\n\n{section_content}".strip() if section_content else heading
            sections.append((heading, full_section))

        chunks: list[str] = []
        for heading, section_text in sections:
            if len(section_text) <= self.max_chunk_size:
                if len(section_text) >= self.min_chunk_size:
                    chunks.append(section_text)
                elif chunks:
                    if len(chunks[-1]) + len(section_text) + 2 <= self.max_chunk_size:
                        chunks[-1] = chunks[-1] + "\n\n" + section_text
                    else:
                        chunks.append(section_text)
                else:
                    chunks.append(section_text)
            else:
                sub_chunker = RecursiveChunker(chunk_size=max(100, self.max_chunk_size - len(heading) - 4))
                sub_pieces = sub_chunker.chunk(section_text)
                for piece in sub_pieces:
                    if piece.startswith(heading):
                        chunks.append(piece)
                    else:
                        chunks.append(f"[{heading}]\n{piece}")

        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(x * x for x in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    dot_prod = _dot(vec_a, vec_b)
    sim = dot_prod / (norm_a * norm_b)
    return max(-1.0, min(1.0, float(sim)))


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        overlap = max(0, chunk_size // 10)
        fixed_chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
        sentence_chunker = SentenceChunker(max_sentences_per_chunk=3)
        recursive_chunker = RecursiveChunker(chunk_size=chunk_size)

        fixed_chunks = fixed_chunker.chunk(text)
        sentence_chunks = sentence_chunker.chunk(text)
        recursive_chunks = recursive_chunker.chunk(text)

        def _stats(chunks: list[str]) -> dict:
            count = len(chunks)
            avg_len = sum(len(c) for c in chunks) / count if count > 0 else 0.0
            return {
                "count": count,
                "avg_length": avg_len,
                "chunks": chunks,
            }

        return {
            "fixed_size": _stats(fixed_chunks),
            "by_sentences": _stats(sentence_chunks),
            "recursive": _stats(recursive_chunks),
        }
