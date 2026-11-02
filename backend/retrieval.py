"""Source loading + chunking + lightweight TF-IDF retrieval over a codebase.

Uses scikit-learn (no external embedding API needed) so the MVP runs fully offline
after pip install. Swap for ChromaDB embeddings as an upgrade (see README).
"""
import pathlib
import subprocess
import tempfile

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

LANG_EXT = {
    "python": [".py"],
    "javascript": [".js", ".mjs"],
    "typescript": [".ts", ".tsx"],
    "java": [".java"],
    "php": [".php"],
    "c": [".c", ".h"],
}

# dirs we never want to read when walking a cloned repo
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}


def load_source(code=None, repo_url=None, language="python"):
    """Return {file_path: content} from a pasted snippet or a public git repo."""
    files = {}

    if repo_url:
        tmp = tempfile.mkdtemp()
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, tmp],
                check=True,
                capture_output=True,
                timeout=120,
            )
        except Exception as e:
            raise RuntimeError(
                f"git clone failed: {e}. Ensure git is installed and the URL is a public repo."
            )
        exts = LANG_EXT.get(language, [".py"])
        for p in pathlib.Path(tmp).rglob("*"):
            if not p.is_file() or p.suffix not in exts:
                continue
            if any(part in _SKIP_DIRS for part in p.parts):
                continue
            try:
                files[str(p.relative_to(tmp))] = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                pass
    elif code:
        ext = LANG_EXT.get(language, [".py"])[0]
        files[f"pasted{ext}"] = code

    if not files:
        raise RuntimeError("No source files found. Paste code or give a valid public repo URL.")
    return files


def chunk_files(files, max_chars=2000):
    """Split each file into chunks small enough to feed an LLM as context."""
    chunks = []
    for path, content in files.items():
        buf, size = [], 0
        for line in content.splitlines():
            buf.append(line)
            size += len(line) + 1
            if size >= max_chars:
                chunks.append({"file": path, "text": "\n".join(buf)})
                buf, size = [], 0
        if buf:
            chunks.append({"file": path, "text": "\n".join(buf)})
    return chunks


class Retriever:
    """TF-IDF nearest-neighbour retrieval over code chunks."""

    def __init__(self, chunks):
        self.chunks = chunks
        docs = [f"{c['file']}\n{c['text']}" for c in chunks]
        self._vec = TfidfVectorizer(stop_words="english")
        self._matrix = self._vec.fit_transform(docs)

    def search(self, query, k=6):
        q = self._vec.transform([query])
        sims = cosine_similarity(q, self._matrix).flatten()
        top = sims.argsort()[::-1][:k]
        return [self.chunks[i] for i in top]
