"""Polycrit — evaluation-first multi-agent code reviewer.

Pipeline (each stage is streamed to the client as an SSE event):
  1. Ingest + chunk + build a retriever over the code
  2. A panel of reviewer agents (Architecture, Security, Performance, Tests, Style),
     each retrieves the code it needs and emits structured findings
  3. A Critic agent re-checks every finding against the actual code and removes
     hallucinated / unsupported ones — the self-improving guard
  4. A Moderator agent synthesizes the final markdown report, including an honest
     "What I could NOT verify" section
"""
from llm import chat, chat_json
from retrieval import Retriever, chunk_files, load_source

# --- The reviewer panel ---------------------------------------------------
# Each reviewer: a role + the retrieval query it uses to pull relevant code.
REVIEWERS = [
    {
        "name": "Architecture Reviewer",
        "system": "You are a senior software architect. Assess structure, modularity, coupling and design clarity.",
        "query": "architecture entry points modules classes functions design coupling structure",
        "instruction": (
            'Return JSON: {"summary": str, "findings": [{"title": str, "severity": '
            '"Low|Med|High", "file": str, "detail": str, "confidence": '
            '"High|Med|Low", "fix": str}]}. Findings are about design, not bugs. '
            "If none, findings: []."
        ),
    },
    {
        "name": "Security Reviewer",
        "system": "You are a security engineer. Find security issues: injection, hardcoded secrets, missing auth, unsafe deserialization, path traversal, XSS.",
        "query": "security authentication secrets input validation sql xss deserialization",
        "instruction": (
            'Return JSON: {"summary": str, "findings": [{"title": str, "severity": '
            '"Low|Med|High", "file": str, "detail": str, "confidence": '
            '"High|Med|Low", "fix": str}]}. Each finding cites a file. If none, findings: []'
        ),
    },
    {
        "name": "Performance Reviewer",
        "system": "You are a performance engineer. Find issues: N+1 queries, O(n^2) loops, unnecessary allocations, blocking calls, missing indexes.",
        "query": "performance loops queries allocation caching hot path database",
        "instruction": (
            'Return JSON: {"summary": str, "findings": [{"title": str, "severity": '
            '"Low|Med|High", "file": str, "detail": str, "confidence": '
            '"High|Med|Low", "fix": str}]}. If none, findings: []'
        ),
    },
    {
        "name": "Test-Gap Reviewer",
        "system": "You are a QA engineer. Identify important untested behaviors and suggest concrete tests.",
        "query": "tests functions logic branches edge cases validation error paths",
        "instruction": (
            'Return JSON: {"summary": str, "findings": [{"title": str, "severity": '
            '"Low|Med|High", "file": str, "detail": str, "confidence": '
            '"High|Med|Low", "fix": str}]}. fix = the test to add and what it asserts. '
            "If none, findings: []."
        ),
    },
    {
        "name": "Style Reviewer",
        "system": "You are a meticulous reviewer. Find maintainability issues: naming, duplication, complexity, missing error handling, dead code.",
        "query": "naming duplication complexity error handling dead code readability",
        "instruction": (
            'Return JSON: {"summary": str, "findings": [{"title": str, "severity": '
            '"Low|Med|High", "file": str, "detail": str, "confidence": '
            '"High|Med|Low", "fix": str}]}. If none, findings: []'
        ),
    },
]


def _context(retriever, query, k=6):
    hits = retriever.search(query, k=k)
    return "\n\n".join(f"# {h['file']}\n{h['text']}" for h in hits)


def _normalize_findings(data, agent_name):
    if not isinstance(data, dict):
        return [], "Agent returned no parseable output."
    findings = data.get("findings", []) or []
    clean = []
    for f in findings:
        if not isinstance(f, dict):
            continue
        clean.append(
            {
                "title": str(f.get("title", "Untitled finding")),
                "severity": str(f.get("severity", "Low")),
                "file": str(f.get("file", "—")),
                "detail": str(f.get("detail", "")),
                "confidence": str(f.get("confidence", "Med")),
                "fix": str(f.get("fix", "")),
                "agent": agent_name,
            }
        )
    return clean, str(data.get("summary", ""))


def _critic(retriever, findings):
    """Re-verify each finding against the code. Returns {index: verdict}."""
    if not findings:
        return {}
    parts = []
    for i, f in enumerate(findings):
        snippet = _context(retriever, f"{f.get('file')} {f.get('title')} {f.get('detail')}", k=3)
        parts.append(
            f"Finding #{i} (agent={f.get('agent')}, file={f.get('file')}, "
            f"severity={f.get('severity')}, confidence={f.get('confidence')}): "
            f"{f.get('title')} — {f.get('detail')}\nRelevant code:\n{snippet}"
        )
    user = (
        "\n\n".join(parts)
        + '\n\nVerify each finding against its code. Remove hallucinated/unsupported findings, '
        'downgrade overconfident ones. Return JSON: {"verdicts": [{"index": int, "verdict": '
        '"keep|downgrade|remove", "reason": str, "new_confidence": "High|Med|Low"}]}.'
    )
    data = chat_json(
        "You are a code review critic. You verify whether each claimed finding is actually "
        "supported by the code. You remove hallucinated findings and downgrade overconfident "
        "ones. Be strict.",
        user,
        temperature=0.1,
        max_tokens=1600,
    )
    verdicts = {}
    if isinstance(data, dict):
        arr = data.get("verdicts", []) or []
        for v in arr:
            if isinstance(v, dict) and "index" in v:
                verdicts[int(v["index"])] = v
    return verdicts


def _moderate(survivors, file_count):
    if not survivors:
        return (
            "## Summary\nNo findings survived the critic's verification — the code looked "
            "clean against what the agents could verify.\n\n"
            "## What I could NOT verify\nThe agents only reviewed the code they could "
            "retrieve; anything outside that context was not checked."
        )
    by_sev = {"High": [], "Med": [], "Low": []}
    for f in survivors:
        by_sev.setdefault(f.get("severity", "Low"), []).append(f)
    findings_text = "\n".join(
        f"- [{f.get('severity')}] {f.get('agent')}: {f.get('title')} ({f.get('file')}) — "
        f"{f.get('detail')} [conf: {f.get('confidence')}]. Fix: {f.get('fix')}"
        for f in survivors
    )
    user = (
        f"Analyzed {file_count} files. Surviving findings (after the critic verified them):\n"
        f"{findings_text}\n\nWrite the final code review report in markdown with sections: "
        "## Summary, ## Architecture, ## Security, ## Performance, ## Test Gaps, "
        "## Style & Maintainability, ## Top Recommendations, ## What I could NOT verify. "
        "Be concise, honest about confidence, and note anything unverified."
    )
    return chat(
        "You are a tech lead writing the final, synthesized code review report from the "
        "verified findings of a multi-agent review. You are honest about uncertainty.",
        user,
        temperature=0.3,
        max_tokens=1800,
    )


def stream_review(code=None, repo_url=None, language="python"):
    """Generator yielding event dicts (caller wraps them as SSE)."""
    files = load_source(code=code, repo_url=repo_url, language=language)
    chunks = chunk_files(files)
    retriever = Retriever(chunks)

    yield {"type": "start", "file_count": len(files), "chunk_count": len(chunks)}

    all_findings = []
    for r in REVIEWERS:
        yield {"type": "agent_start", "agent": r["name"]}
        ctx = _context(retriever, r["query"])
        user = f"Code:\n{ctx}\n\n{r['instruction']}"
        data = chat_json(r["system"], user, temperature=0.2, max_tokens=1600)
        findings, summary = _normalize_findings(data, r["name"])
        all_findings.extend(findings)
        yield {
            "type": "agent_done",
            "agent": r["name"],
            "summary": summary,
            "findings": findings,
        }

    # Critic — verify findings against code
    yield {"type": "critic_start"}
    verdicts = _critic(retriever, all_findings)
    survivors = []
    for i, f in enumerate(all_findings):
        v = verdicts.get(i)
        if v and v.get("verdict") == "remove":
            continue
        if v and v.get("verdict") == "downgrade" and v.get("new_confidence"):
            f["confidence"] = v["new_confidence"]
        f["critic_note"] = v.get("reason", "") if v else ""
        survivors.append(f)
    yield {
        "type": "critic_done",
        "total": len(all_findings),
        "kept": len(survivors),
        "removed": len(all_findings) - len(survivors),
    }

    # Moderator — synthesize the report
    yield {"type": "moderator_start"}
    report = _moderate(survivors, len(files))
    yield {"type": "report", "report": report, "findings": survivors, "file_count": len(files)}
    yield {"type": "done"}
