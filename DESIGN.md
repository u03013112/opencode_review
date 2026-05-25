# OpenCode Session Retrospective — Design Document

## Overview

A Python CLI tool that retrospectively analyzes OpenCode AI coding sessions to extract knowledge, surface failure patterns, and produce actionable improvement artifacts (skill drafts, KB entries, pattern documentation).

**Core Value Proposition:** Turn 990+ accumulated sessions into a reusable knowledge base, so similar problems never require re-discovery.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       CLI (click)                                │
│  retro list | retro analyze | retro report | retro status       │
└─────────┬───────────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────────────┐
│  Pipeline Stages (sequential per session)                        │
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │ Extract  │──▶│  Chunk   │──▶│ Analyze  │──▶│  Report  │    │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘    │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  State Manager (incremental processing, skip analyzed)    │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
          │
┌─────────▼────────────────────┐
│  Data Source                  │
│  SQLite: opencode.db          │
│  Session → Message → Part    │
└──────────────────────────────┘
```

---

## Project Structure

```
opencode_review/
├── src/
│   └── opencode_review/
│       ├── __init__.py
│       ├── cli.py              # Click CLI entry point
│       ├── db.py               # SQLite reader, query helpers
│       ├── extractor.py        # Session → normalized conversation
│       ├── chunker.py          # Semantic chunking logic
│       ├── analyzer.py         # LLM analysis calls
│       ├── reporter.py         # Markdown report generation
│       ├── state.py            # Incremental state tracking
│       └── models.py           # Dataclasses: Session, Chunk, Analysis
├── prompts/
│   ├── chunk_boundary.txt      # Prompt: topic boundary detection
│   └── block_analysis.txt     # Prompt: quality + failure analysis
├── output/
│   ├── reports/                # Per-session Markdown reports
│   └── artifacts/              # Extracted skills, KB entries
├── .state/
│   └── processed.json          # Incremental processing state
├── config.yaml                 # LLM model, thresholds, DB path
├── pyproject.toml
├── DESIGN.md
└── README.md
```

---

## Data Model

```python
@dataclass
class SessionMeta:
    id: str
    parent_id: str | None       # None = top-level session
    created_at: datetime
    message_count: int
    has_children: bool          # Has subagent sessions
    title: str | None           # First user message excerpt (50 chars)
    project_path: str | None    # Working directory

@dataclass
class NormalizedTurn:
    role: str                   # "user" | "assistant" | "tool_result"
    content: str                # Cleaned text (see extraction rules)
    turn_index: int
    has_tool_calls: bool
    tool_names: list[str]       # e.g. ["bash", "read", "write"]

@dataclass
class SemanticChunk:
    chunk_id: str
    session_id: str
    turns: list[NormalizedTurn]
    start_index: int
    end_index: int
    token_count: int

@dataclass
class ChunkAnalysis:
    chunk_id: str
    topic_summary: str
    outcome: Literal["success", "partial", "failure", "unclear"]
    first_try_success: bool
    corrections_required: int
    skills_referenced: list[str]
    kb_referenced: list[str]
    failure_root_cause: str | None  # "missing_skill" | "unclear_instruction" | "agent_limit" | "knowledge_gap"
    failure_detail: str | None
    missing_context: str | None
    recommendations: list[str]
    confidence: float
```

---

## Stage 1: Data Extraction

### The 2GB Problem

Total data is ~2.5GB, but 95% is tool call content (file reads, bash outputs). An LLM doesn't need full tool output to assess quality — it needs to know *what happened*.

### Part Type Handling

| Part Type | Size | Action | Rationale |
|-----------|------|--------|-----------|
| `text` | 41MB | ✅ Include fully | Core conversation, always needed |
| `reasoning` | 7.6MB | ⚠️ Truncate to 500 chars | Reveals agent intent without verbosity |
| `step-start/finish` | 30MB | ✅ Extract tool name only | Structural signal |
| `tool` | 2.1GB | ❌ Summarize to 1 line | **This is the data volume problem** |
| `patch` | 4MB | ⚠️ Diff stats only | `+N/-N lines in file.py` |
| `file` | 390MB | ❌ Filename + size only | Content not needed for quality assessment |
| `compaction` | 56KB | ❌ Skip | Internal compression artifacts |

### Tool Result Summarization

For each tool call, extract a 1-line summary:
```
bash(command="npm test") → exit_code=1, stderr_lines=12 (FAIL)
read(file="src/auth.ts") → 340 lines
write(file="src/auth.ts") → success
task(subagent="explore", desc="Find auth") → completed, 5 results
```

**Estimated output:** 990 sessions × ~3-15K tokens extracted = manageable for LLM.

### Subagent Handling

**Strategy: Inline subagent sessions into parent context.**

Subagent work is logically part of the parent task. Analyzing separately loses the "why" (parent context). Inline preserves the full causal chain.

```python
def flatten_session(session_id: str, db: DB, depth: int = 0) -> list[NormalizedTurn]:
    if depth > 2:  # Cap recursion to avoid explosion
        return [NormalizedTurn(role="system", content="[subagent depth exceeded, skipped]", ...)]
    
    turns = get_session_turns(session_id)
    for i, turn in enumerate(turns):
        if turn.spawns_subagent:
            child_turns = flatten_session(turn.child_session_id, db, depth + 1)
            # Insert with [SUBAGENT:name] prefix for context
            turns = turns[:i+1] + child_turns + turns[i+1:]
    return turns
```

---

## Stage 2: Semantic Chunking

### Goal

Split a session into coherent **task blocks**. Each block = one user intent + agent execution cycle.

### Algorithm: Hybrid Rule + LLM

**Phase 1: Rule-based pre-segmentation (free, fast)**

Boundary signals (any triggers a candidate boundary):
- User message after assistant "completion" signal ("done", "let me know", "complete")
- Explicit topic transition: "now let's", "next", "different question", "can you also"
- Tool pattern reset: completely different tool sequence starts
- Time gap > 30 minutes between turns
- User sends new high-level instruction after a completed sub-task

**Phase 2: LLM boundary refinement (for ambiguous cases only)**

Only invoke LLM when rule-based produces chunks > 8000 tokens or < 3 turns:

```
Prompt (chunk_boundary.txt):
Given this conversation excerpt, identify where the user's intent shifts to a new distinct task.
Return JSON: {"boundaries": [turn_index, ...], "confidence": 0.0-1.0}
Only split if clearly distinct topics. When in doubt, keep as one chunk.
```

**Target chunk size:** 20–80 turns, ~2000–6000 tokens per chunk after extraction.

**Sessions with <10 turns:** Treat as single chunk, skip boundary detection.

### Technology Options

| Library | Use Case | When to Use |
|---------|----------|-------------|
| Rule-based (built-in) | Primary chunker | Always (phase 1) |
| LLM (gpt-4o-mini) | Boundary refinement | Oversized/undersized chunks |
| `semantic-text-splitter` | Fallback for pure-text sessions | If rules fail |

---

## Stage 3: LLM Analysis

### Per-Chunk Analysis Prompt

```
You are analyzing a conversation block from an AI coding assistant session.

CONVERSATION:
{normalized_turns}

CONTEXT:
- Total turns in block: {turn_count}
- Tools used: {tool_summary}
- Session date: {date}
- Available skills at time: {skill_list}

TASK: Analyze this block and return JSON:
{
  "topic": "one sentence description of what was attempted",
  "outcome": "success|partial|failure|unclear",
  "first_try_success": true|false,
  "corrections_required": N,
  "skills_referenced": ["skill_name"],
  "kb_referenced": ["kb_name"],
  "failure_root_cause": null | "missing_skill|unclear_instruction|agent_limit|knowledge_gap",
  "failure_detail": "one sentence explaining what went wrong",
  "missing_context": "what info/skill/pattern would have enabled first-try success",
  "recommendations": ["actionable item, max 3"],
  "confidence": 0.0-1.0
}

RULES:
- first_try_success = true ONLY if agent completed the task without user sending corrections
- User corrections include: "that's wrong", "no I meant", "fix the", "actually", re-stating the same request
- outcome = "success" requires task fully completed to user satisfaction
- outcome = "partial" = eventually completed but after corrections
- outcome = "failure" = never completed or user gave up
- missing_context should be SPECIFIC (e.g. "skill for lark-base field formula syntax" not "better context")
- Ignore stylistic preferences ("I prefer X") — only flag substantive failures
```

### Model Selection Strategy

| Analysis Type | Model | Cost per 1K input tokens |
|---------------|-------|--------------------------|
| Chunk boundary refinement | gpt-4o-mini / claude-haiku | $0.15/M |
| Primary block analysis | gpt-4o-mini | $0.15/M input, $0.60/M output |
| Complex failure deep-dive (escalation) | claude-sonnet / gpt-4o | $3/M input |

**Default: gpt-4o-mini for everything.** Escalate to sonnet only for chunks where confidence < 0.5.

---

## Stage 4: Report Generation

### Per-Session Report (`output/reports/{session_id}.md`)

```markdown
# Session Analysis: {title}
**Date:** {date} | **Duration:** {duration} | **Messages:** {count} | **Project:** {path}

## Summary
{2-3 sentence session overview}

## Blocks

### Block 1: {topic}
- **Outcome:** ✅ Success (first try)
- **Tools:** bash, read, write
- **Turns:** 12
- **Skills Used:** lark-im

> {brief description}

---

### Block 2: {topic}
- **Outcome:** ⚠️ Partial (2 corrections)
- **Tools:** bash, lark-cli
- **Turns:** 23
- **Skills Used:** none

> {description}

**Root Cause:** Missing skill — no documented workflow for X
**What Was Missing:** A skill covering Y procedure with Z specifics
**Recommendations:**
1. Create skill `lark-Y` documenting the Z workflow
2. Add KB entry for X pattern

---

## Quality Scorecard
| Metric | Value |
|--------|-------|
| Blocks analyzed | 3 |
| First-try success rate | 67% (2/3) |
| Failures | 0 |
| Skills referenced | lark-im, lark-base |
| Knowledge gaps found | 1 |

## Extracted Recommendations
1. **[New Skill]** Create `lark-Y` for Z workflow
2. **[KB Entry]** Document X pattern in knowledge base
```

### Aggregate Report (`output/reports/weekly_{date}.md`)

Weekly summary across all sessions processed in that run:
- Overall first-try success rate
- Top failure root causes (ranked by frequency)
- Most-referenced skills (usage vs. failure correlation)
- Recurring knowledge gaps (cluster similar `missing_context`)
- Priority recommendations (impact × frequency)

---

## Stage 5: Incremental State

### State File: `.state/processed.json`

```json
{
  "schema_version": 1,
  "last_run": "2026-05-25T10:00:00Z",
  "sessions": {
    "ses_abc123": {
      "analyzed_at": "2026-05-20T08:00:00Z",
      "message_count_at_analysis": 47,
      "chunk_count": 3,
      "report_path": "output/reports/ses_abc123.md",
      "outcome_summary": {"success": 2, "partial": 1, "failure": 0}
    }
  }
}
```

### Processing Decision Logic

```python
def should_analyze(session_id: str, current_msg_count: int, state: dict) -> bool:
    prev = state["sessions"].get(session_id)
    if prev is None:
        return True  # Never analyzed
    if current_msg_count > prev["message_count_at_analysis"]:
        return True  # Session grew (user continued it)
    return False     # No change, skip
```

### Weekly Workflow

```bash
# Every week (or when you feel like it):
retro analyze --since 7d    # Only processes new/changed sessions
retro report --since 7d     # Generates weekly aggregate
```

Old sessions → already analyzed, skipped automatically.
Updated sessions → re-analyzed in full (re-chunking is cheap, don't try to diff).

---

## Cost Estimates

### Per-Session Token Budget

| Component | Tokens (est.) |
|-----------|---------------|
| Extracted turns (text only, after summarization) | 3,000–15,000 |
| Tool call summaries | 500–2,000 |
| Per-chunk analysis prompt overhead | ~400 |
| **Total input per session (avg)** | ~8,000 |
| **Output per chunk** | ~300–500 |

### Total Cost Projections

| Scenario | Sessions | Estimated Cost (gpt-4o-mini) |
|----------|----------|------------------------------|
| Full backfill (all 990) | 990 | ~$1.50 |
| Weekly incremental (~50 new) | 50 | ~$0.08 |
| Single session re-analysis | 1 | ~$0.002 |

**Extremely cheap.** No need for aggressive cost optimization at this scale.

### Optimization Levers (if needed later)

1. **Skip trivial sessions:** <5 messages → mark as "trivial", no LLM call
2. **Pre-filter for failures:** Run cheap heuristic (user message contains "wrong"/"fix"/"no") before LLM
3. **Batch similar chunks:** Group chunks by tool patterns, analyze in batch
4. **Cache embeddings:** If using semantic chunker, cache to avoid re-embedding

---

## CLI Interface

```bash
# List sessions with metadata (for manual selection)
retro list [--limit 20] [--since 7d] [--unanalyzed-only] [--project PATH]

# Analyze specific sessions
retro analyze <session_id> [session_id ...]
retro analyze --all                    # All unprocessed
retro analyze --since 7d               # New sessions from last 7 days

# Generate reports
retro report [--since 7d] [--output weekly_report.md]
retro report --aggregate               # Cross-session summary

# State management
retro status                           # Show processed/pending counts
retro reset <session_id>               # Force re-analysis
retro reset --all                      # Reset everything
```

---

## Configuration (`config.yaml`)

```yaml
db_path: ~/.local/share/opencode/opencode.db
output_dir: ./output
state_dir: ./.state

llm:
  provider: openai              # openai | anthropic
  model: gpt-4o-mini           # Primary analysis model
  escalation_model: gpt-4o     # For low-confidence re-analysis
  temperature: 0
  max_tokens_per_call: 2000

extraction:
  max_reasoning_chars: 500
  max_tool_summary_chars: 200
  subagent_depth_limit: 2
  skip_sessions_under_messages: 5

chunking:
  target_min_turns: 3
  target_max_tokens: 6000
  time_gap_boundary_minutes: 30
  use_llm_refinement: true

analysis:
  confidence_threshold: 0.5    # Below this → escalate to stronger model
  skip_trivial: true           # Skip sessions with <5 messages
```

---

## Open Design Questions (For Discussion)

1. **Subagent detail level:** Should subagent turns be fully inlined or summarized to "subagent did X, result: Y"? Full inline gives better failure detection but increases token count.

2. **Skill availability tracking:** To detect "missing skill" properly, we need to know which skills were available at the time of the session. Should we track this in state, or infer from file system history?

3. **Human-in-the-loop:** After analysis, should there be a "review & confirm" step where you approve/reject/edit the analysis before it's finalized? Or trust the LLM output?

4. **Artifact generation:** Should the tool auto-generate skill stubs / KB entries, or just recommend them in the report for manual creation?

5. **Session selection UX:** For `retro list`, what metadata helps you decide which sessions to analyze? Date, duration, message count, first user message? Should there be a TUI picker?
