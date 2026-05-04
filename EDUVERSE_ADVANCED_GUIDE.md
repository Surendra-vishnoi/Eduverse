# Eduverse: Advanced Technical Deep Dive & Edge Cases

## Table of Contents
1. [Advanced Retrieval Techniques](#advanced-retrieval-techniques)
2. [LangGraph Internal Mechanics](#langgraph-internal-mechanics)
3. [Error Handling & Recovery](#error-handling--recovery)
4. [Performance Optimization](#performance-optimization)
5. [Security Edge Cases](#security-edge-cases)
6. [Production Lessons Learned](#production-lessons-learned)
7. [Tricky Interview Questions](#tricky-interview-questions)

---

# ADVANCED RETRIEVAL TECHNIQUES

## Query Rewrite LLM Prompt Engineering

**Current Implementation**:
```
"Rewrite the student's question into a concise, keyword-rich search query."
```

**Why This Matters**:
- If prompt is too generic, LLM output is unpredictable (sometimes returns whole sentences).
- If prompt is too specific, LLM over-optimizes and loses intent.

**Example Failure Cases**:

1. **User**: "Can you explain photosynthesis?"
   **Bad Rewrite**: "photosynthesis" (too short, loses "explain" intent)
   **Good Rewrite**: "photosynthesis explanation light reactions Calvin cycle"

2. **User**: "Is glucose a type of sugar?"
   **Bad Rewrite**: "glucose sugar" (loses question-ness)
   **Good Rewrite**: "glucose classification types sugars carbohydrate"

**Interview Q: How do you test query rewrite quality?**
A: Metrics:
1. Precision: Does rewritten query retrieve relevant chunks for original question?
2. Diversity: Does rewrite add keywords vs just copy-pasting?
3. Conciseness: Is rewrite <150 tokens (efficient for LLM)?

Manual evaluation on 100+ question pairs with human feedback.

## Embedding Dimensionality

**Current**: 768 dimensions (Nomic).

**Why not 1536** (OpenAI):
- Larger vectors = more storage + slower search.
- 768-dim is proven to work for educational use case.

**Why not 384** (smaller):
- Lower quality for semantic understanding.
- Embedding dimension correlates with downstream task performance.

**How to Choose**:
1. Start with 768.
2. If retrieval is poor, try 1536 (better quality, higher cost).
3. If performance is bottleneck, try 384 (faster, risk lower quality).

## Semantic vs Keyword Search Failure Modes

**When Vector Fails**:
```
Query: "exam date"
Vector Search: No relevant results (word "exam" weak semantic signal)
FTS: Finds "The exam is on December 15th"
```

**When FTS Fails**:
```
Query: "what does this course teach?"
FTS: Matches "teach" keyword in irrelevant chunks
Vector: Understands intent, finds course overview
```

**How RRF Handles Both**:
- If either retriever succeeds, merged result likely good.
- If both agree on a doc, it gets boosted.
- If they disagree, doc gets moderate score.

## Reranking Strategy

**Why FlashRank is Optional**:

Benefit of reranking:
- Cross-encoder model is stronger than retrieval heuristics.
- Catches false positives from hybrid retrieval.
- Improves recall-@5 by ~10%.

Cost of reranking:
- 100-200ms latency (cross-encode 10-20 docs).
- API call or local model inference.

**Interview Q: When would you enable reranking in production?**
A: If:
1. Answer accuracy is critical (e.g., exam prep).
2. User is willing to wait extra 100ms.
3. Budget allows for extra API calls.

Enable it. Otherwise, RRF + dedup is sufficient.

## Context Expansion Heuristics

**Current Strategy**: Store prev + next chunk in metadata.

**Why 3-chunk window?**
- Too small (1 chunk): lacks context, answers might be wrong.
- Too large (5+ chunks): token budget explodes, slower inference.
- 3-chunk: sweet spot balances richness vs cost.

**Edge Case: Chunk at Boundary**
```
Chunk 1 (no prev): [NULL, "content", "next chunk"]
Chunk N (no next): ["prev chunk", "content", NULL]
```
Handled: `if before or after` filters nulls gracefully.

---

# LANGGRAPH INTERNAL MECHANICS

## State Update Semantics

**In LangGraph, state updates are NOT simple dict merges.**

```python
# Node returns update dict
return {"messages": [new_message], "status": "processing"}

# Graph applies update based on reducer
# Default behavior: Replace old value with new
# For messages list: Append (has built-in reducer)
```

**Implications**:
- If node returns `{"status": "done"}`, old status is overwritten.
- If node returns `{"messages": [msg1, msg2]}`, it APPENDS to existing messages (not replaces).

**Interview Q: What if node needs to reset messages?**
A: Use explicit reducer or return special value. By default, can't clear list; must work within append semantics.

## Checkpointing & Resumption

**Save Point**: After each node execution, full state is saved to DB.

**Resume Logic**:
```python
# First call
result = graph.invoke(input, config={"thread_id": "s1"})
# State saved to checkpoint table

# Later call with same thread_id
result = graph.invoke(new_input, config={"thread_id": "s1"})
# Graph loads last checkpoint
# Appends new input to existing messages
# Resumes from there
```

**Edge Case: What if we want to restart?**
```python
# Option 1: Delete checkpoint
checkpointer.delete_thread("s1")

# Option 2: Use different thread_id
new_id = f"s1_{uuid.uuid4()}"
result = graph.invoke(input, config={"thread_id": new_id})
```

## Conditional Edge Semantics

**Syntax**:
```python
graph.add_conditional_edges(
    "download",
    should_continue,  # Function returning string
    {
        "continue": "process",
        "error": "handle_error"
    }
)
```

**Execution**:
1. After node "download" completes, function `should_continue(state)` is called.
2. Function returns one of the dict keys ("continue" or "error").
3. Based on return value, graph routes to next node.

**Interview Q: What if should_continue throws exception?**
A: Exception propagates. Graph execution fails. Design:Always have safe fallback path in should_continue (never throw).

## Middleware Chain Execution

**Order**:
```
User Input
  ↓
Middleware 1 pre-processing
  ↓
Middleware 2 pre-processing
  ↓
Middleware 3 pre-processing
  ↓
Core Agent Loop
  ↓
Middleware 3 post-processing
  ↓
Middleware 2 post-processing
  ↓
Middleware 1 post-processing
  ↓
Return Result
```

**Implication**: If Middleware 1 detects error in post-processing, can it stop Middleware 2 pre-processing? No. Middleware runs in sequence.

**Interview Q: What if summarization runs while retry is retrying?**
A: Both middlewares are stateless and don't interact. Summarization summarizes the full message history (including retries). Clean separation.

---

# ERROR HANDLING & RECOVERY

## Indexing Failure Paths

### Failure in download_node

**Causes**:
- Credentials expired.
- File deleted from Drive.
- Network timeout.

**Recovery**:
```python
# Node catches, returns status=failed
return {"status": "failed", "error": "Credentials expired"}

# should_continue routes to handle_error
# handle_error_node updates DB: processing_status="failed"

# User sees failed status in UI
# Calls retry endpoint → restarts graph with same file_id
```

### Failure in embed_node

**Causes**:
- Nomic API down.
- Rate limit exceeded.
- Vector dimension mismatch.

**Recovery**:
```python
# embed_node catches, returns status=failed
# DB marked failed
# Could backoff and retry
# Or user manually retries
```

**Improvement**: Add exponential backoff inside embed_node, retry 3x before giving up.

## Chat Agent Failure Paths

### Tool Execution Failure

```python
# Tool throws exception
try:
    tool_result = search_course_materials(query)
except Exception as e:
    # ModelRetryMiddleware catches
    # Retries tool call up to 3x
```

**If retry exhausted**: Exception propagates, caught by agent error handler, agent responds: "I couldn't search materials, but..." then continues.

### LLM API Failure

```python
# Groq API returns 500
try:
    response = llm.invoke(messages)
except APIError as e:
    # ModelRetryMiddleware catches
    # Retries LLM call
```

**If Groq is down**: After 3 retries, exception to user: "Service temporarily unavailable."

## Cascade Failure Prevention

**Potential Cascade**:
```
Nomic Embedding API down
  ↓
Indexing stalls
  ↓
Connection pool fills with waiting tasks
  ↓
Database connection exhaustion
  ↓
Chat API also starves for connections
  ↓
Entire app down
```

**Mitigation**:
1. Semaphore(3): Only 3 indexing jobs; rest queue (don't hold connections).
2. Timeouts: API calls with explicit timeout.
3. Circuit breaker: After N failures, stop calling Nomic temporarily, fail-fast.

## Interview Q&A

**Q: What's the difference between retry and recovery?**
A: Retry = try same operation again. Recovery = try different operation or degrade gracefully. Example: Retry Groq call on timeout. Recover from Groq outage by using cheaper fallback model.

**Q: How do you know if a failure is transient vs permanent?**
A: Heuristics:
- 429/500/503 (rate limit, server error) → transient, retry.
- 400/401/403 (client error, auth) → permanent, don't retry.
- Timeout after 5s → likely transient, retry once.

**Q: What if a user retries an indexing job that keeps failing?**
A: Same failure happens again, file marked failed again. Loop. Better: show user error message, suggest contacting support vs auto-retry.

---

# PERFORMANCE OPTIMIZATION

## Bottleneck Analysis

### Indexing Pipeline Latencies

For a 50-page PDF:

```
download_node:  3-5s (network)
process_node:   5-10s (PDF parse + vision API if images)
chunk_node:     1s (semantic chunking)
embed_node:     30-60s (50 pages * 5-10 chunks/page = 250-500 chunks, Nomic @ ~100-200ms per batch)
update_db_node: 1-2s (DB writes)
---
Total: 40-80s
```

**Bottleneck**: embed_node (Nomic API).

**Optimization Options**:
1. Batch embeddings (send 10 chunks at once). Saves ~30% time.
2. Use local embedding model (quantized). Saves API latency but requires GPU.
3. Async embeddings (start indexing while user continues, show progress).

### Chat Query Pipeline Latencies

For a chat query:

```
Query Rewrite:   500ms (Groq API call)
Vector Retrieval: 50-100ms (pgvector HNSW search)
FTS Retrieval:    100-200ms (PostgreSQL FTS)
RRF Merge:        10ms (in-memory sort)
Reranking:        100-200ms (if enabled)
Context Expansion: 10ms (metadata lookup)
Dedup:            5ms (set operations)
Agent Loop:
  ├─ Tool call + retrieval: 500ms (overlap with query rewrite)
  ├─ LLM call: 2-5s (Groq stream generation)
  └─ Post-processing: 100ms
---
Total (streaming): 3-7s to first token, 5-10s total
Total (non-stream): 5-15s
```

**Bottleneck**: LLM generation (Groq).

**Optimization Options**:
1. Use faster/smaller model (cheaper API, lower quality).
2. Caching: Cache Q&A for repeat questions.
3. Speculative execution: Guess likely answer while waiting for LLM.

## Database Optimization

### Index Tuning

```sql
-- Vector search index (HNSW)
CREATE INDEX idx_embedding_hnsw 
ON langchain_pg_embedding 
USING hnsw (embedding vector_cosine_ops) 
WITH (m=16, ef_construction=64);

-- Parameters tuning for large datasets:
-- m=16 → m=32 (more neighbors, better recall, slower inserts)
-- ef_construction=64 → ef_construction=128 (more thorough building, slower but better search)
```

### Query Optimization

**Current FTS Query**:
```sql
SELECT e.document, e.cmetadata, ts_rank(...) AS rank
FROM langchain_pg_embedding e
JOIN langchain_pg_collection c ON e.collection_id = c.uuid
WHERE c.name = :collection
  AND to_tsvector('english', e.document) @@ websearch_to_tsquery(...)
```

**Optimization**: Pre-compute tsvector column.
```sql
ALTER TABLE langchain_pg_embedding ADD COLUMN fts_doc tsvector;
CREATE INDEX idx_fts_doc ON langchain_pg_embedding USING gin (fts_doc);

-- Then query becomes simpler, faster
```

### Connection Pool Sizing

**Current**: pool_size=5, max_overflow=10 (15 max total).

**How to Right-Size**:
1. Estimate concurrent requests: N.
2. Per-request DB connections: 1-2 (briefly held).
3. pool_size = N / 3 (keep some reserved).
4. max_overflow = N (burst capacity).

**For 100 concurrent users**: pool_size=30, max_overflow=100.

## Caching Strategy

### Query Response Caching

**Idea**: Cache "What is photosynthesis?" answer for 1 hour.

**Implementation**:
```python
cache_key = hashlib.sha256(f"{user_id}_{query}_{course_id}".encode()).hexdigest()

# Check cache
if cache.get(cache_key):
    return cache.get(cache_key)

# Generate answer
answer = agent_invoke(...)

# Cache for 1 hour
cache.set(cache_key, answer, ttl=3600)
return answer
```

**Tradeoff**: 
- Pro: 1000x faster repeated queries.
- Con: Cached answers become stale if course materials update.

**Mitigation**: Invalidate cache on file upload/update.

### Retrieval Caching

**Idea**: Cache retriever object (avoid rebuilding per request).

**Current**: Retriever rebuilt per request.

**Improvement**:
```python
_retriever_cache: dict[str, Retriever] = {}

def get_retriever(user_id, course_id):
    key = f"{user_id}_{course_id}"
    if key not in _retriever_cache:
        _retriever_cache[key] = build_retriever(...)
    return _retriever_cache[key]
```

**Invalidation**: Clear cache when vectors updated (embedding completes).

---

# SECURITY EDGE CASES

## Token Theft Scenarios

### Scenario 1: XSS Injection
```
Attacker injects JS in course material (e.g., PDF embedded JS)
JS runs in browser context
JS steals localStorage (API key, but not tokens due to HttpOnly)
Attacker sends requests with stolen API key
```

**Defense**:
- Sanitize all user inputs (course materials).
- Use Content-Security-Policy header.
- Avoid eval(), innerHTML, etc.

### Scenario 2: Man-in-the-Middle (MITM)
```
User on public WiFi
Attacker intercepts traffic
Cookie captured
Attacker uses stolen cookie
```

**Defense**:
- HTTPS enforced (Secure flag on cookies).
- HSTS header (force HTTPS).
- Certificate pinning (for mobile apps).

### Scenario 3: Token Stored in localStorage
```
If frontend stores JWT in localStorage (design flaw):
JS can access it
XSS can steal it
```

**Current Design**: Tokens in HttpOnly cookies only. Safe.

## Authorization Bypass Scenarios

### Scenario 1: Session ID Guessing
```
Session format: user_{user_id}_{12_random_hex}
Attacker knows user_id (public)
Tries to guess 12 hex chars (16^12 combinations)
Brute force: ~281 trillion attempts
```

**Verdict**: Infeasible brute force, but weak if attacker knows format.

**Improvement**: Use full UUID v4 (2^128), not hex.

### Scenario 2: CSRF Attack
```
Attacker tricks user into clicking link on attacker site
Link makes POST to /chat/delete-session
User's browser automatically sends cookies
Delete happens
```

**Defense**: CSRF token (double-submit pattern). Attacker can't read cookie, so can't forge token header. POST blocked.

**Current Status**: CSRF protection implemented for all POST/DELETE.

### Scenario 3: Privilege Escalation
```
User A tries to access User B's session
/chat/history/{user_b_session_id}
```

**Defense**: Ownership check (session_id must start with user_id).

**Current Status**: Implemented and checked.

## API Key Exposure

### Scenario: Groq Key Leaked
```
User sends Groq key via X-Groq-Api-Key header
Attacker intercepts (if not HTTPS)
Attacker uses key to make expensive API calls
Bill goes to user
```

**Defense**:
- HTTPS only (enforced).
- User can rotate key anytime.
- Alert on unusual API usage (if implemented).

**Edge Case**: What if backend logs header? (Bad!)

**Mitigation**: Never log headers containing API keys. Use structured logging; redact sensitive headers.

---

# PRODUCTION LESSONS LEARNED

## Lesson 1: Postgres Connection Starvation

**What Happened**: Indexing jobs ran concurrently without limit. Each held DB connection. Connection pool exhausted. App hung.

**Fix**: Implement Semaphore(3) for concurrent indexing.

**Lesson**: Always limit concurrency on external resources.

## Lesson 2: Embedding API Rate Limits

**What Happened**: User uploaded 50 files. All 50 try to embed simultaneously. Nomic API returns 429 (too many requests). All indexing fails.

**Fix**: Semaphore throttles to 3 concurrent embeds. Rest queue.

**Lesson**: Assume external APIs have rate limits. Design for backoff.

## Lesson 3: Supabase Idle Connection Timeout

**What Happened**: Chat session idle for 5 minutes. Next query fails with "connection reset". Checkpointer crashes.

**Fix**: Add keepalive settings to connection pool + retry logic.

**Lesson**: Cloud DBs kill idle connections. Handle gracefully.

## Lesson 4: Citation Cache Memory Leak

**What Happened**: Chat sessions accumulate. Citations cached forever. Memory grows unbounded. App runs out of RAM.

**Fix**: Add TTL (5 min) to cache entries + eviction.

**Lesson**: In-memory caches need explicit cleanup.

## Lesson 5: Tool Exception Silencing

**What Happened**: Tool threw exception. Agent silently continued with empty result. User saw incomplete answer. Silent failure, hard to debug.

**Fix**: Middleware logs exceptions. Agent's system prompt tells it to ask for retry if tool fails.

**Lesson**: Failed tool calls should be visible to user/logs, not silently ignored.

---

# TRICKY INTERVIEW QUESTIONS

## Q1: How do you ensure retrieval quality?

**Tricky Part**: Hard to measure automatically.

**Answer**:
1. **Metrics** (automated):
   - Recall: % of relevant docs retrieved (needs labeled data).
   - Precision: % of retrieved docs relevant.
   - MRR: Mean Reciprocal Rank (is best doc ranked first?).

2. **Human Evaluation** (manual):
   - Sample 100 queries.
   - Human judges: is each result relevant?
   - Evaluate retriever quality.

3. **Downstream Task** (practical):
   - Ask LLM to grade answers (self-critique).
   - Track user feedback ("Was this answer helpful?").

**Interview Signal**: Shows understanding that retrieval is critical, requires both metrics and human judgment.

## Q2: What happens when a user asks about something not in their course?

**Tricky Part**: Multiple valid answers.

**Answer**:
1. **Ideal**: Retrieval returns nothing. Agent says "Not in your materials" then provides general knowledge (labeled).

2. **Reality**: Retrieval might return partially relevant docs (confusing context).

3. **Handling**:
   - System prompt tells agent: "If chunks don't match topic, use file names to describe course."
   - Agent learns: if search fails, don't force-fit irrelevant context.

4. **Better**: Add ranking threshold. If top doc's similarity < 0.5, don't use it.

**Interview Signal**: Shows awareness of edge cases and iterative refinement.

## Q3: How do you prevent agent from calling tools infinitely?

**Tricky Part**: Agent is supposed to be agentic (can call tools), but needs stop condition.

**Answer**:
1. **Current**: ModelCallLimitMiddleware caps at 25 tool calls per invocation.

2. **Limitation**: Doesn't detect repeated calls. Agent could call same tool 25 times.

3. **Improvement**:
   - Detect repeated tool calls (call same tool with same args in last 3 calls?).
   - Log and alert.
   - Force stop.

4. **Better**: Add confidence signal. If tool returns same result 2x, stop calling.

**Interview Signal**: Shows deep thinking about agent design.

## Q4: What if LLM hallucinates a citation?

**Tricky Part**: LLM might claim [1] supports claim X when [1] actually says Y.

**Answer**:
1. **Current**: Rely on system prompt: "Only cite sources whose content supports your claim."

2. **Why It Fails**: LLM still hallucinates despite prompt.

3. **Detection**:
   - Extract claimed citations from answer.
   - Validate each cited chunk actually supports claim (semantic matching).
   - Alert if mismatch.

4. **Example**:
   ```
   Agent: "Photosynthesis uses light [1]"
   [1]: "Photosynthesis is the process of..."
   
   Validation: Does [1] mention light? No. Flag as potential hallucination.
   ```

5. **Fix**:
   - Re-prompt agent: "You cited [1] but it doesn't mention light. Revise."
   - Or filter citation from response.

**Interview Signal**: Shows understanding of RAG failure modes and validation techniques.

## Q5: How do you scale retrieval to 10M users?

**Tricky Part**: Per-user collections in Postgres won't scale infinitely.

**Answer**:
1. **Current Architecture**: Per-user collection in single Postgres table.

2. **Bottleneck at 10M users**:
   - Table too large for indexes to be efficient.
   - One user's query might touch millions of vectors (slow scan).
   - Connection pool exhaustion (10M users = many concurrent requests).

3. **Scaling Strategy**:
   ```
   Option 1: Postgres Partitioning
   - Partition by user_id range
   - user_1_to_1M (Postgres 1)
   - user_1M_to_2M (Postgres 2)
   - Each partition handles smaller query volume
   
   Option 2: Specialized Vector DB
   - Postgres good for 1M vectors per user
   - For 10B vectors total, migrate to Pinecone/Weaviate
   - More expensive but designed for scale
   
   Option 3: Denormalization
   - Cache popular queries
   - Pre-compute common retrievals
   - Reduce repeated searches
   ```

4. **Decision Tree**:
   - 1M users? Postgres with partitioning + caching.
   - 10M users? Vector DB + caching + query dedup.

**Interview Signal**: Shows architectural thinking, understanding tradeoffs.

## Q6: How do you monitor if agent is giving wrong answers?

**Tricky Part**: Automatic detection is hard.

**Answer**:
1. **Immediate**: User feedback ("Was this answer correct?").

2. **Passive Monitoring**:
   - Log all queries + answers + retrieval results.
   - Periodically audit:
     - Answers with low retrieval scores (might be hallucinations).
     - Queries with repeated follow-ups (user didn't understand answer, tried rephrasing).

3. **Active Monitoring**:
   - Sample answers, have human reviewer grade correctness.
   - Compute accuracy on graded sample.
   - Alert if accuracy < threshold.

4. **Automated Checks**:
   - Answer length (suspiciously short or very long?).
   - Citation coverage (all claims cited?).
   - Coherence (is answer self-consistent?).

**Interview Signal**: Shows thinking about production monitoring, quality assurance.

---

# Summary Table: Common Interview Traps

| Topic | Trap | Correct Answer |
|-------|------|---|
| **Retrieval** | "Just use vector search." | Hybrid (vector + FTS) is better for educational use case. |
| **Agent** | "Why not 100 tools?" | More tools = harder reasoning. 4-5 focused tools is better. |
| **Auth** | "Store JWT in localStorage?" | No. HttpOnly cookies are more secure against XSS. |
| **Scaling** | "Postgres can't scale?" | It can, but needs partitioning + optimization. At 10M users, consider specialized vector DB. |
| **Memory** | "Just keep all messages?" | No. Summarization middleware keeps history bounded. |
| **Error Handling** | "What if tool fails?" | Retry middleware retries 3x. If still fails, agent degrades gracefully. |

---

End of Advanced Guide.
