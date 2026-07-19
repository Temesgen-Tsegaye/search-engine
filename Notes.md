# RAG Search Engine — Notes

## Project Structure

```
rag-search-engine/
├── main.py                        # Entry point
├── data/
│   ├── movies.json                # Movie dataset (id, title, description)
│   └── stop_words.txt             # Stop words for keyword search
├── cli/
│   ├── semantic_search_cli.py     # CLI for semantic search + chunking
│   ├── keyword_search_cli.py      # CLI for keyword search
│   └── lib/
│       ├── semantic_search.py     # SemanticSearch, ChunkedSemanticSearch
│       ├── keyword_search.py      # InvertedIndex, BM25, TF-IDF
│       └── search_util.py         # Shared utilities (data loading, constants)
├── cash/                          # Cached embeddings & index files
└── cache/                         # Cached chunk embeddings
```

---

# Keyword Search

## Concept

Keyword search retrieves documents by exact term matching. It uses an **inverted index** — a mapping from each unique word (token) to the list of documents containing it.

### TF-IDF (Term Frequency — Inverse Document Frequency)

- **TF**: How often a term appears in a document. Higher = more relevant.
- **IDF**: How rare a term is across all documents. Rare terms get higher weight.
- **TF-IDF = TF × IDF** — balances local frequency with global rarity.

### BM25 (Best Matching 25)

An improved TF-IDF variant with saturation and length normalization:

- **BM25 TF**: `(tf * (k1+1)) / (tf + k1 * (1 - b + b * (docLen / avgDocLen)))`
  - `k1` (default 1.5): controls term frequency saturation.
  - `b` (default 0.75): controls document length normalization.
- **BM25 IDF**: `log((N - n + 0.5) / (n + 0.5))` where `N` = total docs, `n` = docs with term.
- **BM25 Score**: sum of BM25 TF-IDF across all query tokens.

## Code — `cli/lib/keyword_search.py`

### Tokenization (`toknize_text`)
```python
def toknize_text(text):
    text = clean_text(text)               # lowercase + remove punctuation
    stopwords = load_stop_words()
    tokens = [stemmer.stem(tok) for tok in text.split()
              if tok and tok not in stopwords]
    return tokens
```
- Cleans text, removes stop words, applies **Porter stemming** (e.g. "running" → "run").

### `class InvertedIndex`
| Method | Purpose |
|---|---|
| `__add_document(doc_id, text)` | Tokenizes text, updates index + term frequencies + doc lengths |
| `get_documents(term)` | Returns sorted doc IDs containing the term |
| `get_tf(doc_id, term)` | Raw term frequency for a doc |
| `get_idf(term)` | IDF = `log((N+1)/(n+1))` |
| `get_tfidf(doc_id, term)` | TF × IDF |
| `get_bm25_tf(doc_id, term)` | BM25 term frequency with k1, b |
| `get_bm25_idf(term)` | BM25 IDF = `log((N - n + 0.5)/(n + 0.5))` |
| `bm25_score(doc_id, term)` | BM25 IDF × BM25 TF |
| `bm25_search(query, top_k)` | Tokenizes query, sums BM25 scores, returns top-k |
| `build()` | Loads movies, indexes them |
| `save()` / `load()` | Pickles index to `cash/` |

### CLI — `cli/keyword_search_cli.py`
| Command | Action |
|---|---|
| `build` | Build & cache inverted index from movies.json |
| `search <query>` | Boolean keyword search (returns docs matching any query token) |
| `bm25search <query>` | Full BM25 ranked search |
| `tf <doc_id> <term>` | Show term frequency for a doc |
| `idf <term>` | Show inverse document frequency |
| `tfidf <doc_id> <term>` | Show TF-IDF score |
| `bm25tf <doc_id> <term>` | Show BM25 TF score |
| `bm25idf <term>` | Show BM25 IDF score |

### Example
```bash
python cli/keyword_search_cli.py bm25search "police officer"
# Returns ranked list: title + BM25 score
```

---

# Semantic Search

## Concept

Semantic search understands meaning, not just keywords. It converts text into **dense vector embeddings** (high-dimensional numerical representations) using a transformer model like `all-MiniLM-L6-v2`.

### Key Ideas

- **Embedding**: A vector (384-dim for MiniLM) that captures semantic meaning. Similar texts have similar vectors.
- **Cosine Similarity**: Measures the angle between two vectors:
  ```
  cos_sim(A, B) = (A · B) / (||A|| × ||B||)
  ```
  Range: -1 to 1. Higher = more semantically similar.
- **Chunking**: Long documents are split into sentence-based chunks, each embedded separately, for finer-grained search.

## Code — `cli/lib/semantic_search.py`

### `class SemanticSearch`

| Method | Purpose |
|---|---|
| `generate_embeddings(text)` | Encodes text into a 384-dim vector |
| `build_embeddings(documents)` | Encodes all `"title: description"` strings, saves to `cash/` |
| `load_or_create_embeddings(documents)` | Loads cached `.npy` or builds fresh |
| `search(query, limit)` | Embeds query, cosine-similarity against all docs, returns top-k |

### `cosine_similarity`
```python
def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0
```

### `class ChunkedSemanticSearch` (extends `SemanticSearch`)

Breaks each document into **sentence chunks** for finer retrieval.

**Chunking logic** (`build_chunk_embeddings`):
1. Strip whitespace; skip empty docs.
2. Split sentences on `[.!?]` followed by whitespace.
3. If only one sentence without ending punctuation, keep the whole text.
4. Group sentences into chunks of 4, with 1 sentence overlap.
5. Strip each sentence; discard empty ones.
6. Embed every chunk using the transformer model.
7. Cache results to `cache/`.

**Search** (`search_chunks`):
1. Embed the query.
2. Score every chunk via cosine similarity.
3. Per movie, keep only its best chunk score.
4. Return top-k movies ranked by that best chunk score.

### Edge Case Handling in Chunking
| Input | Result |
|---|---|
| `" Leading and trailing spaces. "` | Stripped to `["Leading and trailing spaces."]` |
| `"Text without punctuation"` | Kept as one sentence |
| `" "` (whitespace only) | Returns empty list |
| `""` (empty) | Returns empty list |

### CLI — `cli/semantic_search_cli.py`
| Command | Action |
|---|---|
| `embed <text>` | Generate & display embedding for text |
| `embed_query <query>` | Generate query embedding |
| `search <query>` | Semantic search across movies (full doc) |
| `verify` | Verify embeddings are loaded/created |
| `embed_chunks` | Build chunked embeddings for all movies |
| `search_chunked <query>` | Search across chunked embeddings |
| `chunk <text>` | Word-based chunking (simple split) |
| `semantic_chunk <text>` | Sentence-based semantic chunking |

### Example
```bash
python cli/semantic_search_cli.py search "a comedy about a teddy bear"
# Returns movies ranked by semantic similarity
```

---

# Comparing Keyword vs Semantic Search

| Aspect | Keyword Search | Semantic Search |
|---|---|---|
| **Matching** | Exact token match | Semantic meaning |
| **Model** | Porter stemmer + BM25 | `all-MiniLM-L6-v2` (transformer) |
| **Representation** | Sparse inverted index | Dense vector embeddings |
| **Handles typos** | No (unless stemmed) | Yes (similar context) |
| **Handles synonyms** | No | Yes ("car" ↔ "automobile") |
| **Speed** | Fast (lookup) | Slower (vector computation) |
| **Storage** | Pickled index (small) | .npy embedding files (larger) |
| **Ranking** | BM25/TF-IDF score | Cosine similarity |

---

# Running the Project

```bash
# Keyword search
python cli/keyword_search_cli.py build
python cli/keyword_search_cli.py bm25search "police officer"

# Semantic search
python cli/semantic_search_cli.py search "a teddy bear comedy"
python cli/semantic_search_cli.py embed_chunks
python cli/semantic_search_cli.py search_chunked "friendship story"
python cli/semantic_search_cli.py semantic_chunk "Hello world. This is a test."
```
