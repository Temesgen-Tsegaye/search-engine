import json
import os
from dotenv import load_dotenv
from openai import OpenAI


def _get_client() -> OpenAI:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def correct_spelling(query: str) -> str:
    client = _get_client()
    prompt = (
        f"Fix any spelling errors in the user-provided movie search query below.\n"
        f"Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.\n"
        f"Preserve punctuation and capitalization unless a change is required for a typo fix.\n"
        f"If there are no spelling errors, or if you're unsure, output the original query unchanged.\n"
        f"Output only the final query text, nothing else.\n"
        f'User query: "{query}"'
    )
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def rerank_individual(query: str, doc: dict) -> float:
    client = _get_client()
    prompt = (
        f"Rate how well this movie matches the search query.\n"
        f"\n"
        f'Query: "{query}"\n'
        f"Movie: {doc.get('title', '')} - {doc.get('document', '')}\n"
        f"\n"
        f"Consider:\n"
        f"- Direct relevance to query\n"
        f"- User intent (what they're looking for)\n"
        f"- Content appropriateness\n"
        f"\n"
        f"Rate 0-10 (10 = perfect match).\n"
        f"Output ONLY the number in your response, no other text or explanation.\n"
        f"\n"
        f"Score:"
    )
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content.strip()
    try:
        return float(text)
    except ValueError:
        return 0.0


def rerank_batch(query: str, docs: list[dict]) -> list[int]:
    client = _get_client()
    doc_list_str = "\n".join(
        f'{d["id"]}: {d["title"]} - {d["document"][:200]}'
        for d in docs
    )
    prompt = (
        f"Rank the movies listed below by relevance to the following search query.\n"
        f"\n"
        f'Query: "{query}"\n'
        f"\n"
        f"Movies:\n"
        f"{doc_list_str}\n"
        f"\n"
        f"Return the movie IDs in order of relevance, best match first.\n"
        f"Your response must be a raw JSON array of integers.\n"
        f"Do not wrap the JSON in Markdown. Do not use a ```json code block.\n"
        f"Do not include any explanatory text.\n"
        f"\n"
        f"For example:\n"
        f"[75, 12, 34, 2, 1]\n"
        f"\n"
        f"Ranking:"
    )
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return [d["id"] for d in docs]


def expand_query(query: str) -> str:
    client = _get_client()
    prompt = (
        f"Expand the user-provided movie search query below with related terms.\n"
        f"Add synonyms and related concepts that might appear in movie descriptions.\n"
        f"Keep expansions relevant and focused.\n"
        f"Output only the additional terms; they will be appended to the original query.\n"
        f"\n"
        f"Examples:\n"
        f'- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"\n'
        f'- "action movie with bear" -> "action thriller bear chase fight adventure"\n'
        f'- "comedy with bear" -> "comedy funny bear humor lighthearted"\n'
        f"\n"
        f'User query: "{query}"'
    )
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": prompt}],
    )
    expanded = response.choices[0].message.content.strip()
    return f"{query} {expanded}"
