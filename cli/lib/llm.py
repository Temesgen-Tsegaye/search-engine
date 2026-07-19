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
