from typing import List

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

_TOKENIZER = None
_MODEL = None
DEFAULT_TOP_K = 5


def _get_model_and_tokenizer():
    global _TOKENIZER, _MODEL
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained("google/flan-t5-base")
    if _MODEL is None:
        _MODEL = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return _MODEL, _TOKENIZER


def generate_answer(question: str, retrieved_chunks: List[str], k: int = DEFAULT_TOP_K) -> str:
    """
    Generate an answer from retrieved chunks using a local open-source model.
    """
    k = max(1, k)
    context = "\n\n".join(retrieved_chunks[:k])

    prompt = f"""
Answer the question based only on the context below.
Write a concise 3-5 sentence explanation with key concepts/mechanisms.
Do not answer in a single short phrase.

Question:
{question}

Context:
{context}

If the answer is not in the context, say that clearly.
""".strip()

    model, tokenizer = _get_model_and_tokenizer()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            min_new_tokens=40,
            max_new_tokens=180,
            do_sample=False,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
