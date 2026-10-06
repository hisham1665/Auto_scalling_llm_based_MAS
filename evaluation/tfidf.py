"""TF-IDF keyword richness with a scikit-learn path and small fallback."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", text.lower())


def tfidf_richness(documents: Sequence[str], vocabulary: Sequence[str] | None = None) -> float:
    if not documents:
        return 0.0
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vocabulary_mapping = ({term: index for index, term in enumerate(vocabulary)} if vocabulary else None)
        vectorizer = TfidfVectorizer(vocabulary=vocabulary_mapping)
        matrix = vectorizer.fit_transform(documents)
        return float(matrix.mean()) if matrix.shape[1] else 0.0
    except (ImportError, ValueError):
        token_docs = [_tokens(doc) for doc in documents]
        doc_freq: Counter[str] = Counter()
        for tokens in token_docs:
            doc_freq.update(set(tokens))
        scores: list[float] = []
        allowed = {term.lower() for term in vocabulary} if vocabulary else None
        for tokens in token_docs:
            counts = Counter(tokens)
            for term, count in counts.items():
                if allowed is not None and term not in allowed:
                    continue
                tf = count / max(1, len(tokens))
                idf = math.log((1 + len(token_docs)) / (1 + doc_freq[term])) + 1
                scores.append(tf * idf)
        return sum(scores) / len(scores) if scores else 0.0
