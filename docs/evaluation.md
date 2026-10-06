# Evaluation

The evaluator runs after generation, so metrics do not add inference calls to each turn.

- **Task-related coverage:** configured scenario terms are matched case-insensitively against agent responses. The score is unique found terms divided by unique vocabulary size, with per-category scores. This is a transparent engineering proxy and is not claimed to be identical to the paper’s annotation/evaluation process.
- **TF-IDF richness:** uses scikit-learn when installed; otherwise a documented term-frequency/inverse-document-frequency fallback is used.
- **MTLD:** a dependency-free forward/backward threshold implementation with threshold 0.72.
- **Pairwise topical consistency:** BERTScore between adjacent agent responses when neural BERTScore is enabled; otherwise labeled lexical token-F1.
- **Thematic relevance:** BERTScore/lexical fallback between the complete agent response and the task.

`evaluation/statistics.py` reports Mann-Whitney U with a normal approximation and tie correction, Pearson correlation, and Cliff’s Delta. Results include sample sizes, p-value, effect size, and a conservative interpretation. No statistical significance is asserted without a computed p-value below 0.05.
