# Limitations

- The configured cloud models are behaviorally different from the paper’s GPT-4o setup; exact numerical reproduction is not claimed.
- Cloud latency, context capacity, provider availability, quotas, and rate limits depend on the NVIDIA and Groq accounts and service state.
- Models may generate less reliable role decisions, malformed JSON, or repetitive contributions. Validation, correction retries, and guards degrade gracefully but cannot improve the model’s underlying judgment.
- Model output remains stochastic when temperature is nonzero. Random selection is reproducible for a fixed seed, while model behavior may still vary by provider/model version.
- Default repeated experiments use one run to remain laptop-friendly. Small sample sizes weaken statistical conclusions.
- Keyword coverage depends on the manually configured vocabulary and substring matching.
- The MTLD implementation is a lightweight implementation; neural BERTScore is optional and may require downloading a model. Lexical fallback is not BERTScore.
- The medical scenario is synthetic and educational. It is not a diagnostic system and does not provide medical advice.
- API and plots are intentionally secondary to the research engine.
