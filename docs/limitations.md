# Limitations

- Spark-X2.5-4B is smaller and behaviorally different from the paper’s GPT-4o setup; exact numerical reproduction is not claimed.
- Local inference latency, context capacity, quantization, and thermal limits depend on the laptop and Ollama configuration.
- Smaller models may generate less reliable role decisions, malformed JSON, or repetitive contributions. Validation, correction retries, and guards degrade gracefully but cannot improve the model’s underlying judgment.
- Model output remains stochastic when temperature is nonzero. Random selection is reproducible for a fixed seed, while model behavior may still vary by Ollama/model version.
- Default repeated experiments use one run to remain laptop-friendly. Small sample sizes weaken statistical conclusions.
- Keyword coverage depends on the manually configured vocabulary and substring matching.
- The MTLD implementation is a lightweight implementation; neural BERTScore is optional and may require downloading a model. Lexical fallback is not BERTScore.
- The medical scenario is synthetic and educational. It is not a diagnostic system and does not provide medical advice.
- API and plots are intentionally secondary to the research engine.
