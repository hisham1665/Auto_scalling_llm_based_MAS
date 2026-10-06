"""Synthetic scenarios used for demonstrations and fair comparisons."""

from __future__ import annotations

from dataclasses import dataclass

from agents.agent import Agent
from llm.base_llm import BaseLLM


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    task: str
    seed_agent_specs: tuple[tuple[str, str, str], ...]
    static_agent_specs: tuple[tuple[str, str, str], ...]
    vocabulary: dict[str, tuple[str, ...]]
    disclaimer: str = ""

    def create_agents(self, llm: BaseLLM, *, static: bool = False) -> list[Agent]:
        specs = self.static_agent_specs if static else self.seed_agent_specs
        return [
            Agent(
                name=name,
                role=role,
                system_prompt=prompt,
                model=llm.model,
                _llm=llm,
            )
            for name, role, prompt in specs
        ]


MEDICAL_SCENARIO = Scenario(
    name="medical",
    task=(
        "A synthetic patient has persistent lower-right abdominal pain, fever, nausea, and reduced appetite. "
        "Develop a careful, educational discussion of possible causes, information to gather, diagnostic tests, "
        "urgent warning signs, and how different clinical perspectives would coordinate."
    ),
    seed_agent_specs=(
        (
            "Doctor",
            "primary care physician",
            "Assess the overall presentation, ask clarifying questions, and organize a cautious differential discussion.",
        ),
        (
            "Nurse",
            "clinical nursing specialist",
            "Focus on observations, triage, symptom progression, patient communication, and safety escalation.",
        ),
    ),
    static_agent_specs=(
        (
            "Doctor",
            "primary care physician",
            "Assess the overall presentation and organize a cautious differential discussion.",
        ),
        (
            "Nurse",
            "clinical nursing specialist",
            "Focus on triage, symptom progression, and safety escalation.",
        ),
        (
            "Radiologist",
            "radiology specialist",
            "Explain appropriate imaging questions, limitations, and how findings could inform discussion.",
        ),
        (
            "Surgeon",
            "surgical specialist",
            "Discuss when surgical consultation may be relevant and what information it requires.",
        ),
        (
            "Gastroenterologist",
            "gastrointestinal specialist",
            "Discuss gastrointestinal causes and non-surgical considerations with appropriate uncertainty.",
        ),
    ),
    vocabulary={
        "symptoms": ("abdominal pain", "lower-right", "fever", "nausea", "appetite", "tenderness"),
        "diagnoses": ("appendicitis", "differential", "infection", "inflammation", "gastrointestinal"),
        "tests": ("ultrasound", "ct", "blood test", "urinalysis", "imaging", "examination"),
        "treatment": ("treatment", "surgery", "antibiotics", "fluids", "monitoring", "consultation"),
        "prevention": ("warning signs", "follow-up", "safety", "urgent", "emergency", "uncertainty"),
    },
    disclaimer="This demonstration is for research/educational purposes only and does not provide medical advice.",
)


SOFTWARE_SCENARIO = Scenario(
    name="software_architecture",
    task=(
        "Review a proposed laptop-friendly event-driven software platform with a Python API, a relational database, "
        "background jobs, authentication, and a deployment plan. Identify architectural risks, security concerns, "
        "data design issues, performance assumptions, and practical validation steps."
    ),
    seed_agent_specs=(
        (
            "Lead Architect",
            "software architecture lead",
            "Decompose the system, surface tradeoffs, and keep the review coherent and actionable.",
        ),
        (
            "Product Analyst",
            "requirements and product analyst",
            "Check requirements, user impact, operational constraints, and whether proposed decisions serve the goal.",
        ),
    ),
    static_agent_specs=(
        (
            "Lead Architect",
            "software architecture lead",
            "Decompose the system and surface architecture tradeoffs.",
        ),
        (
            "Product Analyst",
            "requirements and product analyst",
            "Check requirements and user impact.",
        ),
        (
            "Security Specialist",
            "application security specialist",
            "Review authentication, authorization, secrets, threat models, and secure defaults.",
        ),
        (
            "Database Specialist",
            "database specialist",
            "Review schema, transactions, indexing, migrations, and data integrity.",
        ),
        (
            "Performance Engineer",
            "performance engineer",
            "Review bottlenecks, load assumptions, observability, and performance tests.",
        ),
    ),
    vocabulary={
        "architecture": ("api", "event-driven", "service", "deployment", "tradeoff", "architecture"),
        "security": ("authentication", "authorization", "threat", "secret", "encryption", "security"),
        "data": ("database", "schema", "transaction", "index", "migration", "integrity"),
        "performance": ("latency", "throughput", "load", "bottleneck", "cache", "performance"),
        "operations": ("monitoring", "logging", "backup", "recovery", "testing", "scalability"),
    },
)


SCENARIOS = {MEDICAL_SCENARIO.name: MEDICAL_SCENARIO, SOFTWARE_SCENARIO.name: SOFTWARE_SCENARIO}


def get_scenario(name: str) -> Scenario:
    key = name.strip().lower()
    aliases = {"software": "software_architecture", "architecture": "software_architecture"}
    key = aliases.get(key, key)
    try:
        return SCENARIOS[key]
    except KeyError as error:
        raise ValueError(f"Unknown scenario {name!r}; choose from {sorted(SCENARIOS)}") from error
