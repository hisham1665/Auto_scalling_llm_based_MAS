import pytest

from llm.response_parser import StructuredOutputError, parse_structured_response
from manager.agent_generator import AgentSpec


def test_parser_accepts_direct_json_and_fenced_json() -> None:
    direct = parse_structured_response('{"create_agent": false}', AgentSpec)
    fenced = parse_structured_response('Some text\n```json\n{"create_agent": false}\n```', AgentSpec)
    assert direct.create_agent is False
    assert fenced.create_agent is False


def test_parser_rejects_malformed_output() -> None:
    with pytest.raises(StructuredOutputError):
        parse_structured_response("not json at all", AgentSpec)
