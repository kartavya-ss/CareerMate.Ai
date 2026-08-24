"""
Tests for the skill-gap MCP server's pure logic (no API keys or network
access required — these test the matching/scoring functions directly).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from custom_skill_gap_mcp_server import (
    extract_required_skills,
    skill_gap_report,
    _normalize,
)


def test_normalize_collapses_whitespace_and_lowercases():
    assert _normalize("  Python   Developer  ") == "python developer"


def test_extract_required_skills_matches_known_role():
    skills = extract_required_skills(role="backend developer")
    assert "python" in skills
    assert "docker" in skills
    assert "sql" in skills


def test_extract_required_skills_includes_must_have_skills():
    skills = extract_required_skills(
        role="backend developer",
        must_have_skills=["Kubernetes"],
    )
    assert "kubernetes" in skills


def test_extract_required_skills_picks_up_keywords_from_listings():
    skills = extract_required_skills(
        role="unknown role with no taxonomy match",
        job_listings="We need someone strong in React and TypeScript.",
    )
    assert "react" in skills
    assert "typescript" in skills


def test_skill_gap_report_matched_and_missing():
    report = skill_gap_report(
        resume_profile="Experienced in python, sql and git.",
        required_skills=["python", "sql", "docker", "aws"],
    )
    assert set(report["matched_skills"]) == {"python", "sql"}
    assert set(report["missing_skills"]) == {"docker", "aws"}


def test_skill_gap_report_coverage_percent():
    report = skill_gap_report(
        resume_profile="python only",
        required_skills=["python", "sql", "docker", "aws"],
    )
    assert report["coverage_percent"] == 25.0


def test_skill_gap_report_handles_empty_required_skills():
    report = skill_gap_report(resume_profile="anything", required_skills=[])
    assert report["matched_skills"] == []
    assert report["missing_skills"] == []
    assert report["coverage_percent"] == 0.0