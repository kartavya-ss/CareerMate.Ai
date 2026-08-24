import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

mcp = FastMCP("Skill Gap MCP Server")

# A small starter taxonomy of common tech-role skills. This is intentionally
# simple and local (no external API key required) - it can be extended with
# more roles/skills over time, or later swapped for a real skills-database API.
SKILL_TAXONOMY: dict[str, list[str]] = {
    "backend developer": [
        "python", "java", "node.js", "sql", "rest api", "docker",
        "git", "system design", "aws", "postgresql",
    ],
    "frontend developer": [
        "javascript", "react", "html", "css", "typescript",
        "git", "responsive design", "testing",
    ],
    "data analyst": [
        "sql", "python", "excel", "power bi", "tableau",
        "statistics", "data visualization",
    ],
    "data scientist": [
        "python", "machine learning", "statistics", "sql",
        "pandas", "scikit-learn", "deep learning",
    ],
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


@mcp.tool()
def extract_required_skills(
    role: str,
    must_have_skills: list[str] | None = None,
    job_listings: str = "",
) -> list[str]:
    """
    Return the likely required skills for a target role, combining a known
    skill taxonomy, any explicitly requested must-have skills, and simple
    keyword matches found in real job listing text.
    """

    role_key = _normalize(role)
    base_skills = []

    for known_role, skills in SKILL_TAXONOMY.items():
        if known_role in role_key or role_key in known_role:
            base_skills = skills
            break

    required = set(base_skills)

    if must_have_skills:
        required.update(_normalize(s) for s in must_have_skills if s)

    listings_text = _normalize(job_listings)
    for skill in [s for skills in SKILL_TAXONOMY.values() for s in skills]:
        if skill in listings_text:
            required.add(skill)

    return sorted(required)


@mcp.tool()
def skill_gap_report(
    resume_profile: str,
    required_skills: list[str],
) -> dict[str, Any]:
    """
    Compare a resume profile's text against a list of required skills and
    return which are present and which appear to be missing.
    """

    resume_text = _normalize(resume_profile)

    matched = [skill for skill in required_skills if skill in resume_text]
    missing = [skill for skill in required_skills if skill not in resume_text]

    total = len(required_skills) or 1
    coverage_percent = round((len(matched) / total) * 100, 1)

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "coverage_percent": coverage_percent,
    }


if __name__ == "__main__":
    # mcp_client.py launches this as a stdio subprocess.
    mcp.run(transport="stdio")