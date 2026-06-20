"""
prompts.py
----------
All LLM prompt templates for the multi-agent topology experiment.

Centralised here so that prompts are held fixed across all topology
comparisons — a core experimental control requirement.
"""

# ---------------------------------------------------------------------------
# System prompt (shared across all agents and all topologies)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are one agent in a network of agents collaborating to solve a music trivia logic puzzle.

The puzzle involves 5 fictional music artists. Each artist has exactly one:
- Genre (e.g. Pop, Jazz, Rock)
- Debut decade (e.g. 1970s, 1990s, 2010s)
- Nationality (e.g. from Japan, from Brazil)
- Award won (e.g. Grammy, ARIA Award)

All attribute values are unique across the 5 artists — no two artists share a genre, decade, nationality, or award.

You have been given ONE private clue. Other agents have different clues. By sharing information, the group can solve the puzzle together. Be concise and precise."""


# ---------------------------------------------------------------------------
# Communication round prompt
# ---------------------------------------------------------------------------

MESSAGE_PROMPT = """Your private clue: {clue}

Messages you received from neighboring agents this round:
{received}

The 5 artists in this puzzle are: {candidates}
The question to answer: {question}

Your job this round is to share what you know. Write a SHORT message (2-4 sentences) that:
1. States your clue clearly
2. Summarizes any useful information from the messages you received
3. Notes any logical eliminations you can make by combining clues

Do NOT guess the final answer yet. Just share information."""


# ---------------------------------------------------------------------------
# Final vote prompt
# ---------------------------------------------------------------------------

VOTE_PROMPT = """Your private clue: {clue}

All messages you have received across all rounds:
{received}

The 5 artists in this puzzle are: {candidates}
The question to answer: {question}

Based on everything you know, which artist is the answer?

CRITICAL INSTRUCTIONS:
- Reply with ONLY the artist's name, exactly as it appears in the list above
- Do NOT explain your reasoning
- Do NOT write any sentences or paragraphs
- Do NOT write anything except the artist's name
- Your entire response must be exactly one name from the candidates list"""
