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


# ===========================================================================
# Fragment-reconstruction task (distributed secret-code assembly)
# ---------------------------------------------------------------------------
# A pure information-flow task: the answer is the assembly of fragments held
# by different agents, with essentially no per-agent reasoning. Success depends
# only on whether all fragments propagate to a common agent within the fixed
# number of rounds — so collective performance is determined by the
# communication topology (reachability / path length), not by model capability.
# ===========================================================================

FRAGMENT_SYSTEM_PROMPT = """You are one agent in a network of agents collaborating to reconstruct a hidden secret code.

The secret code is an ordered sequence of uppercase letters — one letter at each position (position 1, position 2, and so on). You have been given ONE private fragment: the letter at a single position. Every other agent holds the letter at a different position, and no single agent knows the whole code. Only by relaying fragments across the network can the full code be assembled.

Be concise and exact. Never invent, alter, or drop a position you have learned — preserve every (position, letter) pair faithfully when you pass it on."""


FRAGMENT_MESSAGE_PROMPT = """Your private fragment: {clue}

Messages you received from neighboring agents this round:
{received}

The task: {question}

Write a SHORT message that relays code fragments. List EVERY (position, letter) pair you currently know — both your own fragment and any you have learned from messages you received. Write each pair as "pos N = X". Do not omit any pair you have seen. Do not invent letters for positions you have not learned."""


FRAGMENT_VOTE_PROMPT = """Your private fragment: {clue}

All messages you have received across all rounds:
{received}

The task: {question}

Report the secret code as a list of position=letter pairs — one pair for every position whose letter you know, from your own fragment or any message you received.

CRITICAL INSTRUCTIONS:
- Format: "position=letter", pairs separated by spaces. Example: 1=U 2=D 3=A
- Include EVERY position you know; OMIT any position you do not know (never guess a letter).
- Reply with ONLY the pairs — no words, no explanation, no extra punctuation."""


# ---------------------------------------------------------------------------
# Prompt registry — selected by task type so prompts stay fixed across all
# topology comparisons within a given task (a core experimental control).
# ---------------------------------------------------------------------------

PROMPTS = {
    "logic_grid": {"system": SYSTEM_PROMPT, "message": MESSAGE_PROMPT, "vote": VOTE_PROMPT},
    "fragment":   {"system": FRAGMENT_SYSTEM_PROMPT, "message": FRAGMENT_MESSAGE_PROMPT, "vote": FRAGMENT_VOTE_PROMPT},
}
