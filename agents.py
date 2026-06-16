"""
agents.py
---------
Defines the Agent class and multi-agent communication simulator.

Each agent:
  - Has a unique ID and a private clue
  - Maintains a memory of received messages
  - Generates outgoing messages via LLM (summarizing clues + received info)
  - Votes on a final answer after R communication rounds

Communication runs in discrete rounds:
  Round 1: each agent sends a message based only on its own clue
  Round 2+: each agent sends a message synthesizing its clue + neighbor messages
  Final: each agent votes; majority vote determines collective answer
"""

import os
import json
import random
from collections import Counter
from typing import Optional
import anthropic

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are one agent in a network of agents collaborating to solve a music trivia logic puzzle.

The puzzle involves 5 fictional music artists. Each artist has exactly one:
- Genre (e.g. Pop, Jazz, Rock)
- Debut decade (e.g. 1970s, 1990s, 2010s)
- Nationality (e.g. from Japan, from Brazil)
- Award won (e.g. Grammy, ARIA Award)

All attribute values are unique across the 5 artists — no two artists share a genre, decade, nationality, or award.

You have been given ONE private clue. Other agents have different clues. By sharing information, the group can solve the puzzle together. Be concise and precise."""


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


VOTE_PROMPT = """Your private clue: {clue}

All messages you have received across all rounds:
{received}

The 5 artists in this puzzle are: {candidates}
The question to answer: {question}

Based on everything you know, which artist is the answer?
Reply with ONLY the artist's name from the list above. Nothing else."""


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class Agent:
    def __init__(
        self,
        agent_id: str,
        clue: str,
        question: str,
        candidates: list[str],
        model: str = "claude-haiku-4-5-20251001",
    ):
        self.agent_id = agent_id
        self.clue = clue
        self.question = question
        self.candidates = candidates
        self.model = model
        self.received_messages: list[dict] = []   # {"round": int, "from": str, "text": str}
        self.sent_messages: list[dict] = []        # {"round": int, "text": str}
        self.vote: Optional[str] = None

    def _call_llm(self, prompt: str, max_tokens: int = 200) -> str:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    def _format_received(self) -> str:
        if not self.received_messages:
            return "(none yet)"
        lines = []
        for msg in self.received_messages:
            lines.append(f"[Round {msg['round']} | From {msg['from']}]: {msg['text']}")
        return "\n".join(lines)

    def generate_message(self, round_num: int) -> str:
        """Generate an outgoing message for this round."""
        prompt = MESSAGE_PROMPT.format(
            clue=self.clue,
            received=self._format_received(),
            candidates=", ".join(self.candidates),
            question=self.question,
        )
        text = self._call_llm(prompt, max_tokens=150)
        self.sent_messages.append({"round": round_num, "text": text})
        return text

    def receive_message(self, from_agent: str, text: str, round_num: int) -> None:
        """Store an incoming message from a neighbor."""
        self.received_messages.append({
            "round": round_num,
            "from": from_agent,
            "text": text,
        })

    def cast_vote(self) -> str:
        """After all rounds, vote for the final answer."""
        prompt = VOTE_PROMPT.format(
            clue=self.clue,
            received=self._format_received(),
            candidates=", ".join(self.candidates),
            question=self.question,
        )
        raw = self._call_llm(prompt, max_tokens=30)

        # Match raw output to closest candidate
        vote = raw
        for c in self.candidates:
            if c.lower() in raw.lower() or raw.lower() in c.lower():
                vote = c
                break

        self.vote = vote
        return vote


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class MultiAgentSimulator:
    def __init__(
        self,
        task: dict,
        graph,                        # networkx Graph
        topology_name: str,
        n_rounds: int = 3,
        model: str = "claude-haiku-4-5-20251001",
        seed: int = 0,
    ):
        self.task = task
        self.graph = graph
        self.topology_name = topology_name
        self.n_rounds = n_rounds
        self.model = model
        self.rng = random.Random(seed)

        # Initialise agents
        self.agents: dict[str, Agent] = {}
        for agent_id, clue in task["agent_clues"].items():
            self.agents[agent_id] = Agent(
                agent_id=agent_id,
                clue=clue,
                question=task["question"],
                candidates=task["candidates"],
                model=model,
            )

        self.log: list[dict] = []   # full communication trace

    def run(self) -> dict:
        """
        Run the full simulation: R communication rounds + final vote.
        Returns a result dict with answer, correctness, vote breakdown, and log.
        """
        agent_ids = list(self.agents.keys())

        # Communication rounds
        for round_num in range(1, self.n_rounds + 1):
            round_messages: dict[str, str] = {}

            # Each agent generates its outgoing message
            for aid in agent_ids:
                msg = self.agents[aid].generate_message(round_num)
                round_messages[aid] = msg
                self.log.append({
                    "round": round_num,
                    "event": "send",
                    "agent": aid,
                    "text": msg,
                })

            # Deliver messages along graph edges
            for aid in agent_ids:
                neighbors = list(self.graph.neighbors(aid))
                for neighbor_id in neighbors:
                    if neighbor_id in round_messages:
                        self.agents[aid].receive_message(
                            from_agent=neighbor_id,
                            text=round_messages[neighbor_id],
                            round_num=round_num,
                        )
                        self.log.append({
                            "round": round_num,
                            "event": "receive",
                            "to": aid,
                            "from": neighbor_id,
                        })

        # Final vote
        votes = {}
        for aid in agent_ids:
            vote = self.agents[aid].cast_vote()
            votes[aid] = vote
            self.log.append({
                "round": "final",
                "event": "vote",
                "agent": aid,
                "vote": vote,
            })

        # Majority vote
        vote_counts = Counter(votes.values())
        majority_answer = vote_counts.most_common(1)[0][0]
        correct = majority_answer == self.task["answer"]

        # Handle tie (fall back to random among tied)
        top_count = vote_counts.most_common(1)[0][1]
        tied = [ans for ans, cnt in vote_counts.items() if cnt == top_count]
        if len(tied) > 1:
            majority_answer = self.rng.choice(tied)
            correct = majority_answer == self.task["answer"]

        return {
            "task_id": self.task["task_id"],
            "topology": self.topology_name,
            "question": self.task["question"],
            "correct_answer": self.task["answer"],
            "majority_answer": majority_answer,
            "correct": correct,
            "votes": votes,
            "vote_counts": dict(vote_counts),
            "vote_agreement": top_count / len(agent_ids),   # fraction who voted for winner
            "n_rounds": self.n_rounds,
            "log": self.log,
        }


# ---------------------------------------------------------------------------
# Quick sanity check (one task, chain topology, 2 rounds)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import networkx as nx
    import sys
    sys.path.insert(0, ".")

    # Load tasks
    with open("results/tasks.json") as f:
        tasks = json.load(f)

    task = tasks[0]
    print(f"Task: {task['question']}")
    print(f"Correct answer: {task['answer']}")
    print(f"Candidates: {task['candidates']}\n")

    # Build a tiny chain graph for the smoke test
    agent_ids = list(task["agent_clues"].keys())
    G = nx.path_graph(agent_ids)   # chain: agent_00 - agent_01 - ... - agent_19

    sim = MultiAgentSimulator(
        task=task,
        graph=G,
        topology_name="chain",
        n_rounds=2,           # keep short for the smoke test
    )

    print("Running simulation (chain, 2 rounds)...")
    result = sim.run()

    print(f"\nMajority answer: {result['majority_answer']}")
    print(f"Correct: {result['correct']}")
    print(f"Vote counts: {result['vote_counts']}")
    print(f"Vote agreement: {result['vote_agreement']:.0%}")
    print(f"\nFirst 5 log entries:")
    for entry in result["log"][:5]:
        print(f"  {entry}")