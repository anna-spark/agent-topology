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

Supported backends (set via model string prefix):
  - "gemini/..."   → Google Gemini (via google-genai)
  - anything else  → Anthropic Claude (via anthropic SDK)
"""

import os
import random
from collections import Counter
from typing import Optional

from prompts import SYSTEM_PROMPT, MESSAGE_PROMPT, VOTE_PROMPT


# ---------------------------------------------------------------------------
# Backend helpers — one client per simulator, not per call
# ---------------------------------------------------------------------------

def _make_client(model: str):
    """
    Return a (client, backend_tag) tuple based on the model string.
    The client is created once and reused across all LLM calls.
    """
    if model.startswith("gemini/"):
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "Set GEMINI_API_KEY (or GOOGLE_API_KEY) to use Gemini models."
            )
        client = genai.Client(api_key=api_key)
        return client, "gemini"
    else:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        return client, "anthropic"


def _call_backend(client, backend: str, model: str, system: str, user: str, max_tokens: int) -> str:
    """
    Unified LLM call — dispatches to Gemini or Anthropic based on backend tag.
    """
    if backend == "gemini":
        from google.genai import types
        # Strip the "gemini/" prefix to get the actual model name
        gemini_model = model[len("gemini/"):]
        response = client.models.generate_content(
            model=gemini_model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
                temperature=0.0,
            ),
        )
        return response.text.strip()
    else:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text.strip()


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
        model: str,
        client,
        backend: str,
    ):
        self.agent_id = agent_id
        self.clue = clue
        self.question = question
        self.candidates = candidates
        self.model = model
        self._client = client      # shared client — NOT created per call
        self._backend = backend
        self.received_messages: list[dict] = []   # {"round": int, "from": str, "text": str}
        self.sent_messages: list[dict] = []        # {"round": int, "text": str}
        self.vote: Optional[str] = None

    def _call_llm(self, prompt: str, max_tokens: int = 200) -> str:
        return _call_backend(
            self._client,
            self._backend,
            self.model,
            SYSTEM_PROMPT,
            prompt,
            max_tokens,
        )

    def _format_received(self, round_num: Optional[int] = None) -> str:
        """Format received messages, optionally filtering by a specific round."""
        filtered = self.received_messages
        if round_num is not None:
            filtered = [m for m in self.received_messages if m["round"] == round_num]

        if not filtered:
            return "(none yet)"

        lines = [
            f"[Round {m['round']} | From {m['from']}]: {m['text']}"
            for m in filtered
        ]
        return "\n".join(lines)

    def generate_message(self, round_num: int) -> str:
        """
        Generate an outgoing message.
        Round 1 → no prior messages yet.
        Round 2+ → include messages delivered in the previous round only,
                   so agents react to fresh information each round.
        """
        target_round = round_num - 1 if round_num > 1 else None

        prompt = MESSAGE_PROMPT.format(
            clue=self.clue,
            received=self._format_received(round_num=target_round),
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
        """After all rounds, vote using full message history."""
        prompt = VOTE_PROMPT.format(
            clue=self.clue,
            received=self._format_received(round_num=None),  # all rounds
            candidates=", ".join(self.candidates),
            question=self.question,
        )
        raw = self._call_llm(prompt, max_tokens=30)

        # Robust candidate matching (strip punctuation, case-insensitive)
        clean_raw = raw.strip().strip('"').strip("'").lower()
        vote = raw  # fallback if no match found

        if clean_raw:
            for c in self.candidates:
                if c.lower() == clean_raw:
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
        model: str = "gemini/gemini-2.0-flash-lite",
        seed: int = 0,
    ):
        self.task = task
        self.graph = graph
        self.topology_name = topology_name
        self.n_rounds = n_rounds
        self.model = model
        self.rng = random.Random(seed)

        # Create ONE client for all agents in this simulation
        client, backend = _make_client(model)

        # Initialize agents — pass the shared client in
        self.agents: dict[str, Agent] = {}
        for agent_id, clue in task["agent_clues"].items():
            self.agents[agent_id] = Agent(
                agent_id=agent_id,
                clue=clue,
                question=task["question"],
                candidates=task["candidates"],
                model=model,
                client=client,
                backend=backend,
            )

        self.log: list[dict] = []   # full communication trace

    def run(self) -> dict:
        """Run the full simulation: R communication rounds + final vote."""
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

        # Majority vote (with random tiebreak for reproducibility)
        vote_counts = Counter(votes.values())
        top_count = vote_counts.most_common(1)[0][1]
        tied = [ans for ans, cnt in vote_counts.items() if cnt == top_count]

        if len(tied) > 1:
            majority_answer = self.rng.choice(sorted(tied))  # sorted for reproducibility
        else:
            majority_answer = tied[0]

        correct = majority_answer == self.task["answer"]

        return {
            "task_id":        self.task["task_id"],
            "topology":       self.topology_name,
            "question":       self.task["question"],
            "correct_answer": self.task["answer"],
            "majority_answer":majority_answer,
            "correct":        correct,
            "votes":          votes,
            "vote_counts":    dict(vote_counts),
            "vote_agreement": top_count / len(agent_ids),
            "n_rounds":       self.n_rounds,
            "log":            self.log,
        }
