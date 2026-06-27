"""
agents.py
---------
Defines the Agent class and multi-agent communication simulator.
"""

import os
import time
import random
from collections import Counter
from typing import Optional

from prompts import SYSTEM_PROMPT, MESSAGE_PROMPT, VOTE_PROMPT

def _make_client(model: str):
    if model.startswith("gemini/"):
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("Gemini model requested but neither GEMINI_API_KEY nor GOOGLE_API_KEY set.")
        return genai.Client(api_key=api_key), "gemini"
    else:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("Claude model requested but ANTHROPIC_API_KEY is not set.")
        return anthropic.Anthropic(api_key=api_key), "anthropic"

class Agent:
    def __init__(self, agent_id: int, clue: str, candidates: list[str], question: str, client, backend: str, model: str):
        self.agent_id = agent_id
        self.clue = clue
        self.candidates = candidates
        self.question = question
        self.client = client
        self.backend = backend
        self.model = model.replace("gemini/", "") if backend == "gemini" else model
        self.history: list[str] = []
        # Budget tracking (Step 3 deliverable + evidence the budget is held fixed)
        self.input_tokens = 0
        self.output_tokens = 0
        self.n_calls = 0

    def receive_message(self, round_idx: int, sender_id: int, msg: str) -> None:
        self.history.append(f"[Round {round_idx} from Agent {sender_id}]: {msg}")

    def _record_usage(self, usage, in_attr: str, out_attr: str) -> None:
        """Accumulate token usage from one API response (counts the call regardless)."""
        self.n_calls += 1
        if usage is None:
            return
        self.input_tokens += getattr(usage, in_attr, None) or 0
        self.output_tokens += getattr(usage, out_attr, None) or 0

    def _call_llm(self, prompt: str, max_tokens: int = 200) -> str:
        # CLEAN RESET: No staggers, no forced delays inside the agent
        max_retries = 5
        base_delay = 5.0
        
        for attempt in range(max_retries):
            try:
                if self.backend == "gemini":
                    from google.genai import types
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            max_output_tokens=max_tokens,
                            temperature=0.1,
                        ),
                    )
                    if response and response.text:
                        self._record_usage(
                            getattr(response, "usage_metadata", None),
                            in_attr="prompt_token_count",
                            out_attr="candidates_token_count",
                        )
                        return response.text.strip()
                    else:
                        raise ValueError("Empty or None response text returned from Gemini API.")
                else:
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        system=SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    if response and response.content:
                        self._record_usage(
                            getattr(response, "usage", None),
                            in_attr="input_tokens",
                            out_attr="output_tokens",
                        )
                        return response.content[0].text.strip()
                    raise ValueError("Empty response from Anthropic API.")
                    
            except Exception as e:
                error_str = str(e)
                is_retryable = ("429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "503" in error_str or "UNAVAILABLE" in error_str)
                if is_retryable:
                    if attempt == max_retries - 1:
                        raise e
                    delay = (base_delay * (2 ** attempt)) + random.uniform(1.0, 3.0)
                    print(f"\n[API Exception Overridden] Agent {self.agent_id} hit traffic limit. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                else:
                    raise e

    def generate_message(self, round_idx: int) -> str:
        received_summary = "\n".join(self.history) if self.history else "(No messages received yet)"
        prompt = MESSAGE_PROMPT.format(clue=self.clue, received=received_summary, candidates=", ".join(self.candidates), question=self.question)
        return self._call_llm(prompt, max_tokens=250)

    def cast_vote(self) -> str:
        received_summary = "\n".join(self.history) if self.history else "(No messages received)"
        prompt = VOTE_PROMPT.format(clue=self.clue, received=received_summary, candidates=", ".join(self.candidates), question=self.question)
        return self._call_llm(prompt, max_tokens=30)

class MultiAgentSimulator:
    def __init__(self, task: dict, adj_matrix: list[list[int]], model: str, n_rounds: int,
                 topology_name: str = "unknown", seed: int = 42, edge_drop_rate: float = 0.0,
                 final_vote_only: bool = False):
        self.task = task
        self.adj_matrix = adj_matrix
        self.n_rounds = n_rounds
        self.model = model
        self.topology_name = topology_name
        self.seed = seed
        self.edge_drop_rate = edge_drop_rate
        self.final_vote_only = final_vote_only
        self.rng = random.Random(seed)
        self.log = []
        client, backend = _make_client(model)
        
        self.agents: dict[int, Agent] = {}
        for idx in range(len(task["agent_clues"])):
            clue_key = f"agent_{idx:02d}"
            self.agents[idx] = Agent(agent_id=idx, clue=task["agent_clues"][clue_key], candidates=task["candidates"], question=task["question"], client=client, backend=backend, model=model)

    def _tally(self, votes: dict) -> tuple[str, dict, float]:
        """Majority answer (seeded tie-break), vote counts, and agreement fraction."""
        vote_counts = Counter(votes.values())
        top_count = vote_counts.most_common(1)[0][1]
        tied = [ans for ans, cnt in vote_counts.items() if cnt == top_count]
        majority_answer = self.rng.choice(sorted(tied)) if len(tied) > 1 else tied[0]
        return majority_answer, dict(vote_counts), top_count / len(votes)

    def run(self) -> dict:
        agent_ids = sorted(list(self.agents.keys()))
        num_agents = len(agent_ids)

        round_results = []        # accuracy/agreement after each voting round
        votes = {}                # most recent round's votes (final = last round)
        majority_answer = None
        vote_counts = {}
        vote_agreement = 0.0

        for r in range(1, self.n_rounds + 1):
            round_messages = {}
            for aid in agent_ids:
                msg = self.agents[aid].generate_message(round_idx=r)
                round_messages[aid] = msg
                self.log.append({"round": r, "event": "send", "from": aid, "message": msg})

            for aid in agent_ids:
                for neighbor_id in range(num_agents):
                    if self.adj_matrix[aid][neighbor_id] == 1:
                        self.agents[aid].receive_message(round_idx=r, sender_id=neighbor_id, msg=round_messages[neighbor_id])
                        self.log.append({"round": r, "event": "receive", "to": aid, "from": neighbor_id})

            # Vote after every round (information-propagation curve), or only on the last
            # round when --final-vote-only is set.
            if (not self.final_vote_only) or (r == self.n_rounds):
                votes = {}
                for aid in agent_ids:
                    vote = self.agents[aid].cast_vote()
                    votes[aid] = vote
                    self.log.append({"round": r, "event": "vote", "agent": aid, "vote": vote})

                majority_answer, vote_counts, vote_agreement = self._tally(votes)
                round_results.append({
                    "round": r,
                    "majority": majority_answer,
                    "correct": majority_answer == self.task["answer"],
                    "agreement": vote_agreement,
                })

        # Aggregate per-run budget across all agents (Step 3 deliverable).
        total_input = sum(a.input_tokens for a in self.agents.values())
        total_output = sum(a.output_tokens for a in self.agents.values())
        n_llm_calls = sum(a.n_calls for a in self.agents.values())
        n_messages = sum(1 for e in self.log if e.get("event") == "send")

        # Note: `log` is intentionally NOT included here to keep metrics.csv lean.
        # The runner persists self.log separately to results/raw_logs/.
        return {
            "task_id": self.task["task_id"], "topology": self.topology_name, "question": self.task["question"],
            "correct_answer": self.task["answer"], "majority_answer": majority_answer, "correct": majority_answer == self.task["answer"],
            "votes": votes, "vote_counts": vote_counts, "vote_agreement": vote_agreement,
            "round_results": round_results,
            "n_rounds": self.n_rounds, "model": self.model, "seed": self.seed, "edge_drop_rate": self.edge_drop_rate,
            "total_input_tokens": total_input, "total_output_tokens": total_output,
            "total_tokens": total_input + total_output, "n_messages": n_messages, "n_llm_calls": n_llm_calls,
        }