"""
agents.py
---------
Defines the Agent class and multi-agent communication simulator.
"""

import os
import re
import time
import random
from collections import Counter
from typing import Optional

from prompts import SYSTEM_PROMPT, MESSAGE_PROMPT, VOTE_PROMPT, PROMPTS


def normalize_code(raw: str, code_len: int) -> str:
    """Parse an agent's free-form code guess into a fixed-length string over [A-Z?].

    Agents are asked to reply with the bare code (one char per position, '?' for
    unknown). We defensively handle stray formatting: if the reply looks like the
    structured "pos N = X" relay form instead of a bare code, we rebuild the code
    from those pairs; otherwise we keep the A-Z/? characters in order. The result is
    always exactly `code_len` chars, padded with '?'."""
    raw = (raw or "").upper()

    # Structured fallback: "POS 3 = Q", "3=Q", "POSITION 12: A", etc.
    pairs = re.findall(r'(?:POS(?:ITION)?\s*)?(\d{1,3})\s*[=:]\s*([A-Z?])', raw)
    if pairs:
        slots = ['?'] * code_len
        for pos_str, ch in pairs:
            p = int(pos_str) - 1
            if 0 <= p < code_len:
                slots[p] = ch
        return ''.join(slots)

    # Bare-code form: keep only letters and '?' in order.
    chars = re.findall(r'[A-Z?]', raw)
    code = ''.join(chars)[:code_len]
    return code.ljust(code_len, '?')

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
    def __init__(self, agent_id: int, clue: str, candidates: list[str], question: str, client, backend: str, model: str,
                 task_type: str = "logic_grid"):
        self.agent_id = agent_id
        self.clue = clue
        self.candidates = candidates
        self.question = question
        self.client = client
        self.backend = backend
        self.model = model.replace("gemini/", "") if backend == "gemini" else model
        self.task_type = task_type
        self.prompts = PROMPTS[task_type]
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
        max_retries = 8
        base_delay = 5.0
        
        for attempt in range(max_retries):
            try:
                if self.backend == "gemini":
                    from google.genai import types
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=self.prompts["system"],
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
                        system=self.prompts["system"],
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
                # Empty/None responses are transient (truncation, momentary backend hiccup
                # under load), so retry them with backoff rather than aborting the whole run.
                is_retryable = ("429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "503" in error_str or "UNAVAILABLE" in error_str
                                or "Empty" in error_str)
                if is_retryable:
                    if attempt == max_retries - 1:
                        raise e
                    delay = min(base_delay * (2 ** attempt), 60.0) + random.uniform(1.0, 3.0)
                    print(f"\n[API Exception Overridden] Agent {self.agent_id} hit traffic limit. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                else:
                    raise e

    def generate_message(self, round_idx: int) -> str:
        received_summary = "\n".join(self.history) if self.history else "(No messages received yet)"
        prompt = self.prompts["message"].format(clue=self.clue, received=received_summary, candidates=", ".join(self.candidates), question=self.question)
        return self._call_llm(prompt, max_tokens=250)

    def cast_vote(self) -> str:
        received_summary = "\n".join(self.history) if self.history else "(No messages received)"
        prompt = self.prompts["vote"].format(clue=self.clue, received=received_summary, candidates=", ".join(self.candidates), question=self.question)
        # Fragment reconstruction emits one position=letter pair per known slot, so it
        # needs room for ~code_len pairs; the logic-grid vote is a single name, so a
        # tight cap keeps it from rambling.
        max_tokens = 200 if self.task_type == "fragment" else 30
        return self._call_llm(prompt, max_tokens=max_tokens)

class MultiAgentSimulator:
    def __init__(self, task: dict, adj_matrix: list[list[int]], model: str, n_rounds: int,
                 topology_name: str = "unknown", seed: int = 42, edge_drop_rate: float = 0.0,
                 final_vote_only: bool = False, task_type: str = "logic_grid"):
        self.task = task
        self.adj_matrix = adj_matrix
        self.n_rounds = n_rounds
        self.model = model
        self.topology_name = topology_name
        self.seed = seed
        self.edge_drop_rate = edge_drop_rate
        self.final_vote_only = final_vote_only
        self.task_type = task_type
        self.rng = random.Random(seed)
        self.log = []
        client, backend = _make_client(model)

        self.agents: dict[int, Agent] = {}
        for idx in range(len(task["agent_clues"])):
            clue_key = f"agent_{idx:02d}"
            self.agents[idx] = Agent(agent_id=idx, clue=task["agent_clues"][clue_key], candidates=task["candidates"], question=task["question"], client=client, backend=backend, model=model, task_type=task_type)

    def _tally(self, votes: dict) -> tuple[str, dict, float]:
        """Majority answer (seeded tie-break), vote counts, and agreement fraction."""
        vote_counts = Counter(votes.values())
        top_count = vote_counts.most_common(1)[0][1]
        tied = [ans for ans, cnt in vote_counts.items() if cnt == top_count]
        majority_answer = self.rng.choice(sorted(tied)) if len(tied) > 1 else tied[0]
        return majority_answer, dict(vote_counts), top_count / len(votes)

    def _recovery(self, votes: dict) -> tuple[Optional[float], Optional[float]]:
        """Fragment task only: continuous score = fraction of code positions an agent
        recovered correctly. Returns (mean across agents, best single agent), which read
        as the spec's "designated aggregator" final-answer procedure — mean = expected
        recovery for a randomly chosen aggregator, best = best-case aggregator. Returns
        (None, None) for non-fragment tasks."""
        if self.task_type != "fragment":
            return None, None
        fracs = [self._code_recovery(v) for v in votes.values()]
        return sum(fracs) / len(fracs), max(fracs)

    def _code_recovery(self, code: str) -> float:
        """Fraction of positions in `code` that match the true answer."""
        answer = self.task["answer"]
        n = len(answer)
        return sum(1 for i in range(n) if i < len(code) and code[i] == answer[i]) / n

    def _position_majority(self, votes: dict) -> tuple[str, float]:
        """Fragment final-answer procedure: per-position majority vote across all agents.

        For each slot, the most common character among agents' guesses is chosen
        (a real letter is preferred over '?' on a tie, since '?' is an abstention).
        This pools partial knowledge correctly and — because per-agent transcription
        errors are independent — cancels clerical noise, so a fully-reachable network
        recovers ~all positions even when individual agents are sloppy. Returns the
        aggregated code and the mean per-position agreement (fraction of agents backing
        the winning character, averaged over positions)."""
        answer = self.task["answer"]
        n = len(answer)
        codes = list(votes.values())
        out, agreements = [], []
        for i in range(n):
            col = Counter(c[i] for c in codes if i < len(c))
            if not col:
                out.append('?'); agreements.append(0.0); continue
            top_count = col.most_common(1)[0][1]
            tied = sorted(ch for ch, cnt in col.items() if cnt == top_count)
            non_q = [ch for ch in tied if ch != '?']
            if len(tied) > 1:
                pick = self.rng.choice(non_q) if non_q else '?'
            else:
                pick = tied[0]
            out.append(pick)
            agreements.append(top_count / len(codes))
        return ''.join(out), sum(agreements) / n

    def run(self) -> dict:
        agent_ids = sorted(list(self.agents.keys()))
        num_agents = len(agent_ids)

        round_results = []        # accuracy/agreement after each voting round
        votes = {}                # most recent round's votes (final = last round)
        majority_answer = None
        vote_counts = {}
        vote_agreement = 0.0
        collective_recovery = mean_recovery = best_recovery = None  # fragment task only

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
                    raw_vote = self.agents[aid].cast_vote()
                    # Fragment task: parse the free-form reply into a fixed-length code
                    # so the tally and recovery scoring compare like with like.
                    vote = normalize_code(raw_vote, len(self.task["answer"])) if self.task_type == "fragment" else raw_vote
                    votes[aid] = vote
                    vote_event = {"round": r, "event": "vote", "agent": aid, "vote": vote}
                    if self.task_type == "fragment":
                        vote_event["raw_vote"] = raw_vote
                    self.log.append(vote_event)

                if self.task_type == "fragment":
                    # Per-position majority is the natural aggregator for reconstruction.
                    majority_answer, vote_agreement = self._position_majority(votes)
                    vote_counts = dict(Counter(votes.values()))
                    mean_recovery, best_recovery = self._recovery(votes)
                    collective_recovery = self._code_recovery(majority_answer)
                else:
                    majority_answer, vote_counts, vote_agreement = self._tally(votes)
                    mean_recovery = best_recovery = collective_recovery = None

                round_results.append({
                    "round": r,
                    "majority": majority_answer,
                    "correct": majority_answer == self.task["answer"],
                    "agreement": vote_agreement,
                    "collective_recovery": collective_recovery,
                    "mean_recovery": mean_recovery,
                    "best_recovery": best_recovery,
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
            "collective_recovery": collective_recovery,
            "mean_recovery": mean_recovery, "best_recovery": best_recovery,
            "round_results": round_results,
            "n_rounds": self.n_rounds, "model": self.model, "seed": self.seed, "edge_drop_rate": self.edge_drop_rate,
            "total_input_tokens": total_input, "total_output_tokens": total_output,
            "total_tokens": total_input + total_output, "n_messages": n_messages, "n_llm_calls": n_llm_calls,
        }