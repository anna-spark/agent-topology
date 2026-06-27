"""
tasks.py
--------
Generates distributed logic-grid puzzle instances for topology experiments.
"""
 
import json
import os
import random
import argparse
import itertools
from collections import defaultdict
from pathlib import Path
from constraint import Problem, AllDifferentConstraint
import pandas as pd

from prompts import SYSTEM_PROMPT
 
# ---------------------------------------------------------------------------
# Artist pool (20 fake artists, fixed ground-truth attributes)
# ---------------------------------------------------------------------------
 
ARTIST_POOL = [
    {"name": "Zara Voss",        "genre": "Pop",        "decade": "2010s", "nationality": "the UK",      "award": "Grammy"},
    {"name": "DJ Thunderclap",   "genre": "Electronic", "decade": "2000s", "nationality": "Germany",     "award": "MTV Award"},
    {"name": "Miles Corrigan",   "genre": "Jazz",       "decade": "1970s", "nationality": "the US",      "award": "Grammy"},
    {"name": "Seraphina Ó",      "genre": "Classical",  "decade": "1990s", "nationality": "Ireland",     "award": "Mercury Prize"},
    {"name": "Luka Dray",        "genre": "Hip-Hop",    "decade": "2010s", "nationality": "Canada",      "award": "Juno Award"},
    {"name": "The Riven",        "genre": "Rock",       "decade": "1980s", "nationality": "Australia",   "award": "ARIA Award"},
    {"name": "Celeste Nweke",    "genre": "R&B",        "decade": "2000s", "nationality": "Nigeria",     "award": "MOBO Award"},
    {"name": "Pedro Salvaje",    "genre": "Reggae",     "decade": "1990s", "nationality": "Brazil",      "award": "Latin Grammy"},
    {"name": "Yuki Tanase",      "genre": "Folk",       "decade": "1960s", "nationality": "Japan",       "award": "NHK Award"},
    {"name": "Colette Beaumont", "genre": "Pop",        "decade": "1980s", "nationality": "France",      "award": "Victoires"},
    {"name": "Orion Blaze",      "genre": "Rock",       "decade": "2010s", "nationality": "the US",      "award": "MTV Award"},
    {"name": "Sable Finch",      "genre": "Country",    "decade": "1990s", "nationality": "Canada",      "award": "Juno Award"},
    {"name": "Nneka Abara",      "genre": "Afrobeats",  "decade": "2010s", "nationality": "Nigeria",     "award": "MOBO Award"},
    {"name": "Frost Ellery",     "genre": "Electronic", "decade": "1980s", "nationality": "the UK",      "award": "Mercury Prize"},
    {"name": "Tomás Vidal",      "genre": "Classical",  "decade": "1970s", "nationality": "Brazil",      "award": "Latin Grammy"},
    {"name": "Remy Vail",        "genre": "Jazz",       "decade": "2000s", "nationality": "France",      "award": "Victoires"},
    {"name": "Kaia Storm",       "genre": "Hip-Hop",    "decade": "1990s", "nationality": "the US",      "award": "Grammy"},
    {"name": "The Drift",        "genre": "Folk",       "decade": "2000s", "nationality": "Ireland",     "award": "Mercury Prize"},
    {"name": "Senzo Malik",      "genre": "R&B",        "decade": "1980s", "nationality": "Australia",   "award": "ARIA Award"},
    {"name": "Hana Voss",        "genre": "Country",    "decade": "1970s", "nationality": "Germany",     "award": "NHK Award"},
]
 
ATTRIBUTES = ["genre", "decade", "nationality", "award"]
 
QUESTION_TEMPLATES = [
    ("award",       "Which artist won the {val}?"),
    ("genre",       "Which artist plays {val}?"),
    ("nationality", "Which artist is from {val}?"),
    ("decade",      "Which artist debuted in the {val}?"),
]
 
MAX_PER_BUCKET = 3
 
# ---------------------------------------------------------------------------
# Clue templates
# ---------------------------------------------------------------------------
 
POS_TEMPLATES = {
    "genre":       "{name} plays {val}.",
    "decade":      "{name} debuted in the {val}.",
    "nationality": "{name} is from {val}.",
    "award":       "{name} won the {val}.",
}
 
NEG_NAME_TEMPLATES = {
    "genre":       "{name} does not play {val}.",
    "decade":      "{name} did not debut in the {val}.",
    "nationality": "{name} is not from {val}.",
    "award":       "{name} did not win the {val}.",
}
 
NEG_CROSS_TEMPLATES = {
    ("genre",       "decade"):      "The {val1} artist did not debut in the {val2}.",
    ("genre",       "nationality"): "The {val1} artist is not from {val2}.",
    ("genre",       "award"):       "The {val1} artist did not win the {val2}.",
    ("decade",      "nationality"): "The artist who debuted in the {val1} is not from {val2}.",
    ("decade",      "award"):       "The artist who debuted in the {val1} did not win the {val2}.",
    ("nationality", "award"):       "The artist from {val1} did not win the {val2}.",
}
 
# ---------------------------------------------------------------------------
# Constraint solver
# ---------------------------------------------------------------------------
 
def is_uniquely_solvable(artists: list[dict], clue_texts: list[str]) -> bool:
    """
    Returns True iff clue_texts uniquely determine the correct assignment.
    """
    names = [a["name"] for a in artists]
    problem = Problem()
    
    for name in names:
        for attr in ATTRIBUTES:
            domain = [a[attr] for a in artists]
            problem.addVariable(f"{name}_{attr}", domain)
 
    for attr in ATTRIBUTES:
        problem.addConstraint(AllDifferentConstraint(), [f"{name}_{attr}" for name in names])
 
    # Evaluate all clue strings into constraints dynamically
    for text in clue_texts:
        for a in artists:
            name = a["name"]
            for attr in ATTRIBUTES:
                val = a[attr]
                
                # Positive Clues
                if POS_TEMPLATES[attr].format(name=name, val=val) == text:
                    problem.addConstraint(lambda v, target=val: v == target, [f"{name}_{attr}"])
                
                # Negative Name Clues
                if NEG_NAME_TEMPLATES[attr].format(name=name, val=val) == text:
                    problem.addConstraint(lambda v, target=val: v != target, [f"{name}_{attr}"])

        # Negative Cross Clues
        for (attr1, attr2), tmpl in NEG_CROSS_TEMPLATES.items():
            for a in artists:
                for b in artists:
                    if tmpl.format(val1=a[attr1], val2=b[attr2]) == text:
                        # If artist X has attr1 == val1, then artist X cannot have attr2 == val2
                        def cross_constraint(v1, v2, v1_target=a[attr1], v2_target=b[attr2]):
                            if v1 == v1_target:
                                return v2 != v2_target
                            return True
                        
                        for name in names:
                            problem.addConstraint(cross_constraint, [f"{name}_{attr1}", f"{name}_{attr2}"])
 
    solutions = problem.getSolutions()
    return len(solutions) == 1
 
# ---------------------------------------------------------------------------
# Clue generation with diversity enforcement
# ---------------------------------------------------------------------------
 
def build_clue_pool(artists: list[dict]) -> list[tuple]:
    truth = {a["name"]: a for a in artists}
    val_to_name = {(attr, a[attr]): a["name"] for a in artists for attr in ATTRIBUTES}
    pool = []
 
    # Positive clues
    for a in artists:
        for attr in ATTRIBUTES:
            text = POS_TEMPLATES[attr].format(name=a["name"], val=a[attr])
            pool.append((("pos", attr), text))
 
    # Negative name clues
    for a in artists:
        for attr in ATTRIBUTES:
            for other_a in artists:
                if other_a["name"] == a["name"]:
                    continue
                wrong_val = other_a[attr]
                text = NEG_NAME_TEMPLATES[attr].format(name=a["name"], val=wrong_val)
                pool.append((("neg_name", attr), text))
 
    # Negative cross clues
    for (attr1, attr2), tmpl in NEG_CROSS_TEMPLATES.items():
        for a in artists:
            for b in artists:
                if a[attr1] == b[attr1]:
                    continue
                true_holder = val_to_name.get((attr1, a[attr1]))
                if true_holder and truth[true_holder][attr2] != b[attr2]:
                    text = tmpl.format(val1=a[attr1], val2=b[attr2])
                    pool.append((("neg_cross", attr1, attr2), text))
 
    seen = set()
    deduped = []
    for bucket, text in pool:
        if text not in seen:
            seen.add(text)
            deduped.append((bucket, text))
 
    return deduped
 
def generate_clues(artists: list[dict], rng: random.Random, n_clues: int = 20) -> list[str]:
    pool = build_clue_pool(artists)
    rng.shuffle(pool)
 
    bucket_counts = defaultdict(int)
    selected_texts = []
    solved = False
 
    for bucket, text in pool:
        if bucket_counts[bucket] >= MAX_PER_BUCKET:
            continue
        selected_texts.append(text)
        bucket_counts[bucket] += 1
        if not solved and is_uniquely_solvable(artists, selected_texts):
            solved = True
 
    if not solved:
        all_pos = [
            POS_TEMPLATES[attr].format(name=a["name"], val=a[attr])
            for a in artists for attr in ATTRIBUTES
        ]
        selected_texts = list(dict.fromkeys(all_pos))
 
    if len(selected_texts) > n_clues:
        selected_texts = selected_texts[:n_clues]
    elif len(selected_texts) < n_clues:
        used = set(selected_texts)
        extras = [t for _, t in pool if t not in used]
        selected_texts += extras[: n_clues - len(selected_texts)]
 
    rng.shuffle(selected_texts)
    return selected_texts[:n_clues]
 
# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------
 
def generate_task_instances(n_instances: int = 25, n_agents: int = 20, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    all_combos = list(itertools.combinations(range(len(ARTIST_POOL)), 5))
    rng.shuffle(all_combos)
 
    instances = []
    answer_counts = defaultdict(int)
 
    for indices in all_combos:
        if len(instances) >= n_instances:
            break
 
        artists = [ARTIST_POOL[i] for i in indices]
        skip = False
        for attr in ATTRIBUTES:
            vals = [a[attr] for a in artists]
            if len(vals) != len(set(vals)):
                skip = True
                break
        if skip:
            continue
 
        clues = generate_clues(artists, rng, n_clues=n_agents)
        if len(clues) < n_agents or not is_uniquely_solvable(artists, clues):
            continue
 
        q_attr, q_tmpl = rng.choice(QUESTION_TEMPLATES)
        sorted_artists = sorted(artists, key=lambda a: answer_counts[a["name"]])
        weights = [1.0 / (1 + answer_counts[a["name"]]) for a in sorted_artists]
        total = sum(weights)
        weights = [w / total for w in weights]
        
        r = rng.random()
        cumulative = 0.0
        answer_artist = sorted_artists[-1]
        for a, w in zip(sorted_artists, weights):
            cumulative += w
            if r <= cumulative:
                answer_artist = a
                break
 
        answer_counts[answer_artist["name"]] += 1
        question = q_tmpl.format(val=answer_artist[q_attr])
        answer = answer_artist["name"]
        candidates = [a["name"] for a in artists]
        rng.shuffle(candidates)
 
        agent_clues = {f"agent_{i:02d}": clues[i] for i in range(n_agents)}
 
        instances.append({
            "task_id": len(instances),
            "artists": [a["name"] for a in artists],
            "artist_attributes": {
                a["name"]: {k: v for k, v in a.items() if k != "name"}
                for a in artists
            },
            "question": question,
            "answer": answer,
            "candidates": candidates,
            "agent_clues": agent_clues,
            "n_clues": len(clues),
        })
 
        print(f"  Generated task {len(instances)}/{n_instances}: '{question}' → {answer}")
 
    if len(instances) < n_instances:
        raise RuntimeError(f"Only generated {len(instances)}/{n_instances} valid instances.")
 
    print("\nAnswer distribution:")
    for name, count in sorted(answer_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {name}: {count}")
 
    return instances
 
def save_tasks(instances: list[dict], path: str = "results/tasks.json") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(instances, f, indent=2)
    print(f"\nSaved {len(instances)} task instances to {path}")
 
def load_tasks(path: str = "results/tasks.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)
 
def print_task(task: dict) -> None:
    print(f"\n{'='*60}")
    print(f"Task {task['task_id']}: {task['question']}")
    print(f"Answer: {task['answer']}")
    print(f"Candidates: {task['candidates']}")
    print(f"\nArtist attributes:")
    for name, attrs in task["artist_attributes"].items():
        print(f"  {name}: {attrs}")
    print(f"\nClues (one per agent):")
    for agent_id, clue in task["agent_clues"].items():
        print(f"  {agent_id}: {clue}")
 
# ---------------------------------------------------------------------------
# Single-agent LLM baseline (Strict matching implemented)
# ---------------------------------------------------------------------------
 
def single_agent_baseline(instances: list[dict], clue_index: int = 0,
                          model: str = "gemini/gemini-2.5-flash-lite") -> dict:
    """Controlled single-agent baseline.

    Uses the SAME model and the SAME prompts (SYSTEM_PROMPT + VOTE_PROMPT) as the
    multi-agent experiment, with no messages received — so the only difference vs a
    multi-agent run is the absence of communication. One agent holds one private clue,
    so it cannot reliably solve the puzzle (expected ≈ chance, 1/|candidates|). This is
    the reference the topology runs are compared against.
    """
    from agents import _make_client
    from prompts import VOTE_PROMPT

    client, backend = _make_client(model)
    api_model = model.replace("gemini/", "") if backend == "gemini" else model
    results = []

    for inst in instances:
        agent_id = f"agent_{clue_index:02d}"
        clue = inst["agent_clues"][agent_id]
        question = inst["question"]
        candidates = inst["candidates"]
        answer = inst["answer"]

        prompt = VOTE_PROMPT.format(
            clue=clue,
            received="(No messages received — you are solving alone.)",
            candidates=", ".join(candidates),
            question=question,
        )

        if backend == "gemini":
            from google.genai import types
            response = client.models.generate_content(
                model=api_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT, max_output_tokens=30, temperature=0.1,
                ),
            )
            raw = (response.text or "").strip()
        else:
            response = client.messages.create(
                model=api_model, max_tokens=30, system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()

        clean_raw = raw.strip().strip('"').strip("'").lower()
        guess = raw
        for c in candidates:
            if c.lower() == clean_raw:
                guess = c
                break

        correct = guess == answer
        results.append({
            "task_id": inst["task_id"],
            "question": question,
            "correct_answer": answer,
            "guess": guess,
            "correct": correct,
            "clue_seen": clue,
        })

        status = "✓" if correct else "✗"
        print(f"  Task {inst['task_id']:2d} {status}  guess={guess!r}  answer={answer!r}")

    accuracy = sum(r["correct"] for r in results) / len(results)
    print(f"\nSingle-agent baseline accuracy: {accuracy:.1%}  ({sum(r['correct'] for r in results)}/{len(results)} correct)")

    out_path = Path("results/baseline.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Saved baseline results to {out_path}")
    return {"accuracy": accuracy, "results": results}
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25, help="Number of task instances")
    parser.add_argument("--agents", type=int, default=20, help="Number of agents")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verify-only", action="store_true", help="Verify saved tasks")
    parser.add_argument("--baseline", action="store_true", help="Run LLM single-agent baseline")
    parser.add_argument("--clue-index", type=int, default=0, help="Which agent's clue to use for baseline")
    args = parser.parse_args()
 
    if args.verify_only:
        instances = load_tasks()
        print(f"Loaded {len(instances)} tasks, verifying...")
        all_ok = True
        for inst in instances:
            clues = list(inst["agent_clues"].values())
            artists_raw = [{"name": name, **attrs} for name, attrs in inst["artist_attributes"].items()]
            ok = is_uniquely_solvable(artists_raw, clues)
            status = "✓ unique" if ok else "✗ NOT uniquely solvable"
            print(f"  Task {inst['task_id']:2d}: {status}")
            if not ok:
                all_ok = False
        print(f"\n{'All tasks verified OK!' if all_ok else 'WARNING: some tasks failed verification'}")
    elif args.baseline:
        instances = load_tasks()
        result = single_agent_baseline(instances, clue_index=args.clue_index)
    else:
        instances = generate_task_instances(n_instances=args.n, n_agents=args.agents, seed=args.seed)
        save_tasks(instances)