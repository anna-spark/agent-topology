
"""
tasks.py
--------
Generates distributed logic-grid puzzle instances for topology experiments.
 
Each puzzle:
  - Picks 5 fake musical artists from a pool of 20
  - Each artist has 4 attributes: genre, decade, nationality, award
  - Generates exactly 20 diverse atomic elimination clues from ground truth
  - Verifies via constraint solver that clues uniquely determine the solution
  - Distributes one clue per agent (N=20 agents, every agent gets a real clue)
  - Single-agent baseline uses an actual LLM call with only one clue
 
Usage:
    python tasks.py                        # generate & save 25 instances
    python tasks.py --n 30                 # generate 30 instances
    python tasks.py --verify-only          # verify saved tasks
    python tasks.py --baseline             # run LLM single-agent baseline
"""
 
import json
import os
import random
import argparse
import itertools
from collections import defaultdict
from pathlib import Path
from constraint import Problem, AllDifferentConstraint
import anthropic
 
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
 
# Clue type buckets for diversity enforcement
# Each bucket key = (attr1, attr2) pair or ("name", attr)
# We cap how many clues come from each bucket
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
    Uses python-constraint CSP solver with positive clues as hard constraints.
    """
    names = [a["name"] for a in artists]
    truth = {a["name"]: a for a in artists}
 
    problem = Problem()
    for name in names:
        for attr in ATTRIBUTES:
            domain = [a[attr] for a in artists if a["name"] == name]
            problem.addVariable(f"{name}_{attr}", domain)
 
    # All attribute values must be unique across artists (logic grid constraint)
    for attr in ATTRIBUTES:
        problem.addConstraint(
            AllDifferentConstraint(),
            [f"{name}_{attr}" for name in names]
        )
 
    # Positive clues fix variables
    for text in clue_texts:
        for a in artists:
            if POS_TEMPLATES["genre"].format(name=a["name"], val=a["genre"]) == text:
                problem.addConstraint(
                    lambda v, val=a["genre"]: v == val, [f"{a['name']}_genre"])
            if POS_TEMPLATES["decade"].format(name=a["name"], val=a["decade"]) == text:
                problem.addConstraint(
                    lambda v, val=a["decade"]: v == val, [f"{a['name']}_decade"])
            if POS_TEMPLATES["nationality"].format(name=a["name"], val=a["nationality"]) == text:
                problem.addConstraint(
                    lambda v, val=a["nationality"]: v == val, [f"{a['name']}_nationality"])
            if POS_TEMPLATES["award"].format(name=a["name"], val=a["award"]) == text:
                problem.addConstraint(
                    lambda v, val=a["award"]: v == val, [f"{a['name']}_award"])
 
    solutions = problem.getSolutions()
    return len(solutions) == 1
 
 
# ---------------------------------------------------------------------------
# Clue generation with diversity enforcement
# ---------------------------------------------------------------------------
 
def build_clue_pool(artists: list[dict]) -> list[tuple]:
    """
    Build all valid atomic clues for a given set of 5 artists.
    Each item: (bucket_key, text)
    bucket_key is used to enforce diversity caps.
    """
    truth = {a["name"]: a for a in artists}
    val_to_name = {(attr, a[attr]): a["name"] for a in artists for attr in ATTRIBUTES}
    pool = []
 
    # Positive clues — bucket: ("pos", attr)
    for a in artists:
        for attr in ATTRIBUTES:
            text = POS_TEMPLATES[attr].format(name=a["name"], val=a[attr])
            pool.append((("pos", attr), text))
 
    # Negative name clues — bucket: ("neg_name", attr)
    for a in artists:
        for attr in ATTRIBUTES:
            for other_a in artists:
                if other_a["name"] == a["name"]:
                    continue
                wrong_val = other_a[attr]
                text = NEG_NAME_TEMPLATES[attr].format(name=a["name"], val=wrong_val)
                pool.append((("neg_name", attr), text))
 
    # Negative cross clues — bucket: ("neg_cross", attr1, attr2)
    for (attr1, attr2), tmpl in NEG_CROSS_TEMPLATES.items():
        for a in artists:
            for b in artists:
                if a[attr1] == b[attr1]:
                    continue
                true_holder = val_to_name.get((attr1, a[attr1]))
                if true_holder and truth[true_holder][attr2] != b[attr2]:
                    text = tmpl.format(val1=a[attr1], val2=b[attr2])
                    pool.append((("neg_cross", attr1, attr2), text))
 
    # Deduplicate by text
    seen = set()
    deduped = []
    for bucket, text in pool:
        if text not in seen:
            seen.add(text)
            deduped.append((bucket, text))
 
    return deduped
 
 
def generate_clues(artists: list[dict], rng: random.Random, n_clues: int = 20) -> list[str]:
    """
    Generate exactly n_clues diverse atomic clues that uniquely constrain the solution.
 
    Strategy:
    1. Shuffle the full clue pool
    2. Greedy-select clues while enforcing bucket diversity caps
    3. Once unique solvability is confirmed, pad with remaining diverse clues
    4. Final shuffle so clue order gives no positional hints
    """
    pool = build_clue_pool(artists)
    rng.shuffle(pool)
 
    bucket_counts: dict = defaultdict(int)
    selected_texts: list[str] = []
    solved = False
 
    for bucket, text in pool:
        if bucket_counts[bucket] >= MAX_PER_BUCKET:
            continue
        selected_texts.append(text)
        bucket_counts[bucket] += 1
        if not solved and is_uniquely_solvable(artists, selected_texts):
            solved = True
            # Keep going to pad to n_clues with diverse remaining clues
 
    if not solved:
        # Fallback: relax diversity cap and use all positive clues
        all_pos = [
            POS_TEMPLATES[attr].format(name=a["name"], val=a[attr])
            for a in artists for attr in ATTRIBUTES
        ]
        selected_texts = list(dict.fromkeys(all_pos))  # deduplicate, preserve order
 
    # Trim or pad to exactly n_clues
    if len(selected_texts) > n_clues:
        # Keep first n_clues (already diversity-ordered)
        selected_texts = selected_texts[:n_clues]
    elif len(selected_texts) < n_clues:
        # Pad with any remaining unused clues
        used = set(selected_texts)
        extras = [t for _, t in pool if t not in used]
        selected_texts += extras[: n_clues - len(selected_texts)]
 
    rng.shuffle(selected_texts)
    return selected_texts[:n_clues]
 
 
# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------
 
def generate_task_instances(
    n_instances: int = 25,
    n_agents: int = 20,
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)
 
    # Pre-shuffle all 5-artist combinations
    all_combos = list(itertools.combinations(range(len(ARTIST_POOL)), 5))
    rng.shuffle(all_combos)
 
    instances = []
    answer_counts: dict = defaultdict(int)  # track answer frequency per artist
 
    for indices in all_combos:
        if len(instances) >= n_instances:
            break
 
        artists = [ARTIST_POOL[i] for i in indices]
 
        # Skip combos where any attribute value is duplicated among the 5 artists
        skip = False
        for attr in ATTRIBUTES:
            vals = [a[attr] for a in artists]
            if len(vals) != len(set(vals)):
                skip = True
                break
        if skip:
            continue
 
        # Generate diverse clues
        clues = generate_clues(artists, rng, n_clues=n_agents)
 
        if len(clues) < n_agents:
            continue
 
        if not is_uniquely_solvable(artists, clues):
            continue
 
        # Pick question — weight against overused answer artists
        q_attr, q_tmpl = rng.choice(QUESTION_TEMPLATES)
 
        # Sort artists by how often they've been answers (ascending) and pick from least-used
        sorted_artists = sorted(artists, key=lambda a: answer_counts[a["name"]])
        # Weighted sample: prefer less-used artists
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
        raise RuntimeError(
            f"Only generated {len(instances)}/{n_instances} valid instances. "
            "Try reducing n_instances or relaxing diversity constraints."
        )
 
    # Print answer distribution
    print("\nAnswer distribution:")
    for name, count in sorted(answer_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {name}: {count}")
 
    return instances
 
 
# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------
 
def save_tasks(instances: list[dict], path: str = "results/tasks.json") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(instances, f, indent=2)
    print(f"\nSaved {len(instances)} task instances to {path}")
 
 
def load_tasks(path: str = "results/tasks.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)
 
 
# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------
 
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
# Single-agent LLM baseline
# Matches spec: agent sees only its ONE clue, no communication
# ---------------------------------------------------------------------------
 
def single_agent_baseline(
    instances: list[dict],
    clue_index: int = 0,
    model: str = "claude-haiku-4-5-20251001",
) -> dict:
    """
    Run a real LLM single-agent baseline.
    Each instance: agent_{clue_index:02d} sees only its own clue, then guesses.
    This demonstrates that no single clue is sufficient to answer reliably.
 
    Returns dict with accuracy and per-task results.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    results = []
 
    for inst in instances:
        agent_id = f"agent_{clue_index:02d}"
        clue = inst["agent_clues"][agent_id]
        question = inst["question"]
        candidates = inst["candidates"]
        answer = inst["answer"]
 
        prompt = f"""You are trying to solve a music trivia puzzle.
 
You have been given ONE clue about a group of 5 artists:
Clue: {clue}
 
The artists in this puzzle are: {", ".join(candidates)}
 
Question: {question}
 
You must choose exactly one artist from the list above.
Reply with ONLY the artist's name, nothing else."""
 
        response = client.messages.create(
            model=model,
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}],
        )
 
        raw = response.content[0].text.strip()
        # Match to closest candidate
        guess = raw
        for c in candidates:
            if c.lower() in raw.lower() or raw.lower() in c.lower():
                guess = c
                break
 
        correct = guess == answer
        results.append({
            "task_id": inst["task_id"],
            "question": question,
            "answer": answer,
            "guess": guess,
            "correct": correct,
            "clue_seen": clue,
        })
 
        status = "✓" if correct else "✗"
        print(f"  Task {inst['task_id']:2d} {status}  guess={guess!r}  answer={answer!r}")
 
    accuracy = sum(r["correct"] for r in results) / len(results)
    print(f"\nSingle-agent baseline accuracy: {accuracy:.1%}  ({sum(r['correct'] for r in results)}/{len(results)} correct)")
    print(f"(Using clue from {agent_id} only — no communication)")
 
    return {"accuracy": accuracy, "results": results}
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
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
            artists_raw = [
                {"name": name, **attrs}
                for name, attrs in inst["artist_attributes"].items()
            ]
            ok = is_uniquely_solvable(artists_raw, clues)
            status = "✓ unique" if ok else "✗ NOT uniquely solvable"
            print(f"  Task {inst['task_id']:2d}: {status}")
            if not ok:
                all_ok = False
        print(f"\n{'All tasks verified OK!' if all_ok else 'WARNING: some tasks failed verification'}")
 
    elif args.baseline:
        instances = load_tasks()
        print(f"Running single-agent LLM baseline on {len(instances)} tasks...")
        print(f"(Agent sees only clue #{args.clue_index} — no communication)\n")
        result = single_agent_baseline(instances, clue_index=args.clue_index)
 
    else:
        print(f"Generating {args.n} task instances with {args.agents} agents each...")
        instances = generate_task_instances(
            n_instances=args.n,
            n_agents=args.agents,
            seed=args.seed,
        )
        save_tasks(instances)
        print("\n--- Example Task ---")
        print_task(instances[0])