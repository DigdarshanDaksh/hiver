"""One-command pipeline. Every stage is cached, so a re-run only does missing work.

  python src/run_all.py            # reproduce results from committed caches (~1 min, no API key needed)
  python src/run_all.py --fresh    # re-call the LLM for agent + judge (~10 min, needs OPENAI_API_KEY)
"""
import subprocess
import sys

from config import GOLDEN_CANDIDATES, GOLDEN_LABELS, PAIRS, ROOT

PY = sys.executable
SRC = ROOT / "src"


def step(*args):
    print(f"\n=== {' '.join(args)} ===", flush=True)
    subprocess.run([PY, str(SRC / args[0]), *args[1:]], check=True, cwd=SRC)


if __name__ == "__main__":
    fresh = ["--force"] if "--fresh" in sys.argv else []
    if not PAIRS.exists():
        step("prepare_data.py")              # needs data/raw/twcs.csv
    if not GOLDEN_CANDIDATES.exists():
        step("sample_golden.py")
    if not GOLDEN_LABELS.exists():
        sys.exit(f"Missing {GOLDEN_LABELS}. Label the golden set first: python src/label.py intents")
    step("run_systems.py", "trivial", "simple")
    step("run_systems.py", "agent_no_retrieval", "agent", *fresh)
    step("judge.py", *fresh)
    step("evaluate.py")
