"""
Wisdom corpus ingestion script

Usage:
    python -m data.ingest             # fetch and index remote sources (default)
    python -m data.ingest --local     # ingest local sources only
    python -m data.ingest --remote    # ingest remote sources only
    python -m data.ingest --local --remote  # ingest both local and remote sources
    python -m data.ingest --force     # re-index even if already populated

This writes to data/chroma/ which is .gitignore'd.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.embedder import get_embedder
from agent.tools.wisdom_retriever import WisdomRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

CHROMA_DIR = Path(__file__).parent / "chroma"


def main():
    parser = argparse.ArgumentParser(description="Ingest wisdom corpus into ChromaDB")
    parser.add_argument("--force", action="store_true", help="Re-index even if already populated")
    parser.add_argument("--local", action="store_true", help="Ingest local sources")
    parser.add_argument("--remote", action="store_true", help="Ingest remote sources")
    args = parser.parse_args()

    CHROMA_DIR.mkdir(exist_ok=True)

    print("Loading embedding model...")
    embedder = get_embedder()

    # Instantiate the retriever
    retriever = WisdomRetriever(persist_dir=CHROMA_DIR, embedder=embedder)

    # Determine which ingestion modes to execute
    modes_to_run = []
    if args.local:
        modes_to_run.append("local")

    # Default to remote if --remote is explicitly called OR if no flags are passed
    if args.remote or (not args.local and not args.remote):
        modes_to_run.append("remote")

    # Process the chosen ingestion modes sequentially
    for mode in modes_to_run:
        print(f"\n--- Processing {mode.capitalize()} Source Ingestion ---")

        if retriever.is_populated() and not args.force:
            print(f"ChromaDB collection already contains data ({retriever._collection.count()} chunks). Use --force to re-index.")
            continue

        print(f"Indexing {mode} wisdom corpus")
        retriever.ingest(mode=mode, force=args.force)
        print(f"Done. {retriever._collection.count()} chunks now indexed in {CHROMA_DIR}")


if __name__ == "__main__":
    main()
