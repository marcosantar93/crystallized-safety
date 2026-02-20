#!/usr/bin/env python3
"""
Open notebooks in Google Colab directly from the command line.

Usage:
    python scripts/open_in_colab.py v1522/v1522_critical_controls.ipynb
    python scripts/open_in_colab.py --list  # List all notebooks
"""

import subprocess
import sys
import os
from pathlib import Path

GITHUB_USER = "marcosantar93"
GITHUB_REPO = "paladin_claude"
GITHUB_BRANCH = "main"

def get_colab_url(notebook_path: str) -> str:
    """Generate Colab URL for a notebook."""
    # Remove leading ./ if present
    notebook_path = notebook_path.lstrip("./")
    return f"https://colab.research.google.com/github/{GITHUB_USER}/{GITHUB_REPO}/blob/{GITHUB_BRANCH}/{notebook_path}"

def open_in_browser(url: str):
    """Open URL in default browser (macOS)."""
    subprocess.run(["open", url], check=True)

def list_notebooks():
    """List all notebooks in the repo."""
    repo_root = Path(__file__).parent.parent
    notebooks = list(repo_root.glob("**/*.ipynb"))

    # Filter out checkpoints and archives
    notebooks = [
        n for n in notebooks
        if ".ipynb_checkpoints" not in str(n)
        and "_archive" not in str(n)
    ]

    print("Available notebooks:")
    print("-" * 60)
    for nb in sorted(notebooks):
        rel_path = nb.relative_to(repo_root)
        print(f"  {rel_path}")
    print("-" * 60)
    print(f"\nUsage: python scripts/open_in_colab.py <notebook_path>")

def main():
    if len(sys.argv) < 2:
        list_notebooks()
        sys.exit(0)

    if sys.argv[1] == "--list":
        list_notebooks()
        sys.exit(0)

    notebook_path = sys.argv[1]

    # Check if file exists
    repo_root = Path(__file__).parent.parent
    full_path = repo_root / notebook_path

    if not full_path.exists():
        print(f"Error: Notebook not found: {notebook_path}")
        print("Use --list to see available notebooks")
        sys.exit(1)

    # Check if there are uncommitted changes
    result = subprocess.run(
        ["git", "status", "--porcelain", notebook_path],
        capture_output=True, text=True, cwd=repo_root
    )

    if result.stdout.strip():
        print(f"Warning: {notebook_path} has uncommitted changes!")
        print("Colab will load the last pushed version.")
        response = input("Push changes first? [y/N] ")
        if response.lower() == "y":
            subprocess.run(["git", "add", notebook_path], cwd=repo_root)
            subprocess.run(["git", "commit", "-m", f"Update {notebook_path}"], cwd=repo_root)
            subprocess.run(["git", "push"], cwd=repo_root)
            print("Pushed!")

    url = get_colab_url(notebook_path)
    print(f"\nOpening in Colab:\n{url}\n")
    open_in_browser(url)

if __name__ == "__main__":
    main()
