"""Generate the Grimoire `repositories:` config for the MLOps analytics repos.

Clones canonical/canonical-repo-automation, finds every repo under
`groups/charm-engineering/analytics/repos` that includes `mlops-settings.hcl`,
and tracks the branches listed in each repo's
`.github/automatic_backport_tracks.yaml`.

Usage:
    uv run python scripts/generate_mlops_repos.py [-o repos.yaml]

A GitHub token is read from `GITHUB_TOKEN`, falling back to `gh auth token`.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx
import yaml

AUTOMATION_REPO = "https://github.com/canonical/canonical-repo-automation"
REPOS_DIR = Path("groups/charm-engineering/analytics/repos")
ORG = "canonical"
SETTINGS_MARKER = re.compile(r'find_in_parent_folders\(\s*"mlops-settings\.hcl"\s*\)')
REPO_INPUT = re.compile(r'^\s*repo\s*=\s*"([^"]+)"', re.MULTILINE)
BACKPORT_FILE = ".github/automatic_backport_tracks.yaml"


class _ConfigDumper(yaml.SafeDumper):
    """Block-style mappings with inline string lists, like config.yaml."""


_ConfigDumper.add_representer(
    list,
    lambda dumper, data: dumper.represent_sequence(
        "tag:yaml.org,2002:seq", data, flow_style=all(isinstance(i, str) for i in data)
    ),
)


def clone(dest: Path) -> None:
    subprocess.run(
        ["git", "clone", "--depth", "1", "--quiet", AUTOMATION_REPO, str(dest)], check=True
    )


def find_repo_names(checkout: Path) -> list[str]:
    """Return `canonical/<repo>` for every terragrunt.hcl that uses mlops-settings.hcl."""
    names: set[str] = set()
    for hcl in sorted((checkout / REPOS_DIR).glob("*/terragrunt.hcl")):
        content = hcl.read_text()
        if not SETTINGS_MARKER.search(content):
            continue
        match = REPO_INPUT.search(content)
        if match is None:
            print(f"warning: no `repo` input in {hcl}", file=sys.stderr)
            continue
        names.add(f"{ORG}/{match.group(1)}")
    return sorted(names)


def github_token() -> str | None:
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    try:
        result = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, check=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def backport_tracks(client: httpx.Client, full_name: str) -> list[str]:
    """Return the tracks listed in the repo's backport file, or [] if it doesn't exist."""
    resp = client.get(
        f"/repos/{full_name}/contents/{BACKPORT_FILE}",
        headers={"Accept": "application/vnd.github.raw"},
    )
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    data = yaml.safe_load(resp.text) or {}
    return [str(t) for t in data.get("automatic_backport_tracks") or []]


def build_config(repo_names: list[str], token: str | None) -> list[dict[str, object]]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    entries: list[dict[str, object]] = []
    with httpx.Client(base_url="https://api.github.com", headers=headers, timeout=30) as client:
        for name in repo_names:
            entry: dict[str, object] = {"repo": name}
            tracks = backport_tracks(client, name)
            if tracks:
                entry["branches"] = tracks
            else:
                print(f"note: {name} has no backport tracks, default branch only", file=sys.stderr)
            entries.append(entry)
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-o", "--output", type=Path, help="write YAML here instead of stdout")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        checkout = Path(tmp) / "canonical-repo-automation"
        clone(checkout)
        repo_names = find_repo_names(checkout)

    print(f"found {len(repo_names)} repositories", file=sys.stderr)
    entries = build_config(repo_names, github_token())
    rendered = yaml.dump(
        {"repositories": entries}, Dumper=_ConfigDumper, sort_keys=False, default_flow_style=False
    )

    if args.output:
        args.output.write_text(rendered)
    else:
        sys.stdout.write(rendered)


if __name__ == "__main__":
    main()
