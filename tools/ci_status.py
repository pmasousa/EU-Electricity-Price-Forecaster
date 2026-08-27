"""Print GitHub Actions run statuses for this repo.

Authentication comes from git's credential manager at runtime (the same token
git uses for push/pull). The token is never displayed, logged, or written to
disk — it lives in this process's memory and goes straight into the request
header. Usage: python tools/ci_status.py [n_runs]
"""

import json
import os
import subprocess
import sys
import urllib.request

REPO = "pmasousa/EU-Electricity-Price-Forecaster"


def _token_from_git() -> str:
    fill = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    if fill.returncode != 0:
        raise SystemExit("no stored github.com credential (git credential fill failed)")
    fields = dict(
        line.split("=", 1) for line in fill.stdout.splitlines() if "=" in line
    )
    token = fields.get("password", "")
    if not token:
        raise SystemExit("stored credential carries no token")
    return token


def main(n_runs: int = 5) -> None:
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/actions/runs?per_page={n_runs}",
        headers={
            "Authorization": f"Bearer {_token_from_git()}",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        runs = json.load(resp).get("workflow_runs", [])
    if not runs:
        print("no workflow runs yet")
        return
    for run in runs:
        icon = {"success": "OK  ", "failure": "FAIL", None: "...."}[run["conclusion"]] \
            if run["conclusion"] in (None, "success", "failure") else run["conclusion"]
        print(
            f"{icon}  {run['created_at']}  {run['head_branch']:<26} {run['head_sha'][:7]}  "
            f"{run['display_title'][:60]}"
        )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
