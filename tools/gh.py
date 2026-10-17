#!/usr/bin/env python3
"""Reusable GitHub REST API client for the myarch sprint workflow.

The DeepSeek sandbox may block `curl`; plain Python networking is not blocked,
so every GitHub interaction in this project goes through this module.

Usage:
    python tools/gh.py whoami
    python tools/gh.py issue create --title "..." --body-file /tmp/body.md
    python tools/gh.py pr create --head dev --base main --title "..." --draft
    python tools/gh.py runs --branch dev
    python tools/gh.py comment --issue 3 --body-file /tmp/body.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

API_ROOT = "https://api.github.com"
REPO_ROOT = Path(__file__).resolve().parent.parent


def load_token() -> str:
    """Resolve the token from the environment, then from gitignored secret files."""
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token.strip()
    for name in (".secrets.env", ".env"):
        path = REPO_ROOT / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("GITHUB_TOKEN="):
                return line.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit("GITHUB_TOKEN not found in environment, .secrets.env or .env")


def load_repo() -> str:
    explicit = os.environ.get("GITHUB_REPO")
    if explicit:
        return explicit.strip()
    path = REPO_ROOT / ".secrets.env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("GITHUB_REPO="):
                return line.split("=", 1)[1].strip().strip("'\"")
    return "fufurobot/myarch"


class GitHub:
    def __init__(self, token: str, repo: str) -> None:
        self.token = token
        self.repo = repo

    def request(self, method: str, path: str, payload: dict | None = None) -> tuple[int, object]:
        url = path if path.startswith("http") else API_ROOT + path
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            url,
            method=method,
            data=data,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "myarch-agent",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                raw = resp.read()
                return resp.status, (json.loads(raw) if raw else {})
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode(errors="replace")
            try:
                return exc.code, json.loads(raw)
            except json.JSONDecodeError:
                return exc.code, raw

    # ---- convenience wrappers -------------------------------------------------

    def create_issue(self, title: str, body: str, labels: list[str] | None = None) -> dict:
        status, data = self.request(
            "POST", f"/repos/{self.repo}/issues", {"title": title, "body": body, "labels": labels or []}
        )
        if status >= 300:
            raise SystemExit(f"issue create failed [{status}]: {data}")
        return data  # type: ignore[return-value]

    def comment(self, number: int, body: str) -> dict:
        status, data = self.request(
            "POST", f"/repos/{self.repo}/issues/{number}/comments", {"body": body}
        )
        if status >= 300:
            raise SystemExit(f"comment failed [{status}]: {data}")
        return data  # type: ignore[return-value]

    def create_pr(self, head: str, base: str, title: str, body: str, draft: bool = True) -> dict:
        status, data = self.request(
            "POST",
            f"/repos/{self.repo}/pulls",
            {"head": head, "base": base, "title": title, "body": body, "draft": draft},
        )
        if status >= 300:
            raise SystemExit(f"pr create failed [{status}]: {data}")
        return data  # type: ignore[return-value]

    def ready_pr(self, number: int) -> dict:
        """Mark a draft PR ready for review (GraphQL, as REST has no field for it)."""
        query = (
            "mutation($id:ID!){markPullRequestReadyForReview(input:{pullRequestId:$id})"
            "{pullRequest{number isDraft}}}"
        )
        node = self.request("GET", f"/repos/{self.repo}/pulls/{number}")[1]["node_id"]  # type: ignore[index]
        return self.graphql(query, {"id": node})

    def merge_pr(self, number: int, method: str = "merge") -> dict:
        status, data = self.request("PUT", f"/repos/{self.repo}/pulls/{number}/merge", {"merge_method": method})
        if status >= 300:
            raise SystemExit(f"pr merge failed [{status}]: {data}")
        return data  # type: ignore[return-value]

    def graphql(self, query: str, variables: dict) -> dict:
        status, data = self.request("POST", "/graphql", {"query": query, "variables": variables})
        if status >= 300 or (isinstance(data, dict) and data.get("errors")):
            raise SystemExit(f"graphql failed [{status}]: {data}")
        return data  # type: ignore[return-value]


def read_body(args: argparse.Namespace) -> str:
    if args.body_file:
        return Path(args.body_file).read_text(encoding="utf-8")
    return args.body or ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GitHub API helper for myarch sprints")
    parser.add_argument("--repo", default=None, help="owner/name (default: from .secrets.env)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("whoami")
    sub.add_parser("branches")
    sub.add_parser("issues")
    sub.add_parser("prs")

    p = sub.add_parser("issue"); p.add_argument("action", choices=["create", "comment", "close"])
    p.add_argument("--title"); p.add_argument("--body"); p.add_argument("--body-file")
    p.add_argument("--number", type=int); p.add_argument("--labels", default="")

    p = sub.add_parser("pr"); p.add_argument("action", choices=["create", "ready", "merge", "view"])
    p.add_argument("--head"); p.add_argument("--base", default="dev")
    p.add_argument("--title"); p.add_argument("--body"); p.add_argument("--body-file")
    p.add_argument("--number", type=int); p.add_argument("--no-draft", action="store_true")
    p.add_argument("--method", default="merge", choices=["merge", "squash", "rebase"])

    p = sub.add_parser("comment"); p.add_argument("--issue", type=int, required=True)
    p.add_argument("--body"); p.add_argument("--body-file")

    p = sub.add_parser("runs"); p.add_argument("--branch", default=None)
    p.add_argument("--limit", type=int, default=5)

    p = sub.add_parser("workflow"); p.add_argument("action", choices=["list", "dispatch", "enable", "disable"])
    p.add_argument("--name"); p.add_argument("--ref", default="dev")

    p = sub.add_parser("raw"); p.add_argument("method"); p.add_argument("path")
    p.add_argument("--payload")

    args = parser.parse_args(argv)
    gh = GitHub(load_token(), args.repo or load_repo())

    if args.cmd == "whoami":
        print(json.dumps(gh.request("GET", "/user")[1], indent=2)[:600])
    elif args.cmd == "branches":
        _, data = gh.request("GET", f"/repos/{gh.repo}/branches")
        print([b["name"] for b in data])  # type: ignore[union-attr]
    elif args.cmd == "issues":
        _, data = gh.request("GET", f"/repos/{gh.repo}/issues?state=all&per_page=50")
        for i in data:  # type: ignore[union-attr]
            kind = "PR" if "pull_request" in i else "ISSUE"
            print(f"#{i['number']} [{kind}] {i['state']} {i['title']}")
    elif args.cmd == "prs":
        _, data = gh.request("GET", f"/repos/{gh.repo}/pulls?state=all&per_page=50")
        for i in data:  # type: ignore[union-attr]
            print(f"#{i['number']} {i['state']} draft={i['draft']} {i['head']['ref']}->{i['base']['ref']} {i['title']}")
    elif args.cmd == "issue":
        if args.action == "create":
            out = gh.create_issue(args.title, read_body(args), [x for x in args.labels.split(",") if x])
            print(f"created issue #{out['number']}: {out['html_url']}")
        elif args.action == "comment":
            out = gh.comment(args.number, read_body(args))
            print(f"commented on #{args.number}: {out['html_url']}")
        else:
            status, out = gh.request("PATCH", f"/repos/{gh.repo}/issues/{args.number}", {"state": "closed"})
            print(f"closed #{args.number} [{status}]")
    elif args.cmd == "pr":
        if args.action == "create":
            out = gh.create_pr(args.head, args.base, args.title, read_body(args), draft=not args.no_draft)
            print(f"created PR #{out['number']} draft={out['draft']}: {out['html_url']}")
        elif args.action == "ready":
            print(json.dumps(gh.ready_pr(args.number)))
        elif args.action == "merge":
            print(json.dumps(gh.merge_pr(args.number, args.method)))
        else:
            print(json.dumps(gh.request("GET", f"/repos/{gh.repo}/pulls/{args.number}")[1], indent=2)[:1200])
    elif args.cmd == "comment":
        out = gh.comment(args.issue, read_body(args))
        print(f"commented: {out['html_url']}")
    elif args.cmd == "runs":
        q = f"/repos/{gh.repo}/actions/runs?per_page={args.limit}"
        if args.branch:
            q += f"&branch={args.branch}"
        _, data = gh.request("GET", q)
        runs = data.get("workflow_runs", [])  # type: ignore[union-attr]
        if not runs:
            print("no workflow runs")
        for r in runs:
            print(f"{r['id']} {r['name']} [{r['head_branch']}] {r['status']}/{r['conclusion']} {r['html_url']}")
    elif args.cmd == "workflow":
        if args.action == "list":
            _, data = gh.request("GET", f"/repos/{gh.repo}/actions/workflows")
            for w in data.get("workflows", []):  # type: ignore[union-attr]
                print(f"{w['id']} {w['name']} {w['state']} {w['path']}")
        elif args.action == "dispatch":
            status, data = gh.request(
                "POST", f"/repos/{gh.repo}/actions/workflows/{args.name}/dispatches", {"ref": args.ref}
            )
            print(f"dispatch {args.name}@{args.ref} -> {status} {data}")
        else:
            state = "true" if args.action == "enable" else "false"
            status, data = gh.request(
                "PUT", f"/repos/{gh.repo}/actions/workflows/{args.name}/{args.action}", {"state": state}
            )
            print(f"{args.action} {args.name} -> {status}")
    else:
        payload = json.loads(args.payload) if args.payload else None
        status, data = gh.request(args.method.upper(), args.path, payload)
        print(f"[{status}]")
        print(json.dumps(data, indent=2)[:4000] if isinstance(data, (dict, list)) else data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
