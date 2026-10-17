#!/usr/bin/env python3
"""Fetch GitHub Actions logs for the latest run of a branch.

The REST log endpoint answers with a 302 to a signed blob URL that rejects the
Authorization header, so a plain authenticated fetch returns 401. This script
follows the redirect and fetches the signed URL *without* the header.

Usage:
    python tools/ci_logs.py                      # latest run on dev
    python tools/ci_logs.py --branch main
    python tools/ci_logs.py --run 12345 --job "full test" --tail 150
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gh import GitHub, load_repo, load_token  # noqa: E402


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _get(url: str, token: str | None) -> tuple[int, bytes, dict]:
    headers = {"User-Agent": "myarch-agent", "Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    opener = urllib.request.build_opener(_NoRedirect) if token else urllib.request.build_opener()
    req = urllib.request.Request(url, headers=headers)
    try:
        resp = opener.open(req, timeout=60)
        return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers)


def download_job_log(repo: str, job_id: int, token: str) -> str:
    status, body, headers = _get(
        "https://api.github.com/repos/%s/actions/jobs/%d/logs" % (repo, job_id), token
    )
    if status in (301, 302, 303, 307, 308):
        location = headers.get("Location")
        if not location:
            raise SystemExit("redirect without a Location header")
        # The signed URL must be fetched WITHOUT the Authorization header.
        status, body, _ = _get(location, None)
    if status != 200:
        raise SystemExit("could not download log [%s]: %s" % (status, body[:300]))
    return body.decode(errors="replace")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and print GitHub Actions job logs")
    parser.add_argument("--repo", default=None)
    parser.add_argument("--branch", default="dev")
    parser.add_argument("--run", type=int, default=None, help="run id (default: latest on branch)")
    parser.add_argument("--job", default=None, help="substring of the job name")
    parser.add_argument("--tail", type=int, default=120, help="lines to print")
    parser.add_argument("--list", action="store_true", help="only list runs and jobs")
    args = parser.parse_args(argv)

    repo = args.repo or load_repo()
    gh = GitHub(load_token(), repo)

    run_id = args.run
    if run_id is None:
        status, data = gh.request(
            "GET", "/repos/%s/actions/runs?branch=%s&per_page=1" % (repo, args.branch)
        )
        runs = data.get("workflow_runs", []) if isinstance(data, dict) else []
        if not runs:
            print("no runs found for branch %s" % args.branch)
            return 1
        run_id = runs[0]["id"]
        print(
            "run %s [%s] %s/%s %s"
            % (run_id, runs[0]["head_branch"], runs[0]["status"], runs[0]["conclusion"], runs[0]["html_url"])
        )

    status, jobs = gh.request("GET", "/repos/%s/actions/runs/%s/jobs" % (repo, run_id))
    entries = jobs.get("jobs", []) if isinstance(jobs, dict) else []

    if args.list or not args.job:
        for job in entries:
            print("  job %-28s %s/%s" % (job["name"], job["status"], job["conclusion"]))
            for step in job.get("steps", []):
                mark = "ok " if step["conclusion"] == "success" else ">> "
                print("      %s%s -> %s" % (mark, step["name"], step["conclusion"]))
        if args.list:
            return 0

    token = load_token()
    targets = [j for j in entries if args.job and args.job.lower() in j["name"].lower()]
    if not targets:
        targets = [j for j in entries if j["conclusion"] not in (None, "success", "skipped")] or entries

    for job in targets:
        print("\n" + "=" * 70)
        print("job: %s (%s/%s)" % (job["name"], job["status"], job["conclusion"]))
        print("=" * 70)
        log = download_job_log(repo, job["id"], token)
        lines = log.splitlines()
        if args.tail and len(lines) > args.tail:
            print("... %d earlier lines omitted ..." % (len(lines) - args.tail))
        print("\n".join(lines[-args.tail :] if args.tail else lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
