#!/usr/bin/env bash
# Publish a built docs site to a folder of the gh-pages branch, or remove that folder.
#
#   publish.sh main site/                 # docs for main
#   publish.sh v0.15.0 site/              # docs for a release
#   publish.sh pr-preview/pr-123 site/    # PR preview
#   publish.sh pr-preview/pr-123          # remove a PR preview
#
# Run from a checkout of this repo whose credentials can push. gh-pages is kept as
# a single commit so it does not grow with every publish. --force-with-lease makes
# concurrent publishes retry on top of each other instead of overwriting.
set -euo pipefail

dest=$1
site=${2:-}
if [[ ! "$dest" =~ ^(main|v[0-9]+\.[0-9]+\.[0-9]+|pr-preview/pr-[0-9]+)$ ]]; then
	echo "refusing to publish to '$dest'" >&2
	exit 1
fi
if [ -n "$site" ]; then
	site=$(cd "$site" && pwd)
fi

pages=$(mktemp -d)/gh-pages
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

for attempt in 1 2 3 4 5; do
	git worktree remove --force "$pages" 2>/dev/null || true
	if git ls-remote --exit-code --heads origin gh-pages >/dev/null; then
		git fetch --quiet --depth 1 origin gh-pages
		lease=$(git rev-parse FETCH_HEAD)
		git worktree add --quiet --detach "$pages" FETCH_HEAD
	else
		lease=""
		git worktree add --quiet --detach "$pages"
		git -C "$pages" rm -rq --ignore-unmatch .
	fi

	rm -rf "${pages:?}/$dest"
	if [ -n "$site" ]; then
		mkdir -p "$(dirname "$pages/$dest")"
		cp -R "$site" "$pages/$dest"
	fi

	# Root files: version list for the version picker, redirect to main, shared 404.
	python3 - "$pages" <<'EOF'
import json, re, shutil, sys
from pathlib import Path

pages = Path(sys.argv[1])
releases = [p.name for p in pages.iterdir() if re.fullmatch(r"v\d+\.\d+\.\d+", p.name)]
releases.sort(key=lambda v: [int(x) for x in v[1:].split(".")], reverse=True)
versions = (["main"] if (pages / "main").is_dir() else []) + releases
(pages / "versions.json").write_text(json.dumps(versions) + "\n")
(pages / "index.html").write_text(
    '<!doctype html>\n<meta charset="utf-8">\n<title>OpenEnv docs</title>\n'
    '<meta http-equiv="refresh" content="0; url=main/">\n<a href="main/">OpenEnv docs</a>\n'
)
if (pages / "main" / "404.html").exists():
    shutil.copy(pages / "main" / "404.html", pages / "404.html")
(pages / ".nojekyll").touch()
EOF

	git -C "$pages" add -A
	tree=$(git -C "$pages" write-tree)
	if [ -n "$lease" ] && [ "$tree" = "$(git rev-parse "$lease^{tree}")" ]; then
		echo "gh-pages already up to date"
		exit 0
	fi
	commit=$(git commit-tree "$tree" -m "Publish docs ($dest)")
	if git push --force-with-lease="refs/heads/gh-pages:$lease" origin "$commit:refs/heads/gh-pages"; then
		echo "updated $dest"
		exit 0
	fi
	echo "gh-pages moved, retrying ($attempt)"
	sleep $((attempt * 5))
done
echo "could not publish $dest" >&2
exit 1
