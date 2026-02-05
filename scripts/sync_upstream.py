#!/usr/bin/env python3
"""
Sync vendored repositories with their upstream sources.

Usage:
    python scripts/sync_upstream.py [--repo REPO_NAME] [--all] [--dry-run]

Examples:
    python scripts/sync_upstream.py --repo deep_ep      # Sync single repo
    python scripts/sync_upstream.py --all               # Sync all repos
    python scripts/sync_upstream.py --all --dry-run     # Preview changes
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
MANIFEST_PATH = ROOT_DIR / "VENDOR_MANIFEST.json"

# Files/directories to preserve during sync (not overwritten from upstream)
PRESERVE_PATTERNS = [
    "DEEP_DIVE.md",
    "TECHNICAL_DEEP_DIVE.md",
]


def load_manifest():
    """Load the vendor manifest."""
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def save_manifest(manifest):
    """Save the vendor manifest."""
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def get_all_repos(manifest):
    """Get all repository names from manifest."""
    repos = {}
    for category in ["model", "infra"]:
        for name, info in manifest["repositories"].get(category, {}).items():
            repos[name] = {"category": category, **info}
    return repos


def run_cmd(cmd, cwd=None, check=True):
    """Run a shell command."""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result


def get_upstream_info(repo_path):
    """Get the latest commit info from a cloned repo."""
    result = run_cmd(["git", "rev-parse", "HEAD"], cwd=repo_path, check=True)
    commit = result.stdout.strip()
    
    result = run_cmd(["git", "log", "-1", "--format=%ci"], cwd=repo_path, check=True)
    commit_date = result.stdout.strip()
    
    return commit, commit_date


def sync_repo(repo_name, repo_info, dry_run=False):
    """Sync a single repository with upstream."""
    upstream = repo_info["upstream"]
    local_path = ROOT_DIR / repo_name
    old_commit = repo_info["commit"]
    
    print(f"\n{'='*60}")
    print(f"Syncing: {repo_name}")
    print(f"  Upstream: {upstream}")
    print(f"  Current commit: {old_commit[:12]}")
    print(f"{'='*60}")
    
    if not local_path.exists():
        print(f"  WARNING: Local path {local_path} does not exist, skipping")
        return None
    
    # Create temp directory for clone
    with tempfile.TemporaryDirectory() as tmp_dir:
        clone_path = Path(tmp_dir) / repo_name
        
        # Clone upstream
        print(f"\n  Cloning upstream...")
        run_cmd(["git", "clone", "--depth=1", upstream, str(clone_path)])
        
        # Get new commit info
        new_commit, new_commit_date = get_upstream_info(clone_path)
        print(f"  Latest upstream commit: {new_commit[:12]} ({new_commit_date})")
        
        if new_commit == old_commit:
            print(f"  Already up to date!")
            return None
        
        if dry_run:
            print(f"  [DRY RUN] Would sync from {old_commit[:12]} to {new_commit[:12]}")
            return {
                "commit": new_commit,
                "commit_date": new_commit_date,
            }
        
        # Preserve local-only files
        preserved = {}
        for pattern in PRESERVE_PATTERNS:
            local_file = local_path / pattern
            if local_file.exists():
                preserved[pattern] = local_file.read_text()
                print(f"  Preserving: {pattern}")
        
        # Remove .git from clone
        shutil.rmtree(clone_path / ".git")
        
        # Sync files using rsync
        print(f"\n  Syncing files...")
        run_cmd([
            "rsync", "-av", "--delete",
            "--exclude=DEEP_DIVE.md",
            "--exclude=TECHNICAL_DEEP_DIVE.md",
            f"{clone_path}/",
            f"{local_path}/"
        ])
        
        # Restore preserved files
        for pattern, content in preserved.items():
            (local_path / pattern).write_text(content)
            print(f"  Restored: {pattern}")
        
        print(f"\n  Successfully synced {repo_name}")
        print(f"  {old_commit[:12]} -> {new_commit[:12]}")
        
        return {
            "commit": new_commit,
            "commit_date": new_commit_date,
        }


def main():
    parser = argparse.ArgumentParser(description="Sync vendored repos with upstream")
    parser.add_argument("--repo", help="Sync specific repository")
    parser.add_argument("--all", action="store_true", help="Sync all repositories")
    parser.add_argument("--dry-run", action="store_true", help="Preview without making changes")
    parser.add_argument("--list", action="store_true", help="List all vendored repos")
    args = parser.parse_args()
    
    manifest = load_manifest()
    all_repos = get_all_repos(manifest)
    
    if args.list:
        print("\nVendored Repositories:")
        print("-" * 60)
        for name, info in sorted(all_repos.items()):
            print(f"  {name:<20} [{info['category']}] {info['commit'][:12]}")
        return
    
    if not args.repo and not args.all:
        parser.print_help()
        return
    
    # Determine which repos to sync
    if args.all:
        repos_to_sync = all_repos
    else:
        if args.repo not in all_repos:
            print(f"ERROR: Unknown repository '{args.repo}'")
            print(f"Available: {', '.join(sorted(all_repos.keys()))}")
            return
        repos_to_sync = {args.repo: all_repos[args.repo]}
    
    # Sync repos
    updated = []
    for name, info in repos_to_sync.items():
        result = sync_repo(name, info, dry_run=args.dry_run)
        if result:
            updated.append((name, info["category"], result))
    
    # Update manifest
    if updated and not args.dry_run:
        print(f"\n{'='*60}")
        print("Updating manifest...")
        for name, category, new_info in updated:
            manifest["repositories"][category][name]["commit"] = new_info["commit"]
            manifest["repositories"][category][name]["commit_date"] = new_info["commit_date"]
        
        manifest["last_sync"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_manifest(manifest)
        print(f"Updated {len(updated)} repositories in manifest")
    
    # Summary
    print(f"\n{'='*60}")
    print("SYNC SUMMARY")
    print(f"{'='*60}")
    if updated:
        for name, category, new_info in updated:
            status = "[DRY RUN] " if args.dry_run else ""
            print(f"  {status}{name}: updated to {new_info['commit'][:12]}")
    else:
        print("  All repositories are up to date")


if __name__ == "__main__":
    main()
