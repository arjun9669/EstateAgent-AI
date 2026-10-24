"""Copy the app to a Hugging Face Space.

    HF_TOKEN=... python scripts/deploy_space.py owner/space-name

Run by the deploy job in .github/workflows/ci.yml after tests pass. Only the
files the app needs are uploaded; deploy/space/README.md becomes the Space's
README (it holds the Space settings). Files no longer deployed are removed
from the Space too; huggingface_hub never deletes the Space's .gitattributes.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["app.py", "requirements.txt", "LICENSE"]
FOLDERS = ["src", "data/sample", "data/rules"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")


def stage(target: Path) -> list[str]:
    """Copy the deployable files into target; return their relative paths."""
    for name in FILES:
        shutil.copy2(ROOT / name, target / name)
    for folder in FOLDERS:
        shutil.copytree(ROOT / folder, target / folder, ignore=IGNORE)
    shutil.copy2(ROOT / "deploy" / "space" / "README.md", target / "README.md")
    return sorted(str(p.relative_to(target)) for p in target.rglob("*") if p.is_file())


def main(argv: list[str]) -> int:
    if len(argv) != 1 or "/" not in argv[0]:
        print(__doc__)
        return 2
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set.")
        return 1
    from huggingface_hub import HfApi

    with tempfile.TemporaryDirectory() as tmp:
        files = stage(Path(tmp))
        print(f"Uploading {len(files)} files to spaces/{argv[0]}")
        commit = HfApi(token=token).upload_folder(
            repo_id=argv[0],
            repo_type="space",
            folder_path=tmp,
            commit_message=f"Deploy {os.environ.get('GITHUB_SHA', 'local build')[:7]} from GitHub",
            # Remove Space files no longer deployed; .gitattributes is always kept.
            delete_patterns="*",
        )
    print(f"Done: {commit.commit_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
