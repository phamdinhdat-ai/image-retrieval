"""Inspect staged files before committing, without printing secret values."""
import subprocess
from pathlib import Path

from dotenv import dotenv_values


def main():
    staged = subprocess.check_output(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"], text=True).splitlines()
    secrets = []
    for source in [Path(".env"), Path("key_vilao.key")]:
        if source.exists():
            values = dotenv_values(source)
            secrets += [str(v) for k, v in values.items() if v and len(str(v)) >= 12 and any(t in k.lower() for t in ["key", "token", "secret", "password"])]
    failed = []
    for name in staged:
        path = Path(name)
        if path.suffix.lower() in (".key", ".pem") or (path.name.startswith(".env") and path.name != ".env.example"):
            failed.append(name)
            continue
        data = subprocess.check_output(["git", "show", ":" + name])
        if any(secret.encode() in data for secret in secrets):
            failed.append(name)
    if failed:
        raise SystemExit("Secret scan failed: " + ", ".join(failed))
    print(f"Secret scan passed: {len(staged)} staged files; no local credential values found.")


if __name__ == "__main__":
    main()
