"""Simple LinkedIn access test utility.

Features:
- Loads token from `.env` or from env var or prompts interactively
- Calls `GET /v2/userinfo` (OpenID) and prints masked profile info
- Optionally saves a provided token to `.env` with `--save`

Usage:
  python linkedin_test.py [--token <token>] [--prompt] [--save]

Notes:
- Provide the raw token (no leading "Bearer ").
- Tokens are masked in output.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
import requests


def _ensure_venv():
    proj_root = Path(__file__).parent
    venv_dir = proj_root / ".venv"
    if venv_dir.exists():
        venv_py = (
            venv_dir
            / ("Scripts" if os.name == "nt" else "bin")
            / ("python.exe" if os.name == "nt" else "python")
        )
        if venv_py.exists():
            venv_path = str(venv_py)
            if os.path.abspath(sys.executable) != os.path.abspath(venv_path):
                os.execv(venv_path, [venv_path] + sys.argv)


_ensure_venv()

from dotenv import load_dotenv


# Load .env from multiple candidate locations (script dir, cwd, parent) and report which was used
def _load_env_candidates():
    candidates = []
    # Respect DOTENV_PATH env var if provided
    dot = os.getenv("DOTENV_PATH")
    if dot:
        try:
            candidates.append(Path(dot))
        except Exception:
            pass
    candidates.extend(
        [
            Path(__file__).parent / ".env",
            Path.cwd() / ".env",
            Path(__file__).parent.parent / ".env",
        ]
    )

    found = None
    for p in candidates:
        if p is None:
            continue
        if p.exists():
            # Load but don't overwrite existing env vars unless user supplied --token
            load_dotenv(dotenv_path=p, override=False)
            found = p
            # keep loading other candidates so later ones can add missing vars
    return found


ENV_PATH = _load_env_candidates() or (Path(__file__).parent / ".env")
print("Loaded .env from:", str(ENV_PATH) if ENV_PATH.exists() else "(none found)")


def mask(tok: str | None) -> str:
    if not tok:
        return "NOT SET"
    t = tok.strip()
    return t[:6] + "..." + t[-4:] if len(t) > 10 else "***"


def safe_json(resp: requests.Response):
    try:
        return resp.json()
    except Exception:
        return resp.text


def write_token_to_env(token: str, env_path: Path) -> None:
    lines = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    new_lines = []
    replaced = False
    for line in lines:
        if line.strip().startswith("LINKEDIN_ACCESS_TOKEN="):
            new_lines.append(f"LINKEDIN_ACCESS_TOKEN={token}")
            replaced = True
        else:
            new_lines.append(line)
    if not replaced:
        new_lines.append(f"LINKEDIN_ACCESS_TOKEN={token}")
    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--token",
        type=str,
        help="Provide an access token directly (no 'Bearer ' prefix)",
    )
    parser.add_argument(
        "--prompt", action="store_true", help="Prompt for an access token interactively"
    )
    parser.add_argument(
        "--save", action="store_true", help="Save obtained/provided token to .env"
    )
    args = parser.parse_args(argv)

    token = args.token or os.getenv("LINKEDIN_ACCESS_TOKEN")

    if args.prompt and not token:
        try:
            # use getpass to avoid echoing
            import getpass

            t = getpass.getpass(prompt="Enter LinkedIn access token (input hidden): ")
            token = t.strip() if t else None
        except Exception:
            t = input("Enter LinkedIn access token: ")
            token = t.strip() if t else None

    if token:
        token = token.strip()

    print("=== LinkedIn quick test ===")
    print("LINKEDIN_ACCESS_TOKEN (masked):", mask(token))

    if not token:
        print(
            "No access token available. Provide one via --token, --prompt, or set LINKEDIN_ACCESS_TOKEN in .env"
        )
        return 2

    headers = {
        "Authorization": f"Bearer {token}",
        "X-Restli-Protocol-Version": "2.0.0",
    }

    # Call /v2/userinfo (OpenID)
    try:
        r = requests.get(
            "https://api.linkedin.com/v2/userinfo", headers=headers, timeout=15
        )
        print("/userinfo status:", r.status_code)
        body = safe_json(r)
        print(
            "/userinfo body:",
            json.dumps(body, indent=2) if isinstance(body, dict) else body,
        )

        # If successful, optionally save token
        if args.save:
            write_token_to_env(token, ENV_PATH)
            print("Saved token to .env")

        return 0 if r.status_code == 200 else 3
    except Exception as e:
        print("Exception calling LinkedIn API:", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
