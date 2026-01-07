"""Test LinkedIn connection using values from .env.

This script will:
- Load .env
- If LINKEDIN_ACCESS_TOKEN is not set but LINKEDIN_ACCESS_TOKEN_CODE is, exchange the code for an access token
- Set the token on Config in-memory (won't modify .env)
- Run the LinkedInPublisher.test_connection() and retrieve profile info

It prints only masked tokens to avoid leaking secrets.
"""

import os
import sys
import json
import requests
from pathlib import Path

def _ensure_venv():
    proj_root = Path(__file__).parent
    venv_dir = proj_root / '.venv'
    if venv_dir.exists():
        venv_py = venv_dir / ('Scripts' if os.name == 'nt' else 'bin') / (
            'python.exe' if os.name == 'nt' else 'python'
        )
        if venv_py.exists():
            venv_path = str(venv_py)
            if os.path.abspath(sys.executable) != os.path.abspath(venv_path):
                os.execv(venv_path, [venv_path] + sys.argv)

_ensure_venv()

from dotenv import load_dotenv

# Load .env explicitly from project root
load_dotenv(dotenv_path=Path(__file__).parent / '.env', override=True)

from config import Config
from database import Database
from linkedin_publisher import LinkedInPublisher


def mask_token(tok: str) -> str:
    if not tok:
        return "NOT SET"
    return tok[:6] + "..." + tok[-4:]


def exchange_code_for_token(
    code: str, client_id: str, client_secret: str, redirect_uri: str
) -> str | None:
    url = "https://www.linkedin.com/oauth/v2/accessToken"
    payload = {
        "grant_type": "authorization_code",
        "code": code.strip(),
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "client_secret": client_secret,
    }
    try:
        resp = requests.post(url, data=payload, timeout=20)
        if resp.status_code != 200:
            print(f"Token exchange failed: {resp.status_code} {resp.text}")
            return None
        data = resp.json()
        return data.get("access_token")
    except Exception as e:
        print(f"Exception during token exchange: {e}")
        return None


def main():
    access_token = os.getenv("LINKEDIN_ACCESS_TOKEN")
    code = os.getenv("LINKEDIN_ACCESS_TOKEN_CODE")
    client_id = os.getenv("LINKEDIN_CLIENT_ID")
    client_secret = os.getenv("LINKEDIN_CLIENT_ID_SECRET")
    redirect_uri = os.getenv("LINKEDIN_REDIRECT_URI")

    # Fallback: parse .env file directly if variables are unexpectedly missing
    def _get_from_env_file(key: str) -> str | None:
        env_path = Path(__file__).parent / ".env"
        if not env_path.exists():
            return None
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == key:
                    return v.strip().strip('"')
        return None

    if not code:
        code = _get_from_env_file("LINKEDIN_ACCESS_TOKEN_CODE")
    if not client_id:
        client_id = _get_from_env_file("LINKEDIN_CLIENT_ID")
    if not client_secret:
        client_secret = _get_from_env_file("LINKEDIN_CLIENT_ID_SECRET")
    if not redirect_uri:
        redirect_uri = _get_from_env_file("LINKEDIN_REDIRECT_URI")

    print("Current LINKEDIN_AUTHOR_URN:", Config.LINKEDIN_AUTHOR_URN or "NOT SET")
    print(
        "LINKEDIN_ACCESS_TOKEN_CODE present:", "YES" if code and code.strip() else "NO"
    )

    # Debug: list LINKEDIN-related env var presence (masked)
    for k in (
        "LINKEDIN_ACCESS_TOKEN",
        "LINKEDIN_ACCESS_TOKEN_CODE",
        "LINKEDIN_CLIENT_ID",
        "LINKEDIN_CLIENT_ID_SECRET",
        "LINKEDIN_REDIRECT_URI",
    ):
        v = os.getenv(k)
        if v:
            print(f"  {k}: SET (masked: {mask_token(v)})")
        else:
            print(f"  {k}: NOT SET")

    # Strip whitespace from any env values (common source of issues)
    if access_token:
        access_token = access_token.strip()
    if code:
        code = code.strip()
    if client_id:
        client_id = client_id.strip()
    if client_secret:
        client_secret = client_secret.strip()
    if redirect_uri:
        redirect_uri = redirect_uri.strip()

    if not access_token and code and client_id and client_secret and redirect_uri:
        print(
            "No access token present, attempting to exchange authorization code for an access token..."
        )
        token = exchange_code_for_token(code, client_id, client_secret, redirect_uri)
        if not token:
            print("Failed to obtain access token.")
        else:
            token = token.strip()
            print("Obtained access token:", mask_token(token))
            # Set token in memory for this run (strip again)
            Config.LINKEDIN_ACCESS_TOKEN = token
    elif access_token:
        print("Using existing LINKEDIN_ACCESS_TOKEN:", mask_token(access_token))
        Config.LINKEDIN_ACCESS_TOKEN = access_token
    else:
        print(
            "No token or authorization code available. Please set LINKEDIN_ACCESS_TOKEN in .env or provide a valid code."
        )

    # Debug: ensure Config token is stripped and show masked info
    if Config.LINKEDIN_ACCESS_TOKEN:
        Config.LINKEDIN_ACCESS_TOKEN = Config.LINKEDIN_ACCESS_TOKEN.strip()
        tok = Config.LINKEDIN_ACCESS_TOKEN
        print(
            f"Config.LINKEDIN_ACCESS_TOKEN masked: {mask_token(tok)} (len={len(tok)})"
        )
    else:
        print("Config.LINKEDIN_ACCESS_TOKEN: NOT SET")

    # Run connection test
    try:
        db = Database()
        publisher = LinkedInPublisher(db)

        # Do a direct /userinfo request and print the response details for debugging
        try:
            headers = publisher._get_headers()
            resp = requests.get(
                f"{publisher.BASE_URL}/userinfo", headers=headers, timeout=15
            )
            print("Direct /userinfo request status:", resp.status_code)
            try:
                print("Direct /userinfo response body:", resp.json())
            except Exception:
                print("Direct /userinfo response text:", resp.text)
            # Print a few useful headers if present
            for h in ("www-authenticate", "x-li-request-id", "content-type"):
                if h in resp.headers:
                    print(f"Header {h}:", resp.headers[h])
        except Exception as e:
            print("Exception during direct /userinfo request:", e)

        ok = publisher.test_connection()

        # If the initial connection fails, try exchanging the authorization code (if available)
        if not ok and code and client_id and client_secret and redirect_uri:
            print(
                "Initial connection failed, attempting token exchange using authorization code..."
            )
            token = exchange_code_for_token(
                code, client_id, client_secret, redirect_uri
            )
            if token:
                token = token.strip()
                print("Obtained access token:", mask_token(token))
                Config.LINKEDIN_ACCESS_TOKEN = token
                # Recreate publisher with updated token
                publisher = LinkedInPublisher(db)

                # Retry direct /userinfo to show the new response
                try:
                    headers = publisher._get_headers()
                    resp = requests.get(
                        f"{publisher.BASE_URL}/userinfo", headers=headers, timeout=15
                    )
                    print("Direct /userinfo (after exchange) status:", resp.status_code)
                    try:
                        print("Direct /userinfo (after exchange) body:", resp.json())
                    except Exception:
                        print("Direct /userinfo (after exchange) text:", resp.text)
                except Exception as e:
                    print("Exception during direct /userinfo (after exchange):", e)

                ok = publisher.test_connection()
            else:
                print("Token exchange failed.")

        print("Connection test result:", "SUCCESS" if ok else "FAILED")
        if ok:
            profile = publisher.get_profile_info()
            if profile:
                # Try multiple possible name fields (OpenID vs legacy)
                name = " ".join(
                    part
                    for part in (
                        profile.get("localizedFirstName")
                        or profile.get("given_name")
                        or profile.get("name"),
                        profile.get("localizedLastName") or profile.get("family_name"),
                    )
                    if part
                ).strip()
                if not name:
                    name = (
                        profile.get("name")
                        or profile.get("localizedFirstName")
                        or profile.get("id", "")
                    )
                print("Profile name:", name)
                print("Profile raw:", json.dumps(profile, indent=2))
            else:
                print("Could not read profile info (unexpected response).")
    except Exception as e:
        print("Exception while testing LinkedIn connection:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
