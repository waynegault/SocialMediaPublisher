"""LinkedIn diagnostics utility.

Features:
- Loads env vars from .env
- Optionally exchanges an authorization code for an access token
- Tests token validity by calling /v2/userinfo (OpenID)
- Attempts to fetch email address (requires r_emailaddress)
- Attempts to register an image upload to check w_member_social
- Optionally writes obtained access token to .env (use --save)

Usage:
    python linkedin_diagnostics.py [--save]

Security:
- Tokens are masked in output. If you choose to save a token to `.env`, it will be written only if you pass --save.
"""

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import os
import requests
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)

from config import Config


def mask(tok: str | None) -> str:
    if not tok:
        return "NOT SET"
    t = tok.strip()
    return t[:6] + "..." + t[-4:] if len(t) > 10 else "***"


def exchange_code(
    code: str, client_id: str, client_secret: str, redirect_uri: str
) -> dict | None:
    url = "https://www.linkedin.com/oauth/v2/accessToken"
    payload = {
        "grant_type": "authorization_code",
        "code": code.strip(),
        "redirect_uri": redirect_uri.strip(),
        "client_id": client_id.strip(),
        "client_secret": client_secret.strip(),
    }
    try:
        r = requests.post(url, data=payload, timeout=20)
        return {"status": r.status_code, "body": safe_json(r)}
    except Exception as e:
        return {"status": None, "error": str(e)}


def safe_json(resp: requests.Response):
    try:
        return resp.json()
    except Exception:
        return resp.text


def me_request(token: str) -> dict:
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Restli-Protocol-Version": "2.0.0",
    }
    url = "https://api.linkedin.com/v2/userinfo"
    r = requests.get(url, headers=headers, timeout=15)
    return {"status": r.status_code, "body": safe_json(r), "headers": dict(r.headers)}


def email_request(token: str) -> dict:
    headers = {"Authorization": f"Bearer {token}", "X-Restli-Protocol-Version": "2.0.0"}
    url = "https://api.linkedin.com/v2/emailAddress?q=members&projection=(elements*(handle~))"
    r = requests.get(url, headers=headers, timeout=15)
    return {"status": r.status_code, "body": safe_json(r)}


def register_upload(token: str, author_urn: str) -> dict:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0",
    }
    payload = {
        "registerUploadRequest": {
            "recipes": ["urn:li:digitalmediaRecipe:feedshare-image"],
            "owner": author_urn,
            "serviceRelationships": [
                {
                    "relationshipType": "OWNER",
                    "identifier": "urn:li:userGeneratedContent",
                }
            ],
        }
    }
    r = requests.post(
        "https://api.linkedin.com/v2/assets?action=registerUpload",
        headers=headers,
        json=payload,
        timeout=20,
    )
    return {"status": r.status_code, "body": safe_json(r)}


def get_admin_orgs(token: str) -> dict:
    """Return organizations the token-holder administrates (requires w_organization_social)."""
    headers = {"Authorization": f"Bearer {token}", "X-Restli-Protocol-Version": "2.0.0"}
    url = "https://api.linkedin.com/v2/organizationalEntityAcls?q=roleAssignee&role=ADMINISTRATOR&state=APPROVED&projection=(elements*(organizationalTarget,organizationalTarget~(localizedName)))"
    r = requests.get(url, headers=headers, timeout=15)
    return {"status": r.status_code, "body": safe_json(r), "headers": dict(r.headers)}


def write_token_to_env(token: str, env_path: Path) -> None:
    # Append or replace LINKEDIN_ACCESS_TOKEN in .env
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save obtained token to .env (if exchange succeeds)",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Provide an access token directly (avoid putting tokens in chat)",
    )
    parser.add_argument(
        "--prompt", action="store_true", help="Prompt for an access token interactively"
    )
    args = parser.parse_args()

    env_path = Path(__file__).parent / ".env"

    # Read env
    token = args.token or os.getenv("LINKEDIN_ACCESS_TOKEN")
    code = os.getenv("LINKEDIN_ACCESS_TOKEN_CODE")
    client_id = os.getenv("LINKEDIN_CLIENT_ID")
    client_secret = os.getenv("LINKEDIN_CLIENT_ID_SECRET")
    redirect_uri = os.getenv("LINKEDIN_REDIRECT_URI")
    author_urn = os.getenv("LINKEDIN_AUTHOR_URN") or Config.LINKEDIN_AUTHOR_URN

    # If user requested an interactive prompt for the token
    if args.prompt and not token:
        import getpass

        try:
            t = getpass.getpass(prompt="Enter LinkedIn access token (input hidden): ")
            token = t.strip() if t else None
        except Exception:
            # Fallback to visible input if getpass fails in the environment
            t = input("Enter LinkedIn access token: ")
            token = t.strip() if t else None

    print("=== LinkedIn Diagnostics ===")
    print("Author URN:", author_urn or "NOT SET")
    print("Access token (masked):", mask(token))
    print("Auth code present:", "YES" if code and code.strip() else "NO")

    if not token and code and client_id and client_secret and redirect_uri:
        print("Attempting to exchange authorization code...")
        res = exchange_code(code, client_id, client_secret, redirect_uri)
        if res:
            print("Exchange status:", res.get("status"))
            print("Exchange body:", json.dumps(res.get("body"), indent=2))
            body = res.get("body")
            if (
                res.get("status") == 200
                and isinstance(body, dict)
                and body.get("access_token")
            ):
                token = str(body.get("access_token")).strip()
                expires_in = body.get("expires_in")
                exp_ts = (
                    datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))
                    if expires_in
                    else None
                )
                print("Obtained access token (masked):", mask(token))
                if exp_ts:
                    print(
                        "Expires in:",
                        expires_in,
                        "seconds (about",
                        exp_ts.isoformat(),
                        "UTC)",
                    )
            if args.save and token:
                write_token_to_env(token, env_path)
                print("Saved token to .env")

    if not token:
        print(
            "No access token available. Provide one in .env or set LINKEDIN_ACCESS_TOKEN env var."
        )
        return

    token = token.strip()

    # Test /userinfo (OpenID)
    print("\n--- Testing /v2/userinfo (OpenID) ---")
    me = me_request(token)
    print("Status:", me["status"])
    print("Body:", json.dumps(me["body"], indent=2))

    # Compare the configured author URN to the profile ID and suggest correct URNs
    try:
        if me.get("status") == 200 and isinstance(me.get("body"), dict):
            profile_body = me["body"]
            profile_id = profile_body.get("id") or profile_body.get("sub")
            if profile_id:
                print("\n--- Author URN check ---")
                print("Profile ID:", profile_id)
                candidate_person = f"urn:li:person:{profile_id}"
                candidate_member = f"urn:li:member:{profile_id}"
                print("Suggested author URNs:")
                print("  person:", candidate_person)
                print("  member:", candidate_member)
                if author_urn and author_urn.strip() in (
                    candidate_person,
                    candidate_member,
                ):
                    print("Configured author URN matches the profile ID ✅")
                else:
                    print("Configured author URN does not match the profile ID ❌")
                    if author_urn:
                        print("  Configured:", author_urn)
                    print(
                        "  If this is your account, set LINKEDIN_AUTHOR_URN to one of the suggested URNs in your .env"
                    )
            else:
                print(
                    "Profile response did not include an 'id' or 'sub' field; cannot suggest URN."
                )
    except Exception as _:
        # Non-fatal; continue with other diagnostics
        pass

    # Test email address retrieval
    print("\n--- Testing /v2/emailAddress ---")
    email = email_request(token)
    print("Status:", email["status"])
    print("Body:", json.dumps(email["body"], indent=2))

    # Test register upload (w_member_social check)
    if not author_urn:
        print("\nNo author URN provided; cannot test upload registration.")
    else:
        print("\n--- Testing upload registration (checks w_member_social) ---")
        reg = register_upload(token, author_urn)
        print("Status:", reg["status"])
        print("Body:", json.dumps(reg["body"], indent=2))

    # Check organizations for which this token-holder is an admin (requires w_organization_social)
    print("\n--- Checking organization admin ACLs (organizationalEntityAcls) ---")
    admin = get_admin_orgs(token)
    print("Status:", admin.get("status"))
    print("Body:", json.dumps(admin.get("body"), indent=2))

    # Parse and suggest organization URNs
    suggested_orgs = []
    try:
        body = admin.get("body")
        if isinstance(body, dict) and body.get("elements"):
            for el in body.get("elements", []):
                org_urn = el.get("organizationalTarget")
                org_name = None
                if isinstance(el.get("organizationalTarget~"), dict):
                    org_name = el["organizationalTarget~"].get("localizedName")
                if org_urn:
                    suggested_orgs.append({"urn": org_urn, "name": org_name})

    except Exception:
        pass

    if suggested_orgs:
        print("\nSuggested organization URNs for which token-holder is admin:")
        for o in suggested_orgs:
            print(f"  {o['urn']}  ({o['name'] or 'name unknown'})")
        # Check whether configured author_urn matches one of the org URNs
        matches = [
            o for o in suggested_orgs if author_urn and author_urn.strip() == o["urn"]
        ]
        if matches:
            print("Configured author URN matches an organization you administer ✅")
        else:
            print(
                "Configured author URN does not match any admin organizations listed ❌"
            )
            if author_urn:
                print("  Configured:", author_urn)
            print(
                "  If you want to post as an organization, set LINKEDIN_AUTHOR_URN to one of the suggested org URNs and ensure your token has w_organization_social."
            )
    else:
        print(
            "\nNo admin organizations found for this token-holder. This may indicate the token lacks w_organization_social scope or the member is not an admin of any organizations."
        )

    print("\nDiagnostics complete.")


if __name__ == "__main__":
    main()
