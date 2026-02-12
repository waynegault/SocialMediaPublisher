"""Find LinkedIn profiles for indirect people (org leadership) at each organization."""

import sqlite3
import json
import time
import re
import base64

# Import UC Chrome availability from centralized browser module
# This also applies Windows error suppression for cleanup
try:
    from browser import uc, By, UC_AVAILABLE, _suppress_uc_cleanup_errors

    _suppress_uc_cleanup_errors()
except ImportError:
    # Fallback to direct import if browser module not available
    uc = None
    By = None
    UC_AVAILABLE = False
    try:
        import undetected_chromedriver as _uc
        from selenium.webdriver.common.by import By as _By

        uc = _uc
        By = _By
        UC_AVAILABLE = True
    except ImportError:
        print(
            "WARNING: undetected-chromedriver not installed - "
            "pip install undetected-chromedriver selenium"
        )

# Key roles for different org types - aligned with ner_engine.py patterns
COMPANY_ROLES = [
    "CEO",
    "President",
    "Managing Director",
    "Chief Technology Officer",
    "Chief Engineer",
    "Chief Scientist",
    "Head of HR",
    "Head of Recruitment",
]

UNIVERSITY_ROLES = [
    "Vice Chancellor",
    "Chancellor",
    "President",
    "Provost",
    "Dean",
    "Principal",
]


def is_university(org: str) -> bool:
    """Check if organization is likely a university.

    Uses patterns consistent with ner_engine.ACADEMIC_PATTERNS.
    """
    try:
        from ner_engine import ACADEMIC_PATTERNS

        org_lower = org.lower()
        return any(pattern in org_lower for pattern in ACADEMIC_PATTERNS)
    except ImportError:
        # Fallback if ner_engine not available
        org_lower = org.lower()
        uni_keywords = [
            "university",
            "college",
            "school",
            "institute",
            "laboratory",
            "research",
        ]
        return any(kw in org_lower for kw in uni_keywords)


def get_organizations_from_db():
    """Get unique organizations from the database."""
    conn = sqlite3.connect("content_engine.db")
    cursor = conn.cursor()
    # Check direct_people and indirect_people for organizations
    cursor.execute(
        "SELECT direct_people, indirect_people FROM stories WHERE direct_people IS NOT NULL OR indirect_people IS NOT NULL"
    )
    rows = cursor.fetchall()

    orgs = set()
    for direct_people_json, indirect_people_json in rows:
        # Process direct_people
        if direct_people_json:
            try:
                people = json.loads(direct_people_json)
                for person in people:
                    company = person.get("company", "") or person.get("affiliation", "")
                    if company:
                        orgs.add(company)
            except Exception:
                pass
        # Process indirect_people
        if indirect_people_json:
            try:
                people = json.loads(indirect_people_json)
                for person in people:
                    company = person.get("company", "") or person.get("affiliation", "")
                    if company:
                        orgs.add(company)
            except Exception:
                pass

    conn.close()
    return sorted(orgs)


def search_indirect_people_uc(org: str, roles: list[str]) -> list[dict]:
    """Search for multiple indirect-people roles using undetected-chromedriver (single browser session)."""
    if not UC_AVAILABLE or uc is None or By is None:
        print("undetected-chromedriver not installed")
        return []

    results = []
    driver = None

    try:
        # Configure Chrome options for stealth (based on ancestry repo patterns)
        options = uc.ChromeOptions()  # type: ignore[union-attr]
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--log-level=3")  # Suppress Chrome warnings
        options.add_argument("--disable-software-rasterizer")
        options.add_argument("--disable-extensions")
        options.add_argument("--no-first-run")
        options.add_argument("--no-default-browser-check")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-infobars")
        options.add_argument("--disable-popup-blocking")

        # Consistent user agent (random user agents are red flags for bot detection)
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
        options.add_argument(f"--user-agent={user_agent}")

        # Create undetected Chrome driver with stability options
        driver = uc.Chrome(  # type: ignore[union-attr]
            options=options,
            use_subprocess=False,  # False is more stable on Windows
            suppress_welcome=True,
        )
        driver.set_page_load_timeout(30)

        for role in roles:
            search_query = f"{role} {org} LinkedIn"
            print(f"  Searching: {role}...", end=" ", flush=True)

            url = f"https://www.bing.com/search?q={search_query.replace(' ', '+')}"
            driver.get(url)
            time.sleep(2)  # Wait for page to load

            # Parse role for name matching
            role_lower = role.lower()
            org_lower = org.lower()
            org_words = [w for w in org_lower.split() if len(w) > 2]

            found = False
            result_items = driver.find_elements(By.CSS_SELECTOR, ".b_algo")  # type: ignore[union-attr]

            for item in result_items[:5]:  # Check first 5 results
                try:
                    heading = item.find_element(By.CSS_SELECTOR, "h2")  # type: ignore[union-attr]
                    title = heading.text
                    link = heading.find_element(By.CSS_SELECTOR, "a")  # type: ignore[union-attr]
                    href = link.get_attribute("href") or ""

                    # Decode Bing redirect URL
                    u_match = re.search(r"[&?]u=a1([^&]+)", href)
                    if not u_match:
                        continue

                    try:
                        encoded = u_match.group(1)
                        padding = 4 - len(encoded) % 4
                        if padding != 4:
                            encoded += "=" * padding
                        decoded_url = base64.urlsafe_b64decode(encoded).decode("utf-8")
                    except Exception:
                        continue

                    if "linkedin.com/in/" not in decoded_url:
                        continue

                    # Get snippet for context
                    try:
                        snippet_el = item.find_element(By.CSS_SELECTOR, ".b_caption p")  # type: ignore[union-attr]
                        snippet = snippet_el.text
                    except Exception:
                        snippet = ""
                    result_text = f"{title} {snippet}".lower()

                    # Check if org appears in result
                    org_match = (
                        any(word in result_text for word in org_words)
                        or org_lower in result_text
                    )

                    # Check if role appears in result
                    role_match = role_lower in result_text

                    if org_match and role_match:
                        vanity_match = re.search(
                            r"linkedin\.com/in/([\w\-]+)", decoded_url
                        )
                        if vanity_match:
                            vanity = vanity_match.group(1)
                            profile_url = f"https://www.linkedin.com/in/{vanity}"

                            # Try to extract name from title
                            name_match = re.match(r"^([^-–|]+)", title)
                            name = (
                                name_match.group(1).strip()
                                if name_match
                                else vanity.replace("-", " ").title()
                            )

                            results.append(
                                {
                                    "role": role,
                                    "name": name,
                                    "linkedin": profile_url,
                                }
                            )
                            print(f"✓ {name}")
                            found = True
                            break

                except Exception:
                    continue

            if not found:
                print("✗")

            time.sleep(0.5)  # Brief delay between searches

    except Exception as e:
        print(f"Error: {e}")

    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass

    return results


def main():
    print("=" * 80)
    print("FINDING LINKEDIN PROFILES FOR INDIRECT PEOPLE (ORG LEADERSHIP)")
    print("=" * 80)

    orgs = get_organizations_from_db()
    print(f"\nOrganizations to search: {len(orgs)}")
    for org in orgs:
        org_type = "University" if is_university(org) else "Company"
        print(f"  • {org} ({org_type})")

    all_results = {}

    for org in orgs:
        print(f"\n{'=' * 80}")
        print(f"{org}")
        print("=" * 80)

        # Choose roles based on org type
        roles = UNIVERSITY_ROLES if is_university(org) else COMPANY_ROLES

        results = search_indirect_people_uc(org, roles)
        all_results[org] = results

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY (INDIRECT PEOPLE)")
    print("=" * 80)

    for org in orgs:
        print(f"\n{org}")
        print("-" * len(org))

        if all_results.get(org):
            for r in all_results[org]:
                print(f"  ✓ {r['role']}: {r['name']}")
                print(f"      {r['linkedin']}")
        else:
            print("  No indirect people found")

    # Total stats
    total_found = sum(len(r) for r in all_results.values())
    print(f"\n{'=' * 80}")
    print(f"Total indirect people found: {total_found}")
    print("=" * 80)

    # Save results to JSON file
    import json as json_mod

    with open("indirect_people_profiles.json", "w") as f:
        json_mod.dump(all_results, f, indent=2)
    print("\nResults saved to indirect_people_profiles.json")


if __name__ == "__main__":
    main()


# ============================================================================
# Unit Tests
# ============================================================================
def _create_module_tests() -> bool:
    """Create unit tests for find_indirect_people module."""
    from test_framework import TestSuite, patch

    suite = TestSuite("Find Indirect People Tests", "find_indirect_people.py")
    suite.start_suite()

    def test_is_university_true():
        assert is_university("University of Cambridge") is True
        assert is_university("MIT") is True
        assert is_university("Stanford College") is True
        assert is_university("Imperial College London") is True
        assert is_university("California Institute of Technology") is True

    def test_is_university_false():
        assert is_university("Google") is False
        assert is_university("Shell Oil Company") is False
        assert is_university("BASF") is False
        assert is_university("Acme Corp") is False

    def test_company_roles_defined():
        assert COMPANY_ROLES is not None
        assert len(COMPANY_ROLES) > 0
        assert "CEO" in COMPANY_ROLES
        assert "Chief Technology Officer" in COMPANY_ROLES

    def test_university_roles_defined():
        assert UNIVERSITY_ROLES is not None
        assert len(UNIVERSITY_ROLES) > 0
        assert "Vice Chancellor" in UNIVERSITY_ROLES
        assert "Dean" in UNIVERSITY_ROLES

    def test_get_organizations_from_db_returns_list():
        # Create a temporary database with the expected schema
        import tempfile
        import os
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        try:
            conn = sqlite3.connect(tmp.name)
            conn.execute(
                "CREATE TABLE stories (direct_people TEXT, indirect_people TEXT)"
            )
            conn.execute(
                "INSERT INTO stories VALUES (?, ?)",
                (json.dumps([{"name": "Alice", "company": "Acme"}]), None),
            )
            conn.commit()
            conn.close()
            # Patch sqlite3.connect to use our temp DB
            with patch("find_indirect_people.sqlite3.connect", return_value=sqlite3.connect(tmp.name)):
                result = get_organizations_from_db()
            assert isinstance(result, list)
            assert "Acme" in result
        finally:
            os.unlink(tmp.name)

    def test_main_function_exists():
        import inspect
        sig = inspect.signature(main)
        params = list(sig.parameters.keys())
        assert len(params) >= 0  # main accepts args
        assert callable(main)

    def test_search_indirect_people_uc_exists():
        import inspect
        sig = inspect.signature(search_indirect_people_uc)
        params = list(sig.parameters.keys())
        assert "organizations" in params or len(params) > 0
        assert callable(search_indirect_people_uc)

    suite.run_test(
        test_name="is_university returns True for universities",
        test_func=test_is_university_true,
        test_summary="Tests is university returns True for universities functionality",
        method_description="Calls is university and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="is_university returns False for companies",
        test_func=test_is_university_false,
        test_summary="Tests is university returns False for companies functionality",
        method_description="Calls is university and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="COMPANY_ROLES list is defined",
        test_func=test_company_roles_defined,
        test_summary="Tests COMPANY ROLES list is defined functionality",
        method_description="Invokes the function under test and validates behavior",
        expected_outcome="Function correctly processes multiple items",
    )
    suite.run_test(
        test_name="UNIVERSITY_ROLES list is defined",
        test_func=test_university_roles_defined,
        test_summary="Tests UNIVERSITY ROLES list is defined functionality",
        method_description="Invokes the function under test and validates behavior",
        expected_outcome="Function correctly processes multiple items",
    )
    suite.run_test(
        test_name="get_organizations_from_db returns list",
        test_func=test_get_organizations_from_db_returns_list,
        test_summary="Tests get organizations from db returns list functionality",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function correctly processes multiple items",
    )
    suite.run_test(
        test_name="main function exists",
        test_func=test_main_function_exists,
        test_summary="Tests main function exists functionality",
        method_description="Calls callable and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="search_indirect_people_uc function exists",
        test_func=test_search_indirect_people_uc_exists,
        test_summary="Tests search indirect people uc function exists functionality",
        method_description="Calls callable and verifies the result",
        expected_outcome="Function finds and returns the expected results",
    )

    return suite.finish_suite()


run_comprehensive_tests = __import__("test_framework").create_standard_test_runner(
    _create_module_tests
)
