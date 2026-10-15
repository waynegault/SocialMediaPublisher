"""Find LinkedIn profiles for indirect people (org leadership) at each organization."""

import base64
import json
import logging
import re
import sqlite3
import time
from contextlib import closing

from browser import UC_AVAILABLE, By, uc
from config import Config
from ner_engine import ACADEMIC_PATTERNS

logger = logging.getLogger(__name__)

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
    """Check if organization is likely an academic or research institution."""
    org_lower = org.lower()
    return any(pattern in org_lower for pattern in ACADEMIC_PATTERNS)


def get_organizations_from_db() -> list[str]:
    """Get unique organizations from the configured application database."""
    query = """
        SELECT direct_people, indirect_people
        FROM stories
        WHERE direct_people IS NOT NULL OR indirect_people IS NOT NULL
    """
    organizations: set[str] = set()

    with closing(sqlite3.connect(Config.DB_NAME)) as conn, conn:
        for direct_people_json, indirect_people_json in conn.execute(query):
            for field_name, people_json in (
                ("direct_people", direct_people_json),
                ("indirect_people", indirect_people_json),
            ):
                if not people_json:
                    continue
                try:
                    people = json.loads(people_json)
                except json.JSONDecodeError as e:
                    logger.warning(
                        "Skipping malformed %s JSON in database: %s", field_name, e
                    )
                    continue
                if not isinstance(people, list):
                    logger.warning("Skipping non-list %s data in database", field_name)
                    continue
                for person in people:
                    if not isinstance(person, dict):
                        logger.warning(
                            "Skipping non-object entry in %s data", field_name
                        )
                        continue
                    organization = person.get("company") or person.get("affiliation")
                    if isinstance(organization, str) and organization.strip():
                        organizations.add(organization.strip())

    return sorted(organizations)


def search_indirect_people_uc(org: str, roles: list[str]) -> list[dict]:
    """Search for multiple indirect-people roles using one browser session."""
    if not UC_AVAILABLE or uc is None or By is None:
        raise RuntimeError(
            "Indirect-people search requires undetected-chromedriver and selenium"
        )
    from selenium.common.exceptions import NoSuchElementException

    results: list[dict] = []
    driver = None

    try:
        options = uc.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--log-level=3")
        options.add_argument("--disable-software-rasterizer")
        options.add_argument("--disable-extensions")
        options.add_argument("--no-first-run")
        options.add_argument("--no-default-browser-check")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-infobars")
        options.add_argument("--disable-popup-blocking")

        user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
        )
        options.add_argument(f"--user-agent={user_agent}")

        driver = uc.Chrome(
            options=options,
            use_subprocess=False,
            suppress_welcome=True,
        )
        driver.set_page_load_timeout(30)

        for role in roles:
            search_query = f"{role} {org} LinkedIn"
            print(f"  Searching: {role}...", end=" ", flush=True)

            url = f"https://www.bing.com/search?q={search_query.replace(' ', '+')}"
            driver.get(url)
            time.sleep(2)

            role_lower = role.lower()
            org_lower = org.lower()
            org_words = [word for word in org_lower.split() if len(word) > 2]
            found = False
            result_items = driver.find_elements(By.CSS_SELECTOR, ".b_algo")

            for item in result_items[:5]:
                try:
                    heading = item.find_element(By.CSS_SELECTOR, "h2")
                    title = heading.text
                    link = heading.find_element(By.CSS_SELECTOR, "a")
                    href = link.get_attribute("href") or ""
                    url_match = re.search(r"[&?]u=a1([^&]+)", href)
                    if not url_match:
                        continue

                    encoded = url_match.group(1)
                    padding = 4 - len(encoded) % 4
                    if padding != 4:
                        encoded += "=" * padding
                    try:
                        decoded_url = base64.urlsafe_b64decode(encoded).decode("utf-8")
                    except (ValueError, UnicodeDecodeError) as e:
                        logger.debug("Skipping malformed search result URL: %s", e)
                        continue

                    if "linkedin.com/in/" not in decoded_url:
                        continue

                    try:
                        snippet_el = item.find_element(
                            By.CSS_SELECTOR, ".b_caption p"
                        )
                        snippet = snippet_el.text
                    except NoSuchElementException as e:
                        logger.debug("Search result did not contain a snippet: %s", e)
                        snippet = ""
                    result_text = f"{title} {snippet}".lower()
                    org_match = any(word in result_text for word in org_words) or (
                        org_lower in result_text
                    )
                    if not org_match or role_lower not in result_text:
                        continue

                    vanity_match = re.search(r"linkedin\.com/in/([\w\-]+)", decoded_url)
                    if not vanity_match:
                        continue
                    vanity = vanity_match.group(1)
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
                            "linkedin": f"https://www.linkedin.com/in/{vanity}",
                        }
                    )
                    print(f"✓ {name}")
                    found = True
                    break
                except NoSuchElementException as e:
                    logger.warning("Skipping malformed search result: %s", e)

            if not found:
                print("✗")
            time.sleep(0.5)

    except Exception:
        logger.exception("Indirect-people search failed for %s", org)
        raise
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                logger.exception("Failed to close browser after indirect-people search")

    return results


def main() -> None:
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
        print(org)
        print("=" * 80)

        roles = UNIVERSITY_ROLES if is_university(org) else COMPANY_ROLES
        all_results[org] = search_indirect_people_uc(org, roles)

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY (INDIRECT PEOPLE)")
    print("=" * 80)
    for org in orgs:
        print(f"\n{org}")
        print("-" * len(org))
        if all_results.get(org):
            for result in all_results[org]:
                print(f"  ✓ {result['role']}: {result['name']}")
                print(f"      {result['linkedin']}")
        else:
            print("  No indirect people found")

    total_found = sum(len(results) for results in all_results.values())
    print(f"\n{'=' * 80}")
    print(f"Total indirect people found: {total_found}")
    print("=" * 80)

    output_path = "indirect_people_profiles.json"
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(all_results, output_file, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()


def _create_module_tests() -> bool:
    """Test indirect-people data access and organization routing."""
    import os
    import tempfile

    from test_framework import MagicMock, TestSuite, patch

    suite = TestSuite("Find Indirect People Tests", "find_indirect_people.py")
    suite.start_suite()

    def test_university_patterns_are_shared():
        assert is_university("University of Cambridge")
        assert is_university("MIT")
        assert is_university("Stanford College")
        assert is_university("Imperial College London")
        assert is_university("California Institute of Technology")
        assert not is_university("Google")
        assert not is_university("Shell Oil Company")
        assert not is_university("BASF")
        assert not is_university("Acme Corp")

    def test_roles_are_defined():
        assert "CEO" in COMPANY_ROLES
        assert "Chief Technology Officer" in COMPANY_ROLES
        assert "Vice Chancellor" in UNIVERSITY_ROLES
        assert "Dean" in UNIVERSITY_ROLES

    def test_configured_database_is_read_and_closed():
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "stories.db")
            with closing(sqlite3.connect(db_path)) as connection, connection:
                connection.execute(
                    "CREATE TABLE stories (direct_people TEXT, indirect_people TEXT)"
                )
                connection.execute(
                    "INSERT INTO stories VALUES (?, ?)",
                    (json.dumps([{"name": "Alice", "company": " Acme "}]), None),
                )
                connection.execute(
                    "INSERT INTO stories VALUES (?, ?)",
                    ("not-json", json.dumps([{"name": "Bob", "affiliation": "Beta"}])),
                )
            original_db_name = Config.DB_NAME
            Config.DB_NAME = db_path
            try:
                assert get_organizations_from_db() == ["Acme", "Beta"]
            finally:
                Config.DB_NAME = original_db_name
            os.unlink(db_path)

    def test_unavailable_browser_fails_explicitly():
        with patch("find_indirect_people.UC_AVAILABLE", False):
            try:
                search_indirect_people_uc("Acme Corp", ["CEO"])
            except RuntimeError as error:
                assert "undetected-chromedriver" in str(error)
            else:
                raise AssertionError("Unavailable browser dependency was hidden")

    def test_search_failure_is_visible():
        mocked_uc = MagicMock()
        mocked_uc.Chrome.side_effect = OSError("browser startup failed")
        with (
            patch("find_indirect_people.UC_AVAILABLE", True),
            patch("find_indirect_people.uc", mocked_uc),
            patch("find_indirect_people.By", object()),
        ):
            try:
                search_indirect_people_uc("Acme Corp", ["CEO"])
            except OSError as error:
                assert "browser startup failed" in str(error)
            else:
                raise AssertionError("Browser startup failure was hidden")

    def test_main_selects_correct_roles():
        captured_calls: list[tuple[str, list[str]]] = []

        def mock_search(org: str, roles: list[str]) -> list[dict]:
            captured_calls.append((org, roles))
            return []

        with (
            patch(
                "find_indirect_people.get_organizations_from_db",
                return_value=["MIT", "Shell Oil Company"],
            ),
            patch(
                "find_indirect_people.search_indirect_people_uc",
                side_effect=mock_search,
            ),
            patch("builtins.open"),
        ):
            main()
        assert captured_calls == [
            ("MIT", UNIVERSITY_ROLES),
            ("Shell Oil Company", COMPANY_ROLES),
        ]

    tests = [
        (
            "Shared university classification",
            test_university_patterns_are_shared,
            "Academic patterns classify academic institutions without classifying companies",
        ),
        (
            "Role constants",
            test_roles_are_defined,
            "Company and university leadership roles are available",
        ),
        (
            "Configured database access",
            test_configured_database_is_read_and_closed,
            "Configured database is read, malformed data is skipped, and its handle closes",
        ),
        (
            "Unavailable browser error",
            test_unavailable_browser_fails_explicitly,
            "Missing browser dependencies produce an explicit error",
        ),
        (
            "Search failure visibility",
            test_search_failure_is_visible,
            "Browser startup failures propagate to the caller",
        ),
        (
            "Main role dispatch",
            test_main_selects_correct_roles,
            "Main dispatches each organization to the matching role set",
        ),
    ]
    for name, test, expected_outcome in tests:
        suite.run_test(
            test_name=name,
            test_func=test,
            test_summary=f"Verify {name.lower()}",
            expected_outcome=expected_outcome,
        )

    return suite.finish_suite()


run_comprehensive_tests = __import__("test_framework").create_standard_test_runner(
    _create_module_tests
)
