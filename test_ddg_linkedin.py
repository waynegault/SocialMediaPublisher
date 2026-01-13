"""Test DuckDuckGo for LinkedIn profile discovery."""

from ddgs import DDGS

# Test DuckDuckGo search for LinkedIn profiles - try different strategies
queries = [
    "Bill Gates LinkedIn profile",
    '"Michael Jewett" Stanford LinkedIn',
    "Sergio Ermotti UBS CEO LinkedIn",
]

for query in queries:
    print(f"Searching: {query}")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            if results:
                for r in results:
                    url = r.get("href", "")
                    if "linkedin" in url.lower():
                        print(f"  ✓ LINKEDIN: {url}")
                    else:
                        print(
                            f"  - Other: {r.get('title', 'N/A')[:40]}... ({url[:40]}...)"
                        )
            else:
                print("  No results")
    except Exception as e:
        print(f"  Error: {e}")
    print()
