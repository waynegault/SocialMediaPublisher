"""Test script: Find Wayne Gault at Aberdeenshire ADP and send connection request."""

from linkedin_profile_lookup import LinkedInCompanyLookup
from config import Config
import time

lookup = LinkedInCompanyLookup()

# Connection message using discipline from config
connection_message = f"Hi, I'm trying to expand my professional {Config.DISCIPLINE} network. I'd be grateful if you'd accept my connect request."

# Step 1: Search for Wayne Gault at Aberdeenshire ADP
print("Searching for Wayne Gault at Aberdeenshire ADP...")
profile_url = lookup.search_person("Wayne Gault", "Aberdeenshire ADP")

if profile_url:
    print(f"Found profile: {profile_url}")

    # Step 2: Send connection request with message
    print(f"Sending connection request with message: {connection_message}")
    success, message = lookup.send_connection_via_browser(
        profile_url, message=connection_message
    )
    print(f"Result: success={success}, message={message}")
else:
    print("Profile not found via search. Trying browser-based LinkedIn search...")

    # Fallback: Search directly in LinkedIn
    driver = lookup._get_uc_driver()
    if driver and lookup._ensure_linkedin_login(driver):
        search_url = "https://www.linkedin.com/search/results/people/?keywords=Wayne%20Gault%20Aberdeenshire%20ADP"
        driver.get(search_url)
        time.sleep(5)

        from selenium.webdriver.common.by import By

        # Find first profile link
        links = driver.find_elements(By.XPATH, '//a[contains(@href, "/in/")]')
        profile_urls = []
        for link in links:
            href = link.get_attribute("href")
            if href and "/in/" in href and "miniProfile" not in href:
                # Clean URL
                if "?" in href:
                    href = href.split("?")[0]
                if href not in profile_urls:
                    profile_urls.append(href)

        if profile_urls:
            print(f"Found profiles: {profile_urls[:3]}")
            target_url = profile_urls[0]
            print(f"Connecting to: {target_url}")
            print(f"Message: {connection_message}")

            success, message = lookup.send_connection_via_browser(
                target_url, message=connection_message
            )
            print(f"Result: success={success}, message={message}")
        else:
            print("No profiles found in search results")
    else:
        print("Failed to get browser or login")

lookup.close_browser()
print("Done!")
