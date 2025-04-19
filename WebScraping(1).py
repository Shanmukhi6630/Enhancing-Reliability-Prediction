from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import csv
import re

drive = webdriver.Chrome()
drive.implicitly_wait(5)

drive.get("https://www.amazon.in/")
time.sleep(2)

search = WebDriverWait(drive, 10).until(
    EC.presence_of_element_located((By.XPATH, "//input[@placeholder='Search Amazon.in']"))
)
search.send_keys("airpods")
search.send_keys(Keys.RETURN)
time.sleep(2)

product = WebDriverWait(drive, 10).until(
    EC.element_to_be_clickable((By.XPATH, "//img[@class='s-image']"))
)
product.click()
time.sleep(3)

original_window = drive.current_window_handle
WebDriverWait(drive, 5).until(EC.number_of_windows_to_be(2))
for window_handle in drive.window_handles:
    if window_handle != original_window:
        drive.switch_to.window(window_handle)
        break

try:
    review_button = WebDriverWait(drive, 5).until(
        EC.element_to_be_clickable((By.XPATH, "//a[@id='acrCustomerReviewLink']"))
    )
    review_button.click()
    time.sleep(2)
    print("Navigated to all reviews.")
except:
    print("❌ Review button not found. Proceeding with available reviews...")

input("Please log in manually, then press Enter to continue...")
print("Continuing with scraping...")

# Open CSV file for writing
with open("amazon_reviews.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Review Text", "Review Date", "Helpful Votes", "Verified Purchase"])

    collected_reviews = 0

    while collected_reviews < 1000:
        print(f"Scraping... Total collected: {collected_reviews}")

        # Scroll to load more reviews
        drive.execute_script("window.scrollBy(0, 500);")
        time.sleep(2)

        see_more_buttons = drive.find_elements(By.XPATH, "//span[contains(text(), 'See more')]")
        for btn in see_more_buttons:
            drive.execute_script("arguments[0].click();", btn)
            time.sleep(1)

        reviews = drive.find_elements(By.XPATH, "//span[@data-hook='review-body']")
        dates = drive.find_elements(By.XPATH, "//span[@data-hook='review-date']")
        helpful_votes = drive.find_elements(By.XPATH, "//span[@data-hook='helpful-vote-statement']")
        verified_purchases = drive.find_elements(By.XPATH, "//span[@data-hook='avp-badge']")

        review_texts = [r.text.strip() for r in reviews]
        review_dates = [d.text.strip() for d in dates]
        review_helpful_votes = [re.search(r"(\d+)", h.text).group(1) if re.search(r"(\d+)", h.text) else "0" for h in helpful_votes]
        verified_purchases_list = ["Yes" if "Verified Purchase" in v.text else "No" for v in verified_purchases]

        max_length = max(len(review_texts), len(review_dates), len(review_helpful_votes), len(verified_purchases_list))
        while len(review_texts) < max_length:
            review_texts.append("N/A")
        while len(review_dates) < max_length:
            review_dates.append("N/A")
        while len(review_helpful_votes) < max_length:
            review_helpful_votes.append("0")
        while len(verified_purchases_list) < max_length:
            verified_purchases_list.append("No")

        for text, date, votes, verified in zip(review_texts, review_dates, review_helpful_votes, verified_purchases_list):
            writer.writerow([text, date, votes, verified])
            collected_reviews += 1
            if collected_reviews >= 1000:
                break
        try:
            next_button = WebDriverWait(drive, 3).until(
                EC.element_to_be_clickable((By.XPATH, "//li[@class='a-last']/a"))
            )
            drive.execute_script("arguments[0].scrollIntoView();", next_button)
            next_button.click()
            time.sleep(2)
        except:
            print("✅ No more review pages found.")
            break 

print("✅ Scraping completed. Data saved in 'amazon_reviews.csv'")

drive.quit()
