from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import time

driver = webdriver.Firefox()

# open login page
driver.get("https://ots2026.onlinetestseriesmadeeasy.in")

print("Login manually then press ENTER")
input()

count = 1

while True:
    time.sleep(3)

    # save screenshot
    driver.save_screenshot(f"question_{count}.png")
    print("saved question", count)

    count += 1

    try:
        next_button = driver.find_element(By.XPATH, "//button[contains(text(),'NEXT')]")
        next_button.click()
    except:
        print("No next button found. Probably finished.")
        break

driver.quit()