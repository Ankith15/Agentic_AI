import asyncio
from playwright.async_api import async_playwright
import json

async def extract_qa_playwright(page):
    faq_data = {}
    question_answer_blocks = await page.query_selector_all('div.py-4.first\\:pt-0.last\\:pb-0.undefined')

    for block in question_answer_blocks:
        question_element = await block.query_selector('p.font-bold.dark\\:text-white.w-11\\/12')
        if question_element:
            question = (await question_element.text_content()).strip()
            await question_element.click()
            await asyncio.sleep(0.5)  # Give time for the answer to appear

            actual_answer_element = await block.query_selector('div.rounded-b.dark\\:text-gray-300.mt-3.block span[data-slate-string="true"]')
            if actual_answer_element:
                answer = (await actual_answer_element.text_content()).strip()
                faq_data[question] = answer
                
            else:
                faq_data[question] = "Answer not found after click"
        else:
            print("Warning: Could not find question block.")
    return faq_data

async def scrape_page():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        await page.goto("https://www.myscheme.gov.in/search")
        await page.wait_for_selector("li.h-8.w-8")

        Data = {}
        scheme_id = 1
        for page_num in range(1,339):
            print(f"\n Scraping {page_num}th page")

            if page_num > 1:
                try:
                    page_number = page.locator("li.h-8.w-8", has_text=str(page_num)).first
                    await page_number.click()
                    await page.wait_for_load_state("networkidle")
                    await page.wait_for_timeout(1000)
                except Exception as e:
                    print(f"Failed to click page {page_num}: {e}")
                    continue

            await page.wait_for_selector("div.p-4.lg\\:p-8.w-full")
            card_locator = page.locator("div.p-4.lg\\:p-8.w-full")
            count = await card_locator.count()

            for i in range(count):
                card = card_locator.nth(i).locator("a")
                tit = await card.text_content()
                # print(f"Scraping details for: {tit}")

                detail_page = await context.new_page()
                try:
                    url = await card.get_attribute("href")
                    if url:
                        await detail_page.goto("https://www.myscheme.gov.in" + url)
                        await detail_page.wait_for_load_state("networkidle")

                        titl_elements = await detail_page.query_selector_all(
                            "div.grid.grid-cols-4.gap-2 > div.col-span-4.shadow-md.rounded-md.bg-white.dark\\:bg-dark.py-4 > div > div.overflow-x-auto.overflow-y-hidden.flex.flex-row.items-center.mb-2.border-0.border-b.border-solid.border-gray-200.pr-2.md\\:pr-4.no-scrollbar span"
                        )
                        info_elements = await detail_page.query_selector_all("div.markdown-options")

                        det_title = [(await el.text_content()).strip() for el in titl_elements]
                        det_info = [(await el.text_content()).strip() for el in info_elements]

                        details_dict = {}
                        details_dict['scheme_id'] = scheme_id
                        for j in range(min(len(det_title), len(det_info))):
                            details_dict[det_title[j]] = det_info[j]
                        

                        faq_data = await extract_qa_playwright(detail_page)
                        details_dict["Frequently Asked Questions"] = faq_data

                        Data[tit.strip()] = details_dict
                        scheme_id += 1

                    else:
                        print(f"No URL found for {tit}")
                except Exception as e:
                    print(f"Error scraping scheme {tit}: {e}")
                finally:
                    await detail_page.close()

        print("\n Scraping finished. Schemes found:")
        with open("slmdata.json", "w", encoding="utf-8") as f:
            json.dump(Data, f, indent=4)
        await browser.close()

if __name__ == "__main__":
    asyncio.run(scrape_page())