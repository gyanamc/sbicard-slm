import scrapy
from datetime import datetime
from ..items import SbicardItem

class SbicardSpider(scrapy.Spider):
    name = "sbicard"
    allowed_domains = ["sbicard.com"]
    start_urls = ["https://www.sbicard.com/en/personal/credit-cards.page"]

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(
                url,
                meta={
                    "playwright": True,
                    "playwright_include_page": True,
                },
                callback=self.parse
            )

    async def parse(self, response):
        page = response.meta["playwright_page"]
        
        # Extract links using Playwright to ensure dynamic content is loaded
        # We look for links containing '/personal/credit-cards/' and ending in '.page'
        hrefs = await page.evaluate("""() => {
            return Array.from(document.querySelectorAll('a')).map(a => a.href)
        }""")
        
        card_links = set()
        for link in hrefs:
            if "/personal/credit-cards/" in link and link.endswith(".page") and link != response.url:
                # Exclude category pages which usually have /personal/credit-cards/category.page structure
                # Valid card pages often have deeper structure or specific names
                # We will filter duplicates and self-references
                card_links.add(link)

        self.logger.info(f"Found {len(card_links)} potential card links.")

        for link in card_links:
            yield scrapy.Request(
                link,
                meta={
                    "playwright": True,
                    "playwright_include_page": True,
                },
                callback=self.parse_card_details
            )
            
        await page.close()

    async def parse_card_details(self, response):
        page = response.meta["playwright_page"]
        try:
            item = SbicardItem()
            item["url"] = response.url
            item["timestamp"] = datetime.now().isoformat()
            
            # Extract content using standard Scrapy selectors on the rendered response
            # (scrapy-playwright populates response.body with the final page content)
            
            # Card Name
            item["card_name"] = response.css('h1::text').get("").strip()
            
            # Features - Strategy: Grab all text from list items within feature-like sections
            # We look for ULs that are likely part of the content
            # A more structured approach: group by headers (h3)
            features = []
            
            # Iterate over headers and associated lists (heuristic)
            # This is a bit complex with just CSS, so we'll grab all list items in the main container
            # The main container usually has a class like 'right-section' or 'container'
            # We'll grab all text from li elements that are likely content
            all_list_items = response.css('div.content-area li::text, div.right-container li::text, .features li::text').getall()
            if not all_list_items:
                 all_list_items = response.css('li::text').getall() # Fallback
            
            item["features"] = [text.strip() for text in all_list_items if len(text.strip()) > 10]
            
            # Fees - specific extraction
            # Look for text containing "Annual Fee" or "Renewal Fee" in the entire body or specific tables
            body_text = " ".join(response.css('body *::text').getall())
            
            fees_info = {}
            if "Annual Fee" in body_text:
                # Try to extract the snippet - this is a simple string search
                # In a real html, we'd look for the table row or specific p tag
                # For now, we'll store the raw text segments that contain fee info
                fee_candidates = [
                    t.strip() for t in response.css('div, p, td, li').css('::text').getall() 
                    if "Annual Fee" in t or "Renewal Fee" in t
                ]
                fees_info['raw_fee_text'] = list(set(fee_candidates))
            
            item["fees_and_charges"] = fees_info
            
            # Rewards - specific extraction if identifiable (often same as features but under 'Rewards' header)
            # We'll leave it as part of features for now or try to separate if 'Rewards' header exists
            
            # Welcome benefits
            # Similar to features
            
            # Eligibility
            # Look for 'Eligibility' header
            
            yield item

        except Exception as e:
            self.logger.error(f"Error parsing {response.url}: {e}")
        finally:
             await page.close()
