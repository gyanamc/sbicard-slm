import scrapy

class SbicardItem(scrapy.Item):
    card_name = scrapy.Field()
    category = scrapy.Field()
    features = scrapy.Field()
    fees_and_charges = scrapy.Field()
    rewards = scrapy.Field()
    welcome_benefits = scrapy.Field() # Add this field as well for completeness
    eligibility = scrapy.Field()
    url = scrapy.Field()
    timestamp = scrapy.Field()
    raw_html = scrapy.Field() # Optional: store raw HTML for debugging/later extraction
