import json
import html
import re
import time
import sys
from datetime import datetime, timezone, timedelta
from calendar import timegm
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import feedparser

# Target outputs
DATA_FILE = "data.json"
MAX_WORLD_STORIES = 40
MAX_WEB_STORIES = 15

# Stopwords for deduplication n-grams
STOPWORDS = {
    'a', 'an', 'the', 'and', 'but', 'or', 'as', 'if', 'of', 'at', 'by', 'for', 
    'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 
    'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 
    'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 
    'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 
    'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 
    'now', 'is', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
    'having', 'do', 'does', 'did', 'doing', 'are', 'due'
}

# Request headers to look like a friendly browser/tool
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5"
}

def clean_title_for_ngrams(title):
    """Normalize and tokenize title, returning a list of lowercase words with punctuation removed."""
    # Strip possessive 's (e.g. "iran's" -> "iran") to normalize noun forms
    cleaned = re.sub(r"'s\b", "", title.lower())
    # Strip punctuation
    cleaned = re.sub(r'[^\w\s]', '', cleaned)
    return [word for word in cleaned.split() if word not in STOPWORDS]

def get_trigrams(words):
    """Generate a set of 3-word tuples from a list of words."""
    return set(tuple(words[i:i+3]) for i in range(len(words) - 2))

def deduplicate_stories(stories):
    """
    Deduplicates a list of stories sorted by newest first.
    Groups duplicates and increments the canonical story's score.
    Supports transitive n-gram registration for subsequent duplicates.
    Requires at least 2 matching trigrams (or 1 if the story has only 1 trigram).
    """
    seen_trigrams = {} # Maps trigram tuple -> story dict
    seen_exact = {}     # Maps short exact title key -> story dict
    deduped = []
    
    for story in stories:
        story["score"] = 1 # Initialize consensus score
        story["other_sources"] = [] # Initialize alternative sources list
        words = clean_title_for_ngrams(story["title"])
        
        # If the title is very short, fallback to exact title comparison
        if len(words) < 3:
            exact_key = " ".join(words)
            if exact_key in seen_exact:
                canonical = seen_exact[exact_key]
                source = story.get("source")
                canonical_source = canonical.get("source")
                already_counted = any(x.get("source") == source for x in canonical["other_sources"])
                if source and source != canonical_source and not already_counted:
                    canonical["score"] += 1
                # Add to other_sources
                if source:
                    canonical["other_sources"].append({
                        "source": source,
                        "url": story.get("url", ""),
                        "title": story.get("title", ""),
                        "published": story.get("published", "")
                    })
                continue
            seen_exact[exact_key] = story
            deduped.append(story)
            continue
            
        trigrams = get_trigrams(words)
        
        # Count matching trigrams for each previously seen canonical story
        match_counts = {} # Maps story_id -> [canonical_story, count]
        for tg in trigrams:
            if tg in seen_trigrams:
                canonical = seen_trigrams[tg]
                story_id = id(canonical)
                if story_id not in match_counts:
                    match_counts[story_id] = [canonical, 0]
                match_counts[story_id][1] += 1
                
        # Find the candidate with the highest number of matching trigrams
        duplicate_story = None
        max_matches = 0
        for story_id, (canonical, count) in match_counts.items():
            if count > max_matches:
                duplicate_story = canonical
                max_matches = count
                
        # Require at least 2 matching trigrams, or 1 if the headline is extremely short
        threshold = min(2, len(trigrams))
        
        # Fallback to Jaccard word similarity for headlines with high word overlap but minor reorderings
        if not (duplicate_story and max_matches >= threshold) and len(words) >= 3:
            for seen_story in deduped:
                seen_words = clean_title_for_ngrams(seen_story["title"])
                if len(seen_words) >= 3:
                    set1 = set(words)
                    set2 = set(seen_words)
                    jaccard = len(set1 & set2) / len(set1 | set2)
                    if jaccard >= 0.7:
                        duplicate_story = seen_story
                        max_matches = threshold
                        break
                        
        if duplicate_story and max_matches >= threshold:
            source = story.get("source")
            canonical_source = duplicate_story.get("source")
            already_counted = any(x.get("source") == source for x in duplicate_story["other_sources"])
            if source and source != canonical_source and not already_counted:
                duplicate_story["score"] += 1
            # Add to other_sources
            if source:
                duplicate_story["other_sources"].append({
                    "source": source,
                    "url": story.get("url", ""),
                    "title": story.get("title", ""),
                    "published": story.get("published", "")
                })
            # Register this duplicate's trigrams pointing back to the canonical story
            # to handle transitive duplicates
            for tg in trigrams:
                seen_trigrams[tg] = duplicate_story
            continue
            
        # Update seen list and keep unique story
        for tg in trigrams:
            seen_trigrams[tg] = story
        deduped.append(story)
        
    return deduped

def parse_iso_date(date_str):
    """Parse various ISO 8601 date string formats into a UTC datetime object."""
    if not date_str:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    try:
        # Standard replacement for UTC indicator
        iso_str = date_str.replace("Z", "+00:00")
        # In Python 3.11+, fromisoformat handles decimal seconds and timezone offsets cleanly
        return datetime.fromisoformat(iso_str)
    except Exception:
        # Fallback to epoch time if parsing fails to avoid promoting stories
        return datetime(1970, 1, 1, tzinfo=timezone.utc)

def fetch_rss_feed(source_name, url, retries=2):
    """Fetch and parse a standard RSS feed using feedparser with retry logic."""
    print(f"Fetching RSS feed from {source_name}...")
    stories = []
    for attempt in range(retries + 1):
        try:
            # We fetch manually using requests first to apply headers and timeouts consistently
            response = requests.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            
            feed = feedparser.parse(response.content)
            for entry in feed.entries:
                title = entry.get("title", "").strip()
                link = entry.get("link", "").strip()
                
                if not title or not link:
                    continue
                    
                # Skip generic live-blog placeholders and generic video/audio bulletins
                cleaned_title_lower = title.lower().strip(" .’'\"")
                if (cleaned_title_lower in ["here's the latest", "here’s the latest", "latest updates", "live updates", "here’s what you need to know"]
                        or "news bulletin" in cleaned_title_lower or "latest news" in cleaned_title_lower):
                    continue
                    
                # Parse publication date
                pub_date = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    try:
                        # calendar.timegm converts parsed date tuple to UTC timestamp
                        pub_date = datetime.fromtimestamp(timegm(entry.published_parsed), timezone.utc)
                    except Exception:
                        pass
                
                if not pub_date:
                    # Fallback parsed date strings
                    date_str = entry.get("published") or entry.get("updated")
                    pub_date = parse_iso_date(date_str)
                    
                stories.append({
                    "title": title,
                    "url": link,
                    "source": source_name,
                    "published": pub_date.isoformat()
                })
            print(f"  Successfully fetched {len(stories)} articles from {source_name}.")
            return stories
        except Exception as e:
            if attempt < retries:
                print(f"  Warning: Attempt {attempt + 1} failed for {source_name}. Retrying...", file=sys.stderr)
                time.sleep(1 * (attempt + 1))
                continue
            print(f"  Error fetching {source_name} after {retries} retries: {e}", file=sys.stderr)
    return stories

def fetch_hn_item(item_id):
    """Helper to fetch a single HN item details."""
    url = f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def fetch_hacker_news():
    """Fetch top 15 HN stories in parallel."""
    print("Fetching Hacker News top stories...")
    stories = []
    try:
        # Get top story IDs
        top_ids_url = "https://hacker-news.firebaseio.com/v0/topstories.json"
        r = requests.get(top_ids_url, timeout=10)
        r.raise_for_status()
        top_ids = r.json()[:MAX_WEB_STORIES]
        
        # Fetch items in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fetch_hn_item, item_id): item_id for item_id in top_ids}
            for future in as_completed(futures):
                item = future.result()
                if item and item.get("type") == "story":
                    title = item.get("title", "").strip()
                    url = item.get("url") or f"https://news.ycombinator.com/item?id={item['id']}"
                    score = item.get("score", 0)
                    time_stamp = item.get("time", time.time())
                    pub_date = datetime.fromtimestamp(time_stamp, timezone.utc)
                    
                    stories.append({
                        "title": title,
                        "url": url,
                        "score": score,
                        "source": "Hacker News",
                        "published": pub_date.isoformat()
                    })
        # Sort Hacker News stories to match the original top list order (if needed) or by score
        stories.sort(key=lambda x: x["score"], reverse=True)
        print(f"  Successfully fetched {len(stories)} HN stories.")
    except Exception as e:
        print(f"  Error fetching Hacker News: {e}", file=sys.stderr)
    return stories

def fetch_reddit_subreddit(subreddit):
    """Fetch hot posts from a specific subreddit using its RSS feed."""
    print(f"Fetching Reddit /r/{subreddit} (RSS)...")
    stories = []
    url = f"https://www.reddit.com/r/{subreddit}/hot/.rss"
    try:
        # Reddit guidelines recommend a unique User-Agent to avoid generic blocks
        reddit_headers = {
            "User-Agent": "python:world-state-news-aggregator:v1.0.0 (by /u/nkurien)"
        }
        response = requests.get(url, headers=reddit_headers, timeout=10)
        response.raise_for_status()
        
        feed = feedparser.parse(response.content)
        for entry in feed.entries:
            title = entry.get("title", "").strip()
            comments_link = entry.get("link", "").strip()
            
            if not title or not comments_link:
                continue
                
            # Try to extract the direct article link from the HTML summary
            # If it's a link post, the RSS summary has a link matching: <a href="direct_url">[link]</a>
            direct_link = comments_link
            summary = entry.get("summary", "")
            link_match = re.search(r'href="([^"]+)">\[link\]</a>', summary)
            if link_match:
                direct_link = link_match.group(1).strip()
                
            # Clean up URLs (relative Reddit links should be full links)
            if direct_link.startswith("/r/"):
                direct_link = f"https://www.reddit.com{direct_link}"
                
            # Parse publication date
            pub_date = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    pub_date = datetime.fromtimestamp(timegm(entry.published_parsed), timezone.utc)
                except Exception:
                    pass
            if not pub_date:
                pub_date = parse_iso_date(entry.get("published") or entry.get("updated"))
                
            stories.append({
                "title": title,
                "url": direct_link,
                "comments_url": comments_link,
                "score": None,
                "source": f"/r/{subreddit}",
                "published": pub_date.isoformat()
            })
        print(f"  Successfully fetched {len(stories)} posts from /r/{subreddit}.")
    except Exception as e:
        print(f"  Error fetching /r/{subreddit} RSS: {e}", file=sys.stderr)
    return stories

def fetch_all_reddit():
    """Fetch and merge Reddit stories from multiple subreddits."""
    reddit_stories = []
    subreddits = ["worldnews", "europe", "geopolitics", "technology", "science"]
    for sub in subreddits:
        reddit_stories.extend(fetch_reddit_subreddit(sub))
        time.sleep(1) # Sleep to prevent rapid-fire rate limiting
    
    # Sort merged list by publication date descending, limit to top 15
    reddit_stories.sort(key=lambda x: x["published"], reverse=True)
    return reddit_stories[:MAX_WEB_STORIES]

def fetch_lobsters():
    """Fetch hot stories from Lobste.rs JSON API."""
    print("Fetching Lobste.rs hot stories...")
    stories = []
    url = "https://lobste.rs/hottest.json"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        
        items = response.json()
        for item in items:
            title = item.get("title", "").strip()
            link = item.get("url", "").strip()
            comments_link = item.get("comments_url", "").strip()
            score = item.get("score")
            
            if not title or not link:
                continue
                
            # Parse publication date and convert to UTC
            pub_date = parse_iso_date(item.get("created_at")).astimezone(timezone.utc)
                
            stories.append({
                "title": title,
                "url": link,
                "comments_url": comments_link,
                "score": score,
                "source": "Lobste.rs",
                "published": pub_date.isoformat()
            })
        print(f"  Successfully fetched {len(stories)} stories from Lobste.rs.")
    except Exception as e:
        print(f"  Error fetching Lobste.rs JSON: {e}", file=sys.stderr)
    return stories

def clean_html(text):
    """Strip HTML tags, normalize whitespace, and unescape entities."""
    if not text:
        return ""
    # Strip HTML tags
    clean = re.sub(r'<[^>]+>', '', text)
    # Unescape HTML entities (e.g. &#8217; or &amp;)
    clean = html.unescape(clean)
    # Normalize whitespaces/newlines to single spaces
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean

def truncate_snippet(text, max_len=160):
    """Truncate text at a word boundary close to max_len and append ellipsis."""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    # Try to truncate at a word boundary
    truncated = text[:max_len]
    last_space = truncated.rfind(' ')
    if last_space > max_len - 20:
        return truncated[:last_space] + "..."
    return truncated + "..."


RSS_SOURCES = [
    ("France24", "https://www.france24.com/en/rss"),
    ("Politico EU", "https://www.politico.eu/feed/"),
    ("Semafor", "https://www.semafor.com/rss.xml"),
    ("BBC News", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("The Independent", "https://www.independent.co.uk/news/world/rss"),
    ("NPR World", "https://feeds.npr.org/1004/rss.xml"),
    ("CBC World", "https://rss.cbc.ca/lineup/world.xml"),
    ("The Guardian", "https://www.theguardian.com/world/rss"),
    ("Deutsche Welle", "https://rss.dw.com/xml/rss-en-world"),
    ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
    ("Euronews", "https://www.euronews.com/rss?level=theme&name=news"),
    ("RFI English", "https://www.rfi.fr/en/rss"),
    ("CNA", "https://www.channelnewsasia.com/api/v1/rss-outbound-feed?_format=xml&category=6311"),
    ("Japan Times", "https://www.japantimes.co.jp/feed/"),
    ("SCMP", "https://www.scmp.com/rss/91/feed"),
    ("Straits Times", "https://www.straitstimes.com/news/asia/rss.xml"),
    ("Spiegel", "https://www.spiegel.de/international/index.rss"),
    ("El País", "https://feeds.elpais.com/mrss-s/pages/ep/site/english.elpais.com/portada"),
    ("RNZ", "https://www.rnz.co.nz/rss/news.xml")
]

def fetch_all_rss():
    """Fetch all configured news RSS feeds in parallel."""
    print(f"Fetching {len(RSS_SOURCES)} RSS feeds in parallel...")
    stories = []
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_source = {executor.submit(fetch_rss_feed, name, url): name for name, url in RSS_SOURCES}
        for future in as_completed(future_to_source):
            name = future_to_source[future]
            try:
                res = future.result()
                if res:
                    stories.extend(res)
            except Exception as e:
                print(f"  Error resolving future for RSS source {name}: {e}", file=sys.stderr)
                
    return stories

DIGEST_CONFIGS = [
    {
        "id": "politico_playbook",
        "name": "Politico Playbook",
        "url": "https://rss.politico.com/playbook.xml",
        "column": "briefings",
        "headers": HEADERS
    },
    {
        "id": "the_week",
        "name": "The Week",
        "url": "https://www.theweek.com/rss",
        "column": "briefings",
        "headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        }
    },
    {
        "id": "economist_world",
        "name": "The Economist",
        "url": "https://www.economist.com/leaders/rss.xml",
        "column": "global",
        "headers": HEADERS
    },
    {
        "id": "ft_world",
        "name": "Financial Times",
        "url": "https://www.ft.com/world?format=rss",
        "column": "global",
        "headers": HEADERS
    },
    {
        "id": "tldr_tech",
        "name": "TLDR Tech",
        "url": "https://tldr.tech/rss",
        "column": "tech",
        "headers": HEADERS
    },
    {
        "id": "stratechery",
        "name": "Stratechery",
        "url": "https://stratechery.com/feed/",
        "column": "tech",
        "headers": HEADERS
    }
]

def fetch_single_digest(config):
    """Fetch and parse a single digest feed, returning cleaned entries."""
    name = config["name"]
    url = config["url"]
    headers = config["headers"]
    
    print(f"Fetching digest feed from {name}...")
    entries = []
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        feed = feedparser.parse(response.content)
        # Parse top 3 entries
        for entry in feed.entries[:3]:
            title = entry.get("title", "").strip()
            link = entry.get("link", "").strip()
            
            if not title or not link:
                continue
                
            # Get snippet
            summary = entry.get("summary") or entry.get("description") or ""
            if "content" in entry and entry.content:
                summary = entry.content[0].value
                
            clean_summary = clean_html(summary)
            snippet = truncate_snippet(clean_summary)
            
            # Parse pub date
            pub_date = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    pub_date = datetime.fromtimestamp(timegm(entry.published_parsed), timezone.utc)
                except Exception:
                    pass
            if not pub_date:
                date_str = entry.get("published") or entry.get("updated")
                pub_date = parse_iso_date(date_str)
                
            entries.append({
                "title": title,
                "url": link,
                "published": pub_date.isoformat(),
                "snippet": snippet
            })
        print(f"  Successfully fetched {len(entries)} entries from {name}.")
    except Exception as e:
        print(f"  Error fetching digest {name}: {e}", file=sys.stderr)
        
    return {
        "id": config["id"],
        "name": name,
        "column": config["column"],
        "entries": entries
    }

def fetch_all_digests():
    """Fetch all configured news digests in parallel."""
    print(f"Fetching {len(DIGEST_CONFIGS)} news digests in parallel...")
    results = []
    
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_config = {executor.submit(fetch_single_digest, config): config for config in DIGEST_CONFIGS}
        for future in as_completed(future_to_config):
            config = future_to_config[future]
            try:
                res = future.result()
                if res is not None:
                    results.append(res)
            except Exception as e:
                print(f"  Error resolving future for digest {config['name']}: {e}", file=sys.stderr)
                
    # Sort results in the order defined by DIGEST_CONFIGS
    order_map = {cfg["id"]: i for i, cfg in enumerate(DIGEST_CONFIGS)}
    results.sort(key=lambda x: order_map.get(x["id"], 99))
    return results

TICKER_NAMES = {
    "^GSPC": "S&P 500",
    "VWRP.L": "FTSE All-World",
    "^IXIC": "NASDAQ Composite",
    "^FTSE": "FTSE 100",
    "^DJI": "Dow Jones",
    "^N225": "Nikkei 225",
    "^GDAXI": "DAX Index",
    "^FCHI": "CAC 40",
    "GC=F": "Gold",
    "BZ=F": "Brent Crude",
    "GBPUSD=X": "GBP/USD",
    "EURUSD=X": "EUR/USD",
    "BTC-USD": "Bitcoin",
    "SI=F": "Silver",
    "JPY=X": "USD/JPY",
    "MSFT": "Microsoft",
    "AAPL": "Apple",
    "NVDA": "NVIDIA",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "TSLA": "Tesla",
    "META": "Meta"
}

HEADER_TICKERS = ["^GSPC", "VWRP.L", "^IXIC", "^FTSE"]
INDICES_TICKERS = ["^GSPC", "^FTSE", "^DJI", "^IXIC", "^N225", "^GDAXI", "^FCHI"]
COMMODITIES_FOREX_TICKERS = ["GC=F", "SI=F", "BZ=F", "EURUSD=X", "GBPUSD=X", "JPY=X", "BTC-USD"]
EQUITIES_TICKERS = ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "TSLA", "META"]

def fetch_single_ticker(symbol):
    """Fetch delayed stock price data for a single symbol from Yahoo Finance API with retries."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    ticker_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    retries = 3
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=ticker_headers, timeout=8)
            if response.status_code == 200:
                data = response.json()
                meta = data.get("chart", {}).get("result", [{}])[0].get("meta", {})
                price = meta.get("regularMarketPrice")
                prev_close = meta.get("previousClose")
                
                if price is not None and prev_close is not None:
                    change = price - prev_close
                    change_pct = (change / prev_close) * 100
                    
                    rounded_change = round(change, 2)
                    rounded_change_pct = round(change_pct, 2)
                    
                    # Prevent negative zero (-0.0)
                    if rounded_change == 0.0:
                        rounded_change = 0.0
                    if rounded_change_pct == 0.0:
                        rounded_change_pct = 0.0
                    
                    return {
                        "symbol": symbol,
                        "price": round(price, 2),
                        "change": rounded_change,
                        "change_percent": rounded_change_pct
                    }
            elif response.status_code == 429:
                time.sleep(1 * (attempt + 1))
        except Exception:
            if attempt < retries - 1:
                time.sleep(0.5 * (attempt + 1))
    
    print(f"  Error: Failed to fetch ticker {symbol} after {retries} attempts.", file=sys.stderr)
    return None

def fetch_all_tickers_data(symbols):
    """Fetch data for a list of tickers in parallel."""
    print(f"Fetching {len(symbols)} unique stock tickers in parallel...")
    results = {}
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_symbol = {executor.submit(fetch_single_ticker, sym): sym for sym in symbols}
        for future in as_completed(future_to_symbol):
            sym = future_to_symbol[future]
            try:
                res = future.result()
                if res:
                    results[sym] = res
            except Exception as e:
                print(f"  Error resolving future for {sym}: {e}", file=sys.stderr)
            
    print(f"  Successfully fetched {len(results)} out of {len(symbols)} tickers.")
    return results

def compile_markets_data():
    """Compile and categorize market board tickers."""
    unique_symbols = list(set(HEADER_TICKERS + INDICES_TICKERS + COMMODITIES_FOREX_TICKERS + EQUITIES_TICKERS))
    fetched_data = fetch_all_tickers_data(unique_symbols)
    
    def build_list(symbol_list):
        out = []
        for sym in symbol_list:
            if sym in fetched_data:
                data = fetched_data[sym].copy()
                data["name"] = TICKER_NAMES.get(sym, sym)
                out.append(data)
        return out
        
    header_list = build_list(HEADER_TICKERS)
    indices_list = build_list(INDICES_TICKERS)
    commodities_forex_list = build_list(COMMODITIES_FOREX_TICKERS)
    equities_list = build_list(EQUITIES_TICKERS)
    
    return header_list, {
        "indices": indices_list,
        "commodities_forex": commodities_forex_list,
        "equities": equities_list
    }

def main():
    start_time = time.time()
    
    # 1. Fetch all news sources
    news_stories = []
    
    # RSS Feeds
    news_stories.extend(fetch_all_rss())
    
    # 2. Sort all news stories descending by publication date (newest first)
    # This ensures that our deduplicator keeps the absolute newest version of a story
    news_stories.sort(key=lambda x: x["published"], reverse=True)
    
    # 3. Apply trigram-based deduplication
    deduped_stories = deduplicate_stories(news_stories)
    print(f"Deduplication summary: {len(news_stories)} inputs -> {len(deduped_stories)} outputs.")
    
    # Sort the deduplicated list by consensus score (descending) first, and published date (descending) second
    deduped_stories.sort(key=lambda x: (x["score"], x["published"]), reverse=True)
    
    # 4. Partition into Breaking (published < 3 hours ago) and World (older)
    now = datetime.now(timezone.utc)
    breaking_threshold = now - timedelta(hours=3)
    
    breaking_stories = []
    
    # Track number of breaking stories contributed by each source (max 2)
    breaking_source_counts = {}
    
    for story in deduped_stories:
        pub_dt = parse_iso_date(story["published"])
        src = story["source"]
        
        if pub_dt >= breaking_threshold:
            # Editorial Criteria for Breaking News:
            # - Must be a major story reported by multiple sources (score >= 2) OR
            # - Must be an extremely fresh single-source story (score == 1, published < 45 mins ago)
            minutes_ago = (now - pub_dt).total_seconds() / 60
            is_fresh = minutes_ago < 45
            is_major_breaking = story["score"] >= 2 or (story["score"] == 1 and is_fresh)
            
            if is_major_breaking:
                # Apply per-source cap (maximum of 2 breaking stories per source)
                if breaking_source_counts.get(src, 0) < 2:
                    breaking_source_counts[src] = breaking_source_counts.get(src, 0) + 1
                    breaking_stories.append(story)
            
    # Cap breaking stories at 8 to preserve visual balance
    breaking_stories = breaking_stories[:8]
    
    # Compile patterns (trigrams or exact keys) for each breaking story to verify world overlap
    breaking_ids = {story["url"] for story in breaking_stories}
    breaking_patterns = []
    for story in breaking_stories:
        words = clean_title_for_ngrams(story["title"])
        if len(words) < 3:
            breaking_patterns.append(("exact", " ".join(words)))
        else:
            breaking_patterns.append(("trigrams", get_trigrams(words)))
            
    world_stories = []
    # Track number of world stories contributed by each source (max 6)
    world_source_counts = {}
    
    for story in deduped_stories:
        # Avoid duplicating exactly any story in breaking news
        if story["url"] in breaking_ids:
            continue
            
        pub_dt = parse_iso_date(story["published"])
        src = story["source"]
        
        # Check trigram/exact title overlap with breaking stories
        words = clean_title_for_ngrams(story["title"])
        is_duplicate_of_breaking = False
        
        for pat_type, pat in breaking_patterns:
            if pat_type == "exact":
                if len(words) < 3 and " ".join(words) == pat:
                    is_duplicate_of_breaking = True
                    break
            elif pat_type == "trigrams":
                if len(words) >= 3:
                    trigrams = get_trigrams(words)
                    overlap_count = sum(1 for tg in trigrams if tg in pat)
                    threshold = min(2, len(trigrams))
                    if overlap_count >= threshold:
                        is_duplicate_of_breaking = True
                        break
                
        if is_duplicate_of_breaking:
            print(f"  Filtering out world story '{story['title']}' due to overlap with breaking news.")
            continue
            
        # Apply per-source cap (maximum 6 per source)
        if world_source_counts.get(src, 0) < 6:
            world_source_counts[src] = world_source_counts.get(src, 0) + 1
            world_stories.append(story)
            
    # Cap world stories at 40 to preserve visual balance
    world_stories = world_stories[:MAX_WORLD_STORIES]
    
    # 5. Fetch Web stories (HN, Lobsters & Reddit)
    hn_stories = fetch_hacker_news()
    lobsters_stories = fetch_lobsters()
    
    # Merge and sort Tech & Code stories
    tech_stories = []
    tech_stories.extend(hn_stories)
    tech_stories.extend(lobsters_stories)
    tech_stories.sort(key=lambda x: x["published"], reverse=True)
    tech_stories = tech_stories[:MAX_WEB_STORIES]
    
    reddit_stories = fetch_all_reddit()
    
    # 6. Fetch categorized markets and header ticker data
    ticker_data, market_data = compile_markets_data()
    
    # 6b. Fetch news digests and daily briefings
    digest_data = fetch_all_digests()
    
    # 7. Build final payload
    payload = {
        "last_updated": now.isoformat(),
        "breaking": breaking_stories,
        "world": world_stories,
        "tickers": ticker_data,
        "markets": market_data,
        "digests": digest_data,
        "web": {
            "tech": tech_stories,
            "reddit": reddit_stories
        }
    }
    
    # Sanity check: refuse to write if the pipeline produced an empty or sparse dataset
    if len(world_stories) < 5:
        print(f"Error: Pipeline produced only {len(world_stories)} world stories. Refusing to write empty/corrupted data payload.", file=sys.stderr)
        sys.exit(1)

    # Write payload
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Successfully wrote data payload to {DATA_FILE}")
    except Exception as e:
        print(f"Failed to write data file: {e}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Workflow completed successfully in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
