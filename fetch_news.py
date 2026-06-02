import json
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
    """
    seen_trigrams = {} # Maps trigram tuple -> story dict
    seen_exact = {}     # Maps short exact title key -> story dict
    deduped = []
    
    for story in stories:
        story["score"] = 1 # Initialize consensus score
        words = clean_title_for_ngrams(story["title"])
        
        # If the title is very short, fallback to exact title comparison
        if len(words) < 3:
            exact_key = " ".join(words)
            if exact_key in seen_exact:
                # Vote up the original kept story
                seen_exact[exact_key]["score"] += 1
                continue
            seen_exact[exact_key] = story
            deduped.append(story)
            continue
            
        trigrams = get_trigrams(words)
        
        # Check if any trigram is already seen
        duplicate_story = None
        for tg in trigrams:
            if tg in seen_trigrams:
                duplicate_story = seen_trigrams[tg]
                break
                
        if duplicate_story:
            # We found a duplicate! Increment canonical story's score
            duplicate_story["score"] += 1
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
        return datetime.now(timezone.utc)
    try:
        # Standard replacement for UTC indicator
        iso_str = date_str.replace("Z", "+00:00")
        # In Python 3.11+, fromisoformat handles decimal seconds and timezone offsets cleanly
        return datetime.fromisoformat(iso_str)
    except Exception:
        # Fallback to current time if parsing fails
        return datetime.now(timezone.utc)

def fetch_rss_feed(source_name, url):
    """Fetch and parse a standard RSS feed using feedparser."""
    print(f"Fetching RSS feed from {source_name}...")
    stories = []
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
    except Exception as e:
        print(f"  Error fetching {source_name}: {e}", file=sys.stderr)
    return stories

def fetch_reuters_sitemap():
    """Fetch and parse the Reuters news sitemap XML directly."""
    print("Fetching Reuters news sitemap...")
    url = "https://www.reuters.com/arc/outboundfeeds/news-sitemap/?outputType=xml"
    stories = []
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        
        # Parse XML
        root = ET.fromstring(response.content)
        
        # XML Namespaces used in sitemaps
        ns = {
            'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9',
            'news': 'http://www.google.com/schemas/sitemap-news/0.9'
        }
        
        for url_node in root.findall("sitemap:url", ns):
            loc_node = url_node.find("sitemap:loc", ns)
            news_node = url_node.find("news:news", ns)
            
            if loc_node is None or news_node is None:
                continue
                
            link = loc_node.text.strip()
            title_node = news_node.find("news:title", ns)
            date_node = news_node.find("news:publication_date", ns)
            
            if title_node is None or not title_node.text:
                continue
                
            title = title_node.text.strip()
            pub_date = parse_iso_date(date_node.text if date_node is not None else None)
            
            # Filter: Skip non-English articles based on path segments
            if any(segment in link for segment in ["/es/", "/pt/", "/de/", "/fr/", "/it/", "/jp/", "/latam/"]):
                continue
                
            stories.append({
                "title": title,
                "url": link,
                "source": "Reuters",
                "published": pub_date.isoformat()
            })
            
        print(f"  Successfully fetched {len(stories)} English articles from Reuters sitemap.")
    except Exception as e:
        print(f"  Error fetching Reuters sitemap: {e}", file=sys.stderr)
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
    """Fetch hot posts from a specific subreddit."""
    print(f"Fetching Reddit /r/{subreddit}...")
    stories = []
    url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={MAX_WEB_STORIES}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        for post_wrapper in data.get("data", {}).get("children", []):
            post = post_wrapper.get("data", {})
            # Ignore stickied posts
            if post.get("stickied"):
                continue
                
            title = post.get("title", "").strip()
            link = post.get("url", "").strip()
            score = post.get("score", 0)
            created_utc = post.get("created_utc", time.time())
            pub_date = datetime.fromtimestamp(created_utc, timezone.utc)
            
            if not title or not link:
                continue
                
            # Clean up URLs (relative Reddit links should be full links)
            if link.startswith("/r/"):
                link = f"https://www.reddit.com{link}"
                
            stories.append({
                "title": title,
                "url": link,
                "score": score,
                "source": f"/r/{subreddit}",
                "published": pub_date.isoformat()
            })
        print(f"  Successfully fetched {len(stories)} posts from /r/{subreddit}.")
    except Exception as e:
        print(f"  Error fetching /r/{subreddit}: {e}", file=sys.stderr)
    return stories

def fetch_all_reddit():
    """Fetch and merge Reddit stories from r/worldnews and r/europe."""
    reddit_stories = []
    reddit_stories.extend(fetch_reddit_subreddit("worldnews"))
    reddit_stories.extend(fetch_reddit_subreddit("europe"))
    
    # Sort merged list by score descending, limit to top 15
    reddit_stories.sort(key=lambda x: x["score"], reverse=True)
    return reddit_stories[:MAX_WEB_STORIES]

def main():
    start_time = time.time()
    
    # 1. Fetch all news sources
    news_stories = []
    
    # RSS Feeds
    news_stories.extend(fetch_rss_feed("France24", "https://www.france24.com/en/rss"))
    news_stories.extend(fetch_rss_feed("Politico EU", "https://www.politico.eu/feed/"))
    news_stories.extend(fetch_rss_feed("Semafor", "https://www.semafor.com/rss.xml"))
    news_stories.extend(fetch_rss_feed("BBC News", "https://feeds.bbci.co.uk/news/world/rss.xml"))
    news_stories.extend(fetch_rss_feed("The Independent", "https://www.independent.co.uk/news/world/rss"))
    
    # Custom XML Sitemap Feed
    news_stories.extend(fetch_reuters_sitemap())
    
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
    world_stories = []
    
    for story in deduped_stories:
        pub_dt = parse_iso_date(story["published"])
        if pub_dt >= breaking_threshold:
            breaking_stories.append(story)
        else:
            world_stories.append(story)
            
    # Limit world stories to keep UI clean and balanced
    world_stories = world_stories[:MAX_WORLD_STORIES]
    
    # 5. Fetch Web stories (HN & Reddit)
    hn_stories = fetch_hacker_news()
    reddit_stories = fetch_all_reddit()
    
    # 6. Build final payload
    payload = {
        "last_updated": now.isoformat(),
        "breaking": breaking_stories,
        "world": world_stories,
        "web": {
            "hacker_news": hn_stories,
            "reddit": reddit_stories
        }
    }
    
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
