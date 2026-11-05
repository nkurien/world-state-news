import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock
from fetch_news import (
    select_reddit_posts,
    fetch_all_reddit,
    parse_reddit_feed,
    carry_forward_digests,
    fetch_single_digest,
    get_with_retry,
    pick_snippet,
    OUTLET_META,
    RSS_SOURCES,
    consensus_score,
    clean_html,
    is_safe_url,
    titles_similar,
    clean_title_for_ngrams,
    get_trigrams,
    deduplicate_stories,
    parse_iso_date,
)


class TestFetchNews(unittest.TestCase):
    def test_clean_title_for_ngrams_parameterized(self):
        """Test cleaning of headlines to verify normalization and stopword removal (A-A-A)."""
        # Arrange: Setup list of headlines and their expected clean token lists
        test_cases = [
            ("The Quick Brown Fox!", ["quick", "brown", "fox"]),
            (
                "Germany warns of escalation in Middle East",
                ["germany", "warns", "escalation", "middle", "east"],
            ),
            ("A and AN and THE are standard stopwords", ["standard", "stopwords"]),
            (
                "Punctuation, like commas: should be stripped.",
                ["punctuation", "like", "commas", "stripped"],
            ),
            ("   Spaces   should be   trimmed.  ", ["spaces", "trimmed"]),
        ]

        for input_headline, expected in test_cases:
            with self.subTest(headline=input_headline):
                # Act: Clean the input headline
                result = clean_title_for_ngrams(input_headline)

                # Assert: Check that the cleaned words match the expected output
                self.assertEqual(result, expected)

    def test_get_trigrams_parameterized(self):
        """Test generation of 3-word n-grams from tokenized words (A-A-A)."""
        # Arrange: Define word lists and their expected trigram sets
        test_cases = [
            (["quick", "brown", "fox"], {("quick", "brown", "fox")}),
            (
                ["germany", "warns", "escalation", "middle", "east"],
                {
                    ("germany", "warns", "escalation"),
                    ("warns", "escalation", "middle"),
                    ("escalation", "middle", "east"),
                },
            ),
            (["short"], set()),
            (["two", "words"], set()),
        ]

        for words, expected in test_cases:
            with self.subTest(words=words):
                # Act: Extract trigrams from the list of words
                result = get_trigrams(words)

                # Assert: Check that trigrams generated match expectations
                self.assertEqual(result, expected)

    def test_deduplicate_stories_parameterized(self):
        """Test that stories are correctly deduplicated keeping the newest copy and updating scores (A-A-A)."""
        # Arrange: Prepare mock stories with overlapping content and timestamps
        test_cases = [
            # Case 1: Standard duplicate headlines with varying timestamps
            (
                [
                    {
                        "title": "Germany warns of escalation in Middle East",
                        "published": "2026-06-02T12:00:00Z",
                        "source": "BBC News",
                    },
                    {
                        "title": "Germany warns of escalation in the Middle East conflict",
                        "published": "2026-06-02T10:00:00Z",
                        "source": "France24",
                    },
                    {
                        "title": "Unrelated world news story about trade",
                        "published": "2026-06-02T09:00:00Z",
                        "source": "Politico EU",
                    },
                ],
                [
                    ("Germany warns of escalation in Middle East", 2, ["France24"]),
                    ("Unrelated world news story about trade", 1, []),
                ],
            ),
            # Case 2: Multi-word overlap duplicates with possessive normalization
            (
                [
                    {
                        "title": "US sanctions Iran's crypto exchange over IRGC links",
                        "published": "2026-06-02T14:00:00Z",
                        "source": "BBC News",
                    },
                    {
                        "title": "US imposes sanctions on Iran crypto exchange due to IRGC links",
                        "published": "2026-06-02T13:00:00Z",
                        "source": "France24",
                    },
                    {
                        "title": "Completely different report on weather",
                        "published": "2026-06-02T12:00:00Z",
                        "source": "Politico EU",
                    },
                ],
                [
                    (
                        "US sanctions Iran's crypto exchange over IRGC links",
                        2,
                        ["France24"],
                    ),
                    ("Completely different report on weather", 1, []),
                ],
            ),
            # Case 3: Short titles exact duplicate filter
            (
                [
                    {
                        "title": "Short title",
                        "published": "2026-06-02T12:00:00Z",
                        "source": "BBC News",
                    },
                    {
                        "title": "Short title",
                        "published": "2026-06-02T11:00:00Z",
                        "source": "France24",
                    },
                    {
                        "title": "Short body",
                        "published": "2026-06-02T10:00:00Z",
                        "source": "Politico EU",
                    },
                ],
                [("Short title", 2, ["France24"]), ("Short body", 1, [])],
            ),
            # Case 4: Prevents false positive transitive chaining (A, B, C are kept separate because overlap is below threshold)
            (
                [
                    {
                        "title": "Crisis in eastern europe deepens today",
                        "published": "2026-06-02T14:00:00Z",
                        "source": "BBC News",
                    },
                    {
                        "title": "Crisis in eastern europe reported",
                        "published": "2026-06-02T13:00:00Z",
                        "source": "France24",
                    },
                    {
                        "title": "Eastern europe reported stable",
                        "published": "2026-06-02T12:00:00Z",
                        "source": "Politico EU",
                    },
                ],
                [
                    ("Crisis in eastern europe deepens today", 1, []),
                    ("Crisis in eastern europe reported", 1, []),
                    ("Eastern europe reported stable", 1, []),
                ],
            ),
            # Case 5: Valid transitive duplicate chaining (C matches A through a mix of A's directly and B's transitively registered trigrams)
            (
                [
                    {
                        "title": "Germany warns of escalation in Middle East conflict",
                        "published": "2026-06-02T14:00:00Z",
                        "source": "BBC News",
                    },
                    {
                        "title": "Germany warns of escalation in Middle East region",
                        "published": "2026-06-02T13:00:00Z",
                        "source": "France24",
                    },
                    {
                        "title": "escalation in Middle East region volatile",
                        "published": "2026-06-02T12:00:00Z",
                        "source": "Politico EU",
                    },
                ],
                [
                    (
                        "Germany warns of escalation in Middle East conflict",
                        3,
                        ["France24", "Politico EU"],
                    )
                ],
            ),
            # Case 6: Multiple duplicates from the same source (listed once in other_sources, counted once)
            (
                [
                    {
                        "title": "New tax policy announced by government",
                        "published": "2026-06-02T14:00:00Z",
                        "source": "BBC News",
                    },
                    {
                        "title": "New tax policy announced by government today",
                        "published": "2026-06-02T13:30:00Z",
                        "source": "France24",
                    },
                    {
                        "title": "New tax policy announced by government yesterday",
                        "published": "2026-06-02T13:00:00Z",
                        "source": "France24",
                    },
                ],
                [
                    (
                        "New tax policy announced by government",
                        2,
                        ["France24"],
                    )
                ],
            ),
            # Case 7: High Jaccard similarity fallback (ceasefire example, trigrams don't meet threshold but Jaccard = 0.8)
            (
                [
                    {
                        "title": "Israel and Lebanon agree to conditional ceasefire",
                        "published": "2026-06-02T14:00:00Z",
                        "source": "France24",
                    },
                    {
                        "title": "Israel and Lebanon agree ceasefire",
                        "published": "2026-06-02T13:00:00Z",
                        "source": "Semafor",
                    },
                ],
                [("Israel and Lebanon agree to conditional ceasefire", 2, ["Semafor"])],
            ),
            # Case 8: Overlap Coefficient fallback for headlines with detailed clauses (e.g. Xi Jinping N Korea story)
            (
                [
                    {
                        "title": "Chinese leader Xi Jinping plans North Korea visit",
                        "published": "2026-06-02T14:00:00Z",
                        "source": "Semafor",
                    },
                    {
                        "title": "Xi Jinping to meet Kim Jong Un in rare visit to North Korea",
                        "published": "2026-06-02T13:00:00Z",
                        "source": "BBC News",
                    },
                ],
                [
                    (
                        "Chinese leader Xi Jinping plans North Korea visit",
                        2,
                        ["BBC News"],
                    )
                ],
            ),
        ]

        for input_stories, expected_results in test_cases:
            with self.subTest(case=len(input_stories)):
                # Act: Execute deduplication logic
                result = deduplicate_stories(input_stories)
                result_details = [
                    (
                        story["title"],
                        story["score"],
                        [os["source"] for os in story["other_sources"]],
                    )
                    for story in result
                ]

                # Assert: Check that headlines, scores, and other_sources list match expected results
                self.assertEqual(result_details, expected_results)

    def test_parse_iso_date_parameterized(self):
        """Test ISO 8601 string parsing with different formats (A-A-A)."""
        # Arrange: Setup ISO 8601 strings and their expected parsed timezone-aware UTC datetime values
        test_cases = [
            (
                "2026-06-02T20:30:00Z",
                datetime(2026, 6, 2, 20, 30, 0, tzinfo=timezone.utc),
            ),
            (
                "2026-06-02T20:30:00+00:00",
                datetime(2026, 6, 2, 20, 30, 0, tzinfo=timezone.utc),
            ),
            (
                "2026-06-02T20:30:00.123456+00:00",
                datetime(2026, 6, 2, 20, 30, 0, 123456, tzinfo=timezone.utc),
            ),
        ]

        for date_str, expected_dt in test_cases:
            with self.subTest(date_str=date_str):
                # Act: Parse string to datetime
                result = parse_iso_date(date_str)

                # Assert: Compare parsed datetime to the target datetime
                self.assertEqual(result, expected_dt)

    def test_parse_iso_date_fallback(self):
        """Test fallback parsing logic on invalid input strings (A-A-A)."""
        # Arrange: Define invalid inputs and expected epoch output
        invalid_dates = [None, "", "not-a-date", "2026/06/02"]
        expected_dt = datetime(1970, 1, 1, tzinfo=timezone.utc)

        for invalid_str in invalid_dates:
            with self.subTest(invalid_str=invalid_str):
                # Act: Execute parsing with invalid input
                result = parse_iso_date(invalid_str)

                # Assert: Verify date defaults to epoch (1970-01-01) in UTC
                self.assertIsInstance(result, datetime)
                self.assertEqual(result.tzinfo, timezone.utc)
                self.assertEqual(result, expected_dt)

    def test_clean_html(self):
        """Test HTML tag removal, entity unescaping, and whitespace normalization (A-A-A)."""
        from fetch_news import clean_html

        test_cases = [
            ("<p>Hello <b>World</b>!</p>", "Hello World!"),
            (
                "Text with &amp; entity and &#8217; curly quote.",
                "Text with & entity and ’ curly quote.",
            ),
            ("Multiple   spaces \n and \t newlines.", "Multiple spaces and newlines."),
            (None, ""),
            ("", ""),
        ]
        for html_input, expected in test_cases:
            with self.subTest(html_input=html_input):
                result = clean_html(html_input)
                self.assertEqual(result, expected)

    def test_truncate_snippet(self):
        """Test snippet truncation at word boundaries close to limit (A-A-A)."""
        from fetch_news import truncate_snippet

        text = "This is a very long sentence that has multiple words and we want to truncate it cleanly without cutting words in half if possible."
        # Truncate at length 40
        res = truncate_snippet(text, max_len=40)
        # Verify it ends with "..." and is less than length 43, and truncates at a space
        self.assertTrue(res.endswith("..."))
        self.assertTrue(len(res) <= 43)
        self.assertEqual(res, "This is a very long sentence that has...")

        # Shorter text should not be truncated
        short_text = "Short sentence."
        self.assertEqual(truncate_snippet(short_text, max_len=40), short_text)

        # None/empty text
        self.assertEqual(truncate_snippet(None), "")
        self.assertEqual(truncate_snippet(""), "")


class TestFetchNewsNetwork(unittest.TestCase):
    @patch("fetch_news.requests.get")
    def test_fetch_rss_feed_success(self, mock_get):
        """Test fetch_rss_feed parses a valid RSS XML response correctly (A-A-A)."""
        from fetch_news import fetch_rss_feed

        # Arrange: Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"""<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title>Escalation in Middle East reported</title>
      <link>https://example.com/escalation-middle-east</link>
      <pubDate>Thu, 04 Jun 2026 12:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Unrelated story</title>
      <link>https://example.com/unrelated</link>
      <pubDate>Thu, 04 Jun 2026 11:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>"""
        mock_get.return_value = mock_response

        # Act: Fetch the RSS feed
        result = fetch_rss_feed("Test Source", "https://example.com/rss")

        # Assert: Verify details of stories
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["title"], "Escalation in Middle East reported")
        self.assertEqual(result[0]["url"], "https://example.com/escalation-middle-east")
        self.assertEqual(result[0]["source"], "Test Source")
        self.assertEqual(result[0]["published"], "2026-06-04T12:00:00+00:00")

    @patch("fetch_news.requests.get")
    def test_fetch_rss_feed_failure(self, mock_get):
        """Test fetch_rss_feed returns empty list on HTTP error (A-A-A)."""
        from fetch_news import fetch_rss_feed

        # Arrange: Mock a requests failure
        mock_get.side_effect = Exception("Network connection timeout")

        # Act: Execute fetch
        result = fetch_rss_feed("Failed Source", "https://example.com/fail")

        # Assert: Verify that list is empty and did not crash the pipeline
        self.assertEqual(result, [])


class TestSimilarityAndGrouping(unittest.TestCase):
    @staticmethod
    def _story(title, source, minute=0):
        return {
            "title": title,
            "url": f"https://example.com/{source}/{minute}",
            "source": source,
            "published": f"2026-06-02T12:{59 - minute:02d}:00Z",
        }

    def test_titles_similar_trigram_and_reordered(self):
        a = clean_title_for_ngrams(
            "Germany warns of escalation in Middle East conflict"
        )
        b = clean_title_for_ngrams("Germany warns of escalation in Middle East region")
        c = clean_title_for_ngrams("Stock markets rally on tech earnings")
        self.assertTrue(titles_similar(a, b))
        self.assertFalse(titles_similar(a, c))

    def test_titles_similar_short_titles_need_exact_match(self):
        self.assertTrue(titles_similar(["live", "updates"], ["live", "updates"]))
        self.assertFalse(titles_similar(["live", "updates"], ["live", "blog"]))
        self.assertFalse(titles_similar([], []))

    def test_same_outlet_listed_once_and_counted_once(self):
        stories = [
            self._story("Bangladesh garment factories shut after unrest", "CNA", 0),
            self._story(
                "Bangladesh garment factories shut after unrest", "Straits Times", 1
            ),
            self._story(
                "Bangladesh garment factories shut after unrest", "Straits Times", 2
            ),
        ]
        result = deduplicate_stories(stories)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["score"], 2)
        self.assertEqual(
            [o["source"] for o in result[0]["other_sources"]], ["Straits Times"]
        )

    def test_canonical_outlet_not_listed_in_its_own_other_sources(self):
        stories = [
            self._story("Bangladesh garment factories shut after unrest", "CNA", 0),
            self._story("Bangladesh garment factories shut after unrest", "CNA", 1),
        ]
        result = deduplicate_stories(stories)
        self.assertEqual(result[0]["score"], 1)
        self.assertEqual(result[0]["other_sources"], [])

    def test_merging_does_not_steal_trigrams_from_other_cluster(self):
        # B merges into A but contains a trigram owned by C; D must still group with C.
        c = self._story("quake strikes coastal region tonight", "BBC News", 0)
        a = self._story("senate passes budget bill friday", "France24", 1)
        b = self._story(
            "senate passes budget bill strikes coastal region", "Politico EU", 2
        )
        d = self._story("quake strikes coastal region", "Al Jazeera", 3)
        result = deduplicate_stories([c, a, b, d])
        by_title = {r["title"]: r for r in result}
        self.assertEqual(len(result), 2)
        self.assertEqual(
            [o["source"] for o in by_title[c["title"]]["other_sources"]], ["Al Jazeera"]
        )
        self.assertEqual(
            [o["source"] for o in by_title[a["title"]]["other_sources"]],
            ["Politico EU"],
        )


class TestSanitizing(unittest.TestCase):
    def test_clean_html_strips_entity_encoded_tags(self):
        self.assertEqual(
            clean_html("&lt;p&gt;Hello &amp; welcome&lt;/p&gt;"), "Hello & welcome"
        )

    def test_clean_html_strips_plain_tags_and_whitespace(self):
        self.assertEqual(clean_html("<p>Hi   <b>there</b></p>\n"), "Hi there")

    def test_clean_html_drops_literal_null(self):
        self.assertEqual(clean_html("null"), "")
        self.assertEqual(clean_html(" <p>NULL</p> "), "")
        self.assertEqual(clean_html("null and void"), "null and void")

    def test_is_safe_url(self):
        self.assertTrue(is_safe_url("https://example.com/a?b=1"))
        self.assertTrue(is_safe_url("http://example.com"))
        for bad in [
            "javascript:alert(1)",
            "data:text/html,x",
            "//example.com",
            "/relative",
            "",
            None,
            "ftp://x.com",
        ]:
            self.assertFalse(is_safe_url(bad), bad)


class TestConsensusScore(unittest.TestCase):
    @staticmethod
    def _story(source, others):
        return {"source": source, "other_sources": [{"source": o} for o in others]}

    def test_single_outlet(self):
        self.assertEqual(consensus_score(self._story("BBC News", [])), 1)

    def test_commonly_owned_outlets_count_once(self):
        # Euronews owns Africanews: one family, two regions -> 1 + 0.5
        self.assertEqual(consensus_score(self._story("Euronews", ["Africanews"])), 1.5)
        # France24 and RFI share an owner and a region
        self.assertEqual(consensus_score(self._story("France24", ["RFI English"])), 1)

    def test_regional_diversity_outranks_one_region(self):
        asia_only = self._story("CNA", ["Straits Times", "SCMP"])
        global_mix = self._story("BBC News", ["NPR World", "CNA"])
        self.assertEqual(consensus_score(asia_only), 3)
        self.assertEqual(consensus_score(global_mix), 4)
        self.assertGreater(consensus_score(global_mix), consensus_score(asia_only))

    def test_unknown_outlets_are_their_own_family(self):
        self.assertEqual(
            consensus_score(self._story("New Outlet", ["Other Outlet"])), 2
        )

    def test_every_rss_source_has_outlet_metadata(self):
        # A new feed without an entry would silently count as its own family
        self.assertEqual([n for n, _ in RSS_SOURCES if n not in OUTLET_META], [])

    def test_empty_story(self):
        self.assertEqual(consensus_score({"other_sources": []}), 0)


class TestSourceHealth(unittest.TestCase):
    @patch("fetch_news.time.sleep")
    @patch("fetch_news.requests.get")
    def test_failed_feed_reported_as_not_ok(self, mock_get, _sleep):
        from fetch_news import fetch_rss_feed

        mock_get.side_effect = Exception("boom")
        health = {}
        self.assertEqual(fetch_rss_feed("X", "https://example.com", health=health), [])
        self.assertFalse(health["ok"])
        self.assertEqual(health["count"], 0)
        self.assertIn("boom", health["error"])

    @patch("fetch_news.requests.get")
    def test_empty_but_successful_feed_is_ok(self, mock_get):
        from fetch_news import fetch_rss_feed

        resp = MagicMock()
        resp.content = (
            b'<?xml version="1.0"?><rss version="2.0"><channel></channel></rss>'
        )
        mock_get.return_value = resp
        health = {}
        self.assertEqual(fetch_rss_feed("X", "https://example.com", health=health), [])
        self.assertTrue(health["ok"])
        self.assertEqual(health["count"], 0)

    @patch("fetch_news.RSS_SOURCES", [("Good", "https://g"), ("Bad", "https://b")])
    @patch("fetch_news.fetch_rss_feed")
    def test_fetch_all_rss_collects_health_per_source(self, mock_feed):
        from fetch_news import fetch_all_rss

        def fake(name, url, health=None):
            if name == "Good":
                health.update(ok=True, count=1, error=None)
                return [
                    {
                        "title": "t",
                        "url": "https://x",
                        "source": "Good",
                        "published": "2026-01-01T00:00:00+00:00",
                    }
                ]
            health.update(ok=False, count=0, error="down")
            return []

        mock_feed.side_effect = fake
        health = {}
        stories = fetch_all_rss(health)
        self.assertEqual(len(stories), 1)
        self.assertTrue(health["Good"]["ok"])
        self.assertFalse(health["Bad"]["ok"])
        self.assertIsInstance(health["Good"]["ms"], int)


class TestPickSnippet(unittest.TestCase):
    def test_prefers_summary_over_content(self):
        entry = {
            "summary": "A proper one-line dek",
            "content": [{"value": "<p>Author bio that is not the article</p>"}],
        }
        self.assertEqual(pick_snippet(entry), "A proper one-line dek")

    def test_the_week_null_placeholder_in_content(self):
        entry = {
            "summary": "",
            "content": [{"value": "null"}, {"value": "<p>Body text</p>"}],
        }
        self.assertEqual(pick_snippet(entry), "Body text")

    def test_falls_back_to_description(self):
        self.assertEqual(
            pick_snippet({"description": "From description"}), "From description"
        )

    def test_nothing_usable_returns_empty(self):
        self.assertEqual(
            pick_snippet({"summary": "null", "content": [{"value": " "}]}), ""
        )
        self.assertEqual(pick_snippet({}), "")

    def test_truncates(self):
        self.assertLessEqual(len(pick_snippet({"summary": "word " * 100})), 163)


class TestDigestResilience(unittest.TestCase):
    @staticmethod
    def _resp(status, headers=None):
        r = MagicMock()
        r.status_code = status
        r.headers = headers or {}
        r.raise_for_status.side_effect = (
            None if status < 400 else Exception(f"{status} error")
        )
        return r

    @patch("fetch_news.time.sleep")
    @patch("fetch_news.requests.get")
    def test_retries_429_then_succeeds_and_honours_retry_after(
        self, mock_get, mock_sleep
    ):
        mock_get.side_effect = [self._resp(429, {"Retry-After": "3"}), self._resp(200)]
        resp = get_with_retry("https://example.com")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(mock_get.call_count, 2)
        mock_sleep.assert_called_once_with(4)  # Retry-After + 1s slack

    @patch("fetch_news.time.sleep")
    @patch("fetch_news.requests.get")
    def test_gives_up_after_retries(self, mock_get, _sleep):
        mock_get.side_effect = [self._resp(429)] * 3
        with self.assertRaises(Exception):
            get_with_retry("https://example.com", retries=2)
        self.assertEqual(mock_get.call_count, 3)

    @patch("fetch_news.time.sleep")
    @patch("fetch_news.requests.get")
    def test_does_not_retry_client_errors(self, mock_get, _sleep):
        mock_get.side_effect = [self._resp(404)]
        with self.assertRaises(Exception):
            get_with_retry("https://example.com")
        self.assertEqual(mock_get.call_count, 1)

    def test_carry_forward_fills_only_empty_digests(self):
        prev = {
            "digests": [
                {"id": "a", "entries": [{"title": "old a"}]},
                {"id": "b", "entries": [{"title": "old b"}]},
            ]
        }
        new = [
            {"id": "a", "name": "A", "entries": []},
            {"id": "b", "name": "B", "entries": [{"title": "fresh b"}]},
            {"id": "c", "name": "C", "entries": []},
        ]
        out = carry_forward_digests(new, prev)
        self.assertEqual(out[0]["entries"], [{"title": "old a"}])
        self.assertTrue(out[0]["stale"])
        self.assertEqual(out[1]["entries"], [{"title": "fresh b"}])
        self.assertNotIn("stale", out[1])
        self.assertEqual(out[2]["entries"], [])

    def test_carry_forward_without_previous_payload(self):
        new = [{"id": "a", "name": "A", "entries": []}]
        self.assertEqual(carry_forward_digests(new, None)[0]["entries"], [])


class TestDigestPromoFilter(unittest.TestCase):
    FEED = b"""<?xml version="1.0"?><rss version="2.0"><channel>
<item><title>Start your subscription today</title>
<link>https://theweek.com/subscription/promo</link></item>
<item><title>Real one</title><link>https://theweek.com/politics/a</link></item>
<item><title>Real two</title><link>https://theweek.com/politics/b</link></item>
<item><title>Real three</title><link>https://theweek.com/politics/c</link></item>
<item><title>Real four</title><link>https://theweek.com/politics/d</link></item>
</channel></rss>"""

    @patch("fetch_news.get_with_retry")
    def test_skips_promo_links_and_still_returns_three(self, mock_get):
        mock_get.return_value = MagicMock(content=self.FEED)
        out = fetch_single_digest(
            {
                "id": "x",
                "name": "X",
                "url": "https://example.com/rss",
                "column": "briefings",
                "headers": {},
                "skip_url_substrings": ["/subscription/"],
            }
        )
        self.assertEqual(
            [e["title"] for e in out["entries"]], ["Real one", "Real two", "Real three"]
        )


class TestReddit(unittest.TestCase):
    @staticmethod
    def _resp(status, headers=None, content=b""):
        r = MagicMock()
        r.status_code = status
        r.headers = headers or {}
        r.content = content
        r.raise_for_status.side_effect = (
            None if status < 400 else Exception(f"{status}")
        )
        return r

    _FEED_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <category term="technology" label="r/technology"/>
    <title>Chip maker announces layoffs</title>
    <link href="https://www.reddit.com/r/technology/comments/abc/chip_maker/"/>
    <content type="html">&lt;a href="https://example.com/story"&gt;[link]&lt;/a&gt;</content>
    <updated>{ago1}</updated>
  </entry>
  <entry>
    <category term="europe" label="r/europe"/>
    <title>Self post</title>
    <link href="https://www.reddit.com/r/europe/comments/def/self_post/"/>
    <updated>{ago2}</updated>
  </entry>
  <entry>
    <title>Bad link</title>
    <link href="javascript:alert(1)"/>
    <updated>{ago3}</updated>
  </entry>
</feed>"""

    # Dates are relative to now: fetch_all_reddit drops posts older than 72h,
    # so fixed dates would make these tests start failing once they age out.
    FEED = (
        _FEED_TEMPLATE.replace(
            "{ago1}", (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        )
        .replace(
            "{ago2}", (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        )
        .replace(
            "{ago3}", (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        )
        .encode()
    )

    def test_parse_reddit_feed_uses_category_for_subreddit_and_direct_link(self):
        import feedparser

        stories = parse_reddit_feed(feedparser.parse(self.FEED))
        self.assertEqual([s["source"] for s in stories], ["/r/technology", "/r/europe"])
        self.assertEqual(stories[0]["url"], "https://example.com/story")
        self.assertEqual(
            stories[0]["comments_url"],
            "https://www.reddit.com/r/technology/comments/abc/chip_maker/",
        )
        # Self post links to its own thread
        self.assertEqual(stories[1]["url"], stories[1]["comments_url"])

    @patch("fetch_news.requests.get")
    def test_fetch_all_reddit_makes_a_single_request(self, mock_get):
        mock_get.return_value = self._resp(200, content=self.FEED)
        stories = fetch_all_reddit()
        self.assertEqual(mock_get.call_count, 1)
        self.assertIn("worldnews+europe", mock_get.call_args[0][0])
        self.assertEqual(len(stories), 2)
        # Feed (hot) order is preserved even though the second post is older
        self.assertEqual([s["source"] for s in stories], ["/r/technology", "/r/europe"])

    @patch("fetch_news.time.sleep")
    @patch("fetch_news.requests.get")
    def test_fetch_all_reddit_waits_out_rate_limit_reset(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            self._resp(429, {"X-Ratelimit-Reset": "20"}),
            self._resp(200, content=self.FEED),
        ]
        self.assertEqual(len(fetch_all_reddit()), 2)
        mock_sleep.assert_called_once_with(21)

    @patch("fetch_news.time.sleep")
    @patch("fetch_news.requests.get")
    def test_fetch_all_reddit_returns_empty_when_rate_limited_throughout(
        self, mock_get, _sleep
    ):
        mock_get.return_value = self._resp(429, {"X-Ratelimit-Reset": "20"})
        self.assertEqual(fetch_all_reddit(), [])
        self.assertEqual(mock_get.call_count, 3)


class TestSelectRedditPosts(unittest.TestCase):
    NOW = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)

    @classmethod
    def _post(cls, title, hours_ago, sub="worldnews", url=None):
        return {
            "title": title,
            "url": url or f"https://example.com/{abs(hash(title))}",
            "source": f"/r/{sub}",
            "published": (cls.NOW - timedelta(hours=hours_ago)).isoformat(),
        }

    def test_keeps_hot_order_not_recency(self):
        posts = [
            self._post("Older but hotter story about parliament vote", 10),
            self._post("Newer story about central bank decision", 1),
        ]
        out = select_reddit_posts(posts, now=self.NOW)
        self.assertEqual([p["title"] for p in out], [p["title"] for p in posts])

    def test_drops_stale_pinned_posts(self):
        posts = [
            self._post("Megathread: community rules and guidelines", 24 * 59),
            self._post("Fresh story about parliament vote", 2),
        ]
        out = select_reddit_posts(posts, now=self.NOW)
        self.assertEqual(
            [p["title"] for p in out], ["Fresh story about parliament vote"]
        )

    def test_drops_cross_posts_keeping_best_ranked(self):
        dup_url = "https://example.com/dup"
        posts = [
            self._post(
                "Supermarket freezers across Europe struck in second cyberattack",
                3,
                "europe",
            ),
            self._post("Unrelated story about lithium mining regulation", 4, "science"),
            self._post(
                "Supermarket freezers across Europe struck in second cyberattack",
                3,
                "technology",
            ),
            self._post("Different headline", 2, "worldnews", url=dup_url),
            self._post("Another headline entirely", 2, "worldnews", url=dup_url),
        ]
        out = select_reddit_posts(posts, now=self.NOW)
        self.assertEqual(
            [(p["source"], p["title"][:12]) for p in out],
            [
                ("/r/europe", "Supermarket "),
                ("/r/science", "Unrelated st"),
                ("/r/worldnews", "Different he"),
            ],
        )

    def test_respects_limit(self):
        posts = [
            self._post(f"Distinct topic number {w}", 1)
            for w in ["alpha beta", "gamma delta", "epsilon zeta", "eta theta"]
        ]
        self.assertEqual(len(select_reddit_posts(posts, now=self.NOW, limit=2)), 2)


if __name__ == "__main__":
    unittest.main()
