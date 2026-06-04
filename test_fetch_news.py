import unittest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
from fetch_news import clean_title_for_ngrams, get_trigrams, deduplicate_stories, parse_iso_date

class TestFetchNews(unittest.TestCase):

    def test_clean_title_for_ngrams_parameterized(self):
        """Test cleaning of headlines to verify normalization and stopword removal (A-A-A)."""
        # Arrange: Setup list of headlines and their expected clean token lists
        test_cases = [
            ("The Quick Brown Fox!", ["quick", "brown", "fox"]),
            ("Germany warns of escalation in Middle East", ["germany", "warns", "escalation", "middle", "east"]),
            ("A and AN and THE are standard stopwords", ["standard", "stopwords"]),
            ("Punctuation, like commas: should be stripped.", ["punctuation", "like", "commas", "stripped"]),
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
                    ("escalation", "middle", "east")
                }
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
                    {"title": "Germany warns of escalation in Middle East", "published": "2026-06-02T12:00:00Z"},
                    {"title": "Germany warns of escalation in the Middle East conflict", "published": "2026-06-02T10:00:00Z"},
                    {"title": "Unrelated world news story about trade", "published": "2026-06-02T09:00:00Z"},
                ],
                [
                    ("Germany warns of escalation in Middle East", 2),
                    ("Unrelated world news story about trade", 1)
                ]
            ),
            # Case 2: Multi-word overlap duplicates with possessive normalization
            (
                [
                    {"title": "US sanctions Iran's crypto exchange over IRGC links", "published": "2026-06-02T14:00:00Z"},
                    {"title": "US imposes sanctions on Iran crypto exchange due to IRGC links", "published": "2026-06-02T13:00:00Z"},
                    {"title": "Completely different report on weather", "published": "2026-06-02T12:00:00Z"},
                ],
                [
                    ("US sanctions Iran's crypto exchange over IRGC links", 2),
                    ("Completely different report on weather", 1)
                ]
            ),
            # Case 3: Short titles exact duplicate filter
            (
                [
                    {"title": "Short title", "published": "2026-06-02T12:00:00Z"},
                    {"title": "Short title", "published": "2026-06-02T11:00:00Z"},
                    {"title": "Short body", "published": "2026-06-02T10:00:00Z"},
                ],
                [
                    ("Short title", 2),
                    ("Short body", 1)
                ]
            ),
            # Case 4: Prevents false positive transitive chaining (A, B, C are kept separate because overlap is below threshold)
            (
                [
                    {"title": "Crisis in eastern europe deepens today", "published": "2026-06-02T14:00:00Z"},
                    {"title": "Crisis in eastern europe reported", "published": "2026-06-02T13:00:00Z"},
                    {"title": "Eastern europe reported stable", "published": "2026-06-02T12:00:00Z"},
                ],
                [
                    ("Crisis in eastern europe deepens today", 1),
                    ("Crisis in eastern europe reported", 1),
                    ("Eastern europe reported stable", 1)
                ]
            ),
            # Case 5: Valid transitive duplicate chaining (C matches A through a mix of A's directly and B's transitively registered trigrams)
            (
                [
                    {"title": "Germany warns of escalation in Middle East conflict", "published": "2026-06-02T14:00:00Z"},
                    {"title": "Germany warns of escalation in Middle East region", "published": "2026-06-02T13:00:00Z"},
                    {"title": "escalation in Middle East region volatile", "published": "2026-06-02T12:00:00Z"},
                ],
                [
                    ("Germany warns of escalation in Middle East conflict", 3)
                ]
            )
        ]

        for input_stories, expected_results in test_cases:
            with self.subTest(case=len(input_stories)):
                # Act: Execute deduplication logic
                result = deduplicate_stories(input_stories)
                result_titles_and_scores = [(story["title"], story["score"]) for story in result]

                # Assert: Check that only unique / newest duplicate headlines remain with correct consensus scores
                self.assertEqual(result_titles_and_scores, expected_results)

    def test_parse_iso_date_parameterized(self):
        """Test ISO 8601 string parsing with different formats (A-A-A)."""
        # Arrange: Setup ISO 8601 strings and their expected parsed timezone-aware UTC datetime values
        test_cases = [
            ("2026-06-02T20:30:00Z", datetime(2026, 6, 2, 20, 30, 0, tzinfo=timezone.utc)),
            ("2026-06-02T20:30:00+00:00", datetime(2026, 6, 2, 20, 30, 0, tzinfo=timezone.utc)),
            ("2026-06-02T20:30:00.123456+00:00", datetime(2026, 6, 2, 20, 30, 0, 123456, tzinfo=timezone.utc)),
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
            ("Text with &amp; entity and &#8217; curly quote.", "Text with & entity and ’ curly quote."),
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

    @patch('fetch_news.requests.get')
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

    @patch('fetch_news.requests.get')
    def test_fetch_rss_feed_failure(self, mock_get):
        """Test fetch_rss_feed returns empty list on HTTP error (A-A-A)."""
        from fetch_news import fetch_rss_feed
        # Arrange: Mock a requests failure
        mock_get.side_effect = Exception("Network connection timeout")

        # Act: Execute fetch
        result = fetch_rss_feed("Failed Source", "https://example.com/fail")

        # Assert: Verify that list is empty and did not crash the pipeline
        self.assertEqual(result, [])

if __name__ == "__main__":
    unittest.main()
