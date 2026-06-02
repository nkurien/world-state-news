import unittest
from datetime import datetime, timezone
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
        """Test that stories are correctly deduplicated keeping the newest copy (A-A-A)."""
        # Arrange: Prepare mock stories with overlapping content and timestamps
        test_cases = [
            # Case 1: Standard duplicate headlines with varying timestamps
            (
                [
                    {"title": "Germany warns of escalation in Middle East", "published": "2026-06-02T12:00:00Z"},
                    {"title": "Germany warns of escalation in the Middle East conflict", "published": "2026-06-02T10:00:00Z"},
                    {"title": "Unrelated world news story about trade", "published": "2026-06-02T09:00:00Z"},
                ],
                ["Germany warns of escalation in Middle East", "Unrelated world news story about trade"]
            ),
            # Case 2: Multi-word overlap duplicates
            (
                [
                    {"title": "US sanctions Iran's crypto exchange over IRGC links", "published": "2026-06-02T14:00:00Z"},
                    {"title": "US imposes sanctions on Iran crypto exchange due to IRGC links", "published": "2026-06-02T13:00:00Z"},
                    {"title": "Completely different report on weather", "published": "2026-06-02T12:00:00Z"},
                ],
                ["US sanctions Iran's crypto exchange over IRGC links", "Completely different report on weather"]
            ),
            # Case 3: Short titles exact duplicate filter
            (
                [
                    {"title": "Short title", "published": "2026-06-02T12:00:00Z"},
                    {"title": "Short title", "published": "2026-06-02T11:00:00Z"},
                    {"title": "Short body", "published": "2026-06-02T10:00:00Z"},
                ],
                ["Short title", "Short body"]
            )
        ]

        for input_stories, expected_titles in test_cases:
            with self.subTest(case=len(input_stories)):
                # Act: Execute deduplication logic
                result = deduplicate_stories(input_stories)
                result_titles = [story["title"] for story in result]

                # Assert: Check that only unique / newest duplicate headlines remain
                self.assertEqual(result_titles, expected_titles)

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
        # Arrange: Define invalid inputs and capture time bounds before parsing
        invalid_dates = [None, "", "not-a-date", "2026/06/02"]
        now_before = datetime.now(timezone.utc)

        for invalid_str in invalid_dates:
            with self.subTest(invalid_str=invalid_str):
                # Act: Execute parsing with invalid input
                result = parse_iso_date(invalid_str)
                now_after = datetime.now(timezone.utc)

                # Assert: Verify date defaults to present time (within bounds) in UTC
                self.assertIsInstance(result, datetime)
                self.assertEqual(result.tzinfo, timezone.utc)
                self.assertTrue(now_before <= result <= now_after)

if __name__ == "__main__":
    unittest.main()
