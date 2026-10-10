import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import spotify_transcribe as st  # noqa: E402


@pytest.mark.parametrize(
    "value",
    [
        "https://open.spotify.com/episode/2tUttFvmQYPFrJqgw9VWLj?si=ZmrqOeM0RvW-L0JzctwAJQ&utm_source=copy-link",
        "https://open.spotify.com/intl-de/episode/2tUttFvmQYPFrJqgw9VWLj",
        "spotify:episode:2tUttFvmQYPFrJqgw9VWLj",
        "2tUttFvmQYPFrJqgw9VWLj",
    ],
)
def test_parse_episode_id(value):
    assert st.parse_episode_id(value) == "2tUttFvmQYPFrJqgw9VWLj"


def test_parse_episode_id_rejects_other_links():
    with pytest.raises(ValueError):
        st.parse_episode_id("https://open.spotify.com/track/abc")


def test_parse_episode_page():
    page = """
    <meta property="og:title" content="Ep. 42: Building &amp; Shipping"/>
    <meta property="og:description" content="Listen to this episode from The Test Show on Spotify. Blah."/>
    <meta name="music:release_date" content="2024-05-01"/>
    <meta name="music:duration" content="3600"/>
    """
    info = st.parse_episode_page(page)
    assert info.title == "Ep. 42: Building & Shipping"
    assert info.show == "The Test Show"
    assert info.release_date == date(2024, 5, 1)
    assert info.duration_s == 3600


FEED = b"""<?xml version="1.0"?>
<rss xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd"><channel>
  <item>
    <title>#43 - Something Else</title>
    <pubDate>Wed, 08 May 2024 06:00:00 GMT</pubDate>
    <enclosure url="https://cdn.example.com/43.mp3" type="audio/mpeg"/>
    <itunes:duration>59:00</itunes:duration>
  </item>
  <item>
    <title>#42 - Building &amp; Shipping</title>
    <pubDate>Wed, 01 May 2024 06:00:00 GMT</pubDate>
    <enclosure url="https://cdn.example.com/42.mp3?x=1" type="audio/mpeg"/>
    <itunes:duration>01:00:00</itunes:duration>
  </item>
  <item><title>No audio</title></item>
</channel></rss>"""


def test_parse_feed_and_match():
    episodes = st.parse_feed(FEED)
    assert len(episodes) == 2
    assert episodes[1].duration_s == 3600

    target = st.EpisodeInfo("Ep. 42: Building & Shipping", "The Test Show", date(2024, 5, 1), 3600)
    ep, score = st.match_episode(target, episodes)
    assert ep.audio_url == "https://cdn.example.com/42.mp3?x=1"
    assert score > st.MATCH_THRESHOLD


def test_match_episode_returns_none_when_nothing_close():
    episodes = st.parse_feed(FEED)
    assert st.match_episode(st.EpisodeInfo("Totally unrelated title"), episodes) is None


@pytest.mark.parametrize("raw,expected", [("3600", 3600), ("1:02:03", 3723), ("05:30", 330), ("", None), ("x", None)])
def test_parse_duration(raw, expected):
    assert st.parse_duration(raw) == expected


def test_srt_and_text_output():
    segs = [{"start": 0.0, "end": 2.5, "text": "Hello."}, {"start": 3661.2, "end": 3662.0, "text": "Bye."}]
    srt = st.to_srt(segs)
    assert "1\n00:00:00,000 --> 00:00:02,500\nHello.\n" in srt
    assert "2\n01:01:01,200 --> 01:01:02,000\nBye.\n" in srt
    assert st.to_text(segs, timestamps=True) == "[00:00:00] Hello.\n[01:01:01] Bye.\n"


def test_write_outputs(tmp_path):
    paths = st.write_outputs([{"start": 0, "end": 1, "text": "Hi"}], tmp_path / "out" / "ep", {"title": "t"})
    assert sorted(p.suffix for p in paths) == [".json", ".srt", ".txt"]
    assert all(p.exists() for p in paths)
