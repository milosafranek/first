#!/usr/bin/env python3
"""Transcribe a podcast episode from its Spotify link.

Spotify streams podcast audio in a protected format, so this tool doesn't
download from Spotify. It does this instead:

1. Reads the episode's metadata (title, show, date, length) from Spotify.
2. Finds the show's public RSS feed through the Apple Podcasts directory.
3. Matches the episode in that feed and downloads the publisher's own audio file.
4. Transcribes the audio locally with Whisper (faster-whisper).

Episodes that are exclusive to Spotify have no public feed, so they can't be
found this way. For those, pass your own audio file with --audio.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import sys
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests

USER_AGENT = "Mozilla/5.0 (compatible; spotify-transcribe/1.0)"
ITUNES_NS = "{http://www.itunes.com/dtds/podcast-1.0.dtd}"
MATCH_THRESHOLD = 0.6


@dataclass
class EpisodeInfo:
    title: str
    show: str | None = None
    release_date: date | None = None
    duration_s: float | None = None


@dataclass
class FeedEpisode:
    title: str
    audio_url: str
    pub_date: date | None = None
    duration_s: float | None = None


# ---------------------------------------------------------------------------
# Spotify metadata
# ---------------------------------------------------------------------------

def parse_episode_id(url_or_id: str) -> str:
    """Return the 22-char Spotify episode ID from a URL, URI or bare ID."""
    m = re.search(r"(?:episode[/:])([A-Za-z0-9]{22})", url_or_id)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9]{22}", url_or_id):
        return url_or_id
    raise ValueError(f"Not a Spotify episode link: {url_or_id}")


def fetch_episode_via_api(episode_id: str, client_id: str, client_secret: str) -> EpisodeInfo:
    """Use the Spotify Web API (client-credentials flow)."""
    auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    tok = requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "client_credentials"},
        headers={"Authorization": f"Basic {auth}"},
        timeout=20,
    )
    tok.raise_for_status()
    r = requests.get(
        f"https://api.spotify.com/v1/episodes/{episode_id}",
        params={"market": os.environ.get("SPOTIFY_MARKET", "US")},
        headers={"Authorization": f"Bearer {tok.json()['access_token']}"},
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()
    return EpisodeInfo(
        title=data["name"],
        show=data.get("show", {}).get("name"),
        release_date=_parse_iso_date(data.get("release_date")),
        duration_s=data.get("duration_ms", 0) / 1000 or None,
    )


def fetch_episode_via_page(episode_id: str) -> EpisodeInfo:
    """Read the public episode web page's meta tags (no credentials needed)."""
    r = requests.get(
        f"https://open.spotify.com/episode/{episode_id}",
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en"},
        timeout=20,
    )
    r.raise_for_status()
    return parse_episode_page(r.text)


def parse_episode_page(page: str) -> EpisodeInfo:
    def meta(key: str) -> str | None:
        m = re.search(
            rf'<meta[^>]+(?:property|name)="{re.escape(key)}"[^>]+content="([^"]*)"', page
        )
        return html.unescape(m.group(1)) if m else None

    title = meta("og:title")
    if not title:
        raise RuntimeError("Couldn't read the episode title from the Spotify page.")

    show = None
    desc = meta("og:description") or meta("description") or ""
    m = re.search(r"Listen to this episode from (.+?) on Spotify", desc)
    if m:
        show = m.group(1).strip()

    duration = meta("music:duration")
    release = meta("music:release_date")
    return EpisodeInfo(
        title=title,
        show=show,
        release_date=_parse_iso_date(release),
        duration_s=float(duration) if duration and duration.isdigit() else None,
    )


def get_episode_info(episode_id: str) -> EpisodeInfo:
    cid, secret = os.environ.get("SPOTIFY_CLIENT_ID"), os.environ.get("SPOTIFY_CLIENT_SECRET")
    if cid and secret:
        try:
            return fetch_episode_via_api(episode_id, cid, secret)
        except requests.RequestException as e:
            log(f"Spotify API lookup failed ({e}); falling back to the web page.")
    return fetch_episode_via_page(episode_id)


# ---------------------------------------------------------------------------
# RSS feed discovery and episode matching
# ---------------------------------------------------------------------------

def find_feed_urls(show: str, limit: int = 5) -> list[str]:
    """Search the Apple Podcasts directory for the show's RSS feed."""
    r = requests.get(
        "https://itunes.apple.com/search",
        params={"term": show, "media": "podcast", "entity": "podcast", "limit": limit},
        timeout=20,
    )
    r.raise_for_status()
    results = r.json().get("results", [])
    results.sort(key=lambda x: similarity(x.get("collectionName", ""), show), reverse=True)
    return [x["feedUrl"] for x in results if x.get("feedUrl")]


def fetch_feed(url: str) -> list[FeedEpisode]:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    return parse_feed(r.content)


def parse_feed(xml_bytes: bytes) -> list[FeedEpisode]:
    root = ET.fromstring(xml_bytes)
    episodes = []
    for item in root.iter("item"):
        enclosure = item.find("enclosure")
        if enclosure is None or not enclosure.get("url"):
            continue
        pub = item.findtext("pubDate")
        episodes.append(
            FeedEpisode(
                title=(item.findtext("title") or "").strip(),
                audio_url=enclosure.get("url"),
                pub_date=_parse_rfc822_date(pub),
                duration_s=parse_duration(item.findtext(f"{ITUNES_NS}duration")),
            )
        )
    return episodes


def match_episode(target: EpisodeInfo, candidates: list[FeedEpisode]) -> tuple[FeedEpisode, float] | None:
    """Pick the feed item that best matches the Spotify episode."""
    best, best_score = None, 0.0
    for ep in candidates:
        score = similarity(ep.title, target.title)
        if target.release_date and ep.pub_date:
            days = abs((ep.pub_date - target.release_date).days)
            score += 0.2 if days <= 1 else (0.1 if days <= 3 else -0.1)
        if target.duration_s and ep.duration_s:
            if abs(ep.duration_s - target.duration_s) / target.duration_s <= 0.05:
                score += 0.1
        if score > best_score:
            best, best_score = ep, score
    if best is None or best_score < MATCH_THRESHOLD:
        return None
    return best, best_score


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return " ".join(text.split())


def similarity(a: str, b: str) -> float:
    a, b = normalize(a), normalize(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ratio = SequenceMatcher(None, a, b).ratio()
    # Feeds often add prefixes like "#123 -" or "Ep. 5:", so reward containment.
    if a in b or b in a:
        ratio = max(ratio, 0.9)
    return ratio


def parse_duration(value: str | None) -> float | None:
    """Parse itunes:duration, which is either seconds or HH:MM:SS / MM:SS."""
    if not value:
        return None
    value = value.strip()
    try:
        if ":" in value:
            secs = 0.0
            for part in value.split(":"):
                secs = secs * 60 + float(part)
            return secs
        return float(value)
    except ValueError:
        return None


def _parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _parse_rfc822_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value.strip()).date()
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Download and transcription
# ---------------------------------------------------------------------------

def download(url: str, dest: Path) -> Path:
    with requests.get(url, headers={"User-Agent": USER_AGENT}, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                done += len(chunk)
                if total and sys.stderr.isatty():
                    print(f"\r  downloading… {done * 100 // total}%", end="", file=sys.stderr)
    if total and sys.stderr.isatty():
        print(file=sys.stderr)
    return dest


def transcribe(audio: Path, model_size: str, language: str | None, device: str) -> list[dict]:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("faster-whisper isn't installed. Run: pip install -r requirements.txt")

    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(str(audio), language=language, vad_filter=True)
    log(f"Detected language: {info.language} ({info.language_probability:.0%})")

    out = []
    for seg in segments:
        out.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
        if sys.stderr.isatty() and info.duration:
            print(f"\r  transcribing… {min(seg.end / info.duration, 1):.0%}", end="", file=sys.stderr)
    if sys.stderr.isatty():
        print(file=sys.stderr)
    return out


def format_timestamp(seconds: float, sep: str = ",") -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(segments: list[dict]) -> str:
    blocks = []
    for i, seg in enumerate(segments, 1):
        blocks.append(
            f"{i}\n{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n{seg['text']}\n"
        )
    return "\n".join(blocks)


def to_text(segments: list[dict], timestamps: bool) -> str:
    if timestamps:
        return "\n".join(f"[{format_timestamp(s['start'], '.')[:8]}] {s['text']}" for s in segments) + "\n"
    return "\n".join(s["text"] for s in segments) + "\n"


def write_outputs(segments: list[dict], base: Path, meta: dict) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        ".txt": to_text(segments, timestamps=True),
        ".srt": to_srt(segments),
        ".json": json.dumps({**meta, "segments": segments}, indent=2, ensure_ascii=False),
    }
    written = []
    for ext, content in paths.items():
        p = base.with_suffix(ext)
        p.write_text(content, encoding="utf-8")
        written.append(p)
    return written


def slugify(text: str, max_len: int = 80) -> str:
    return (re.sub(r"[^a-z0-9]+", "-", normalize(text)).strip("-") or "episode")[:max_len]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def resolve_audio_url(args: argparse.Namespace, info: EpisodeInfo) -> str:
    feeds = [args.feed] if args.feed else []
    if not feeds:
        if not info.show:
            sys.exit("Couldn't work out the show name. Pass it with --show \"Show Name\" or give --feed.")
        log(f"Searching Apple Podcasts for “{info.show}”…")
        feeds = find_feed_urls(info.show)
        if not feeds:
            sys.exit("No public RSS feed found. The show may be Spotify-exclusive; use --audio with your own file.")

    for feed in feeds:
        log(f"Checking feed {feed}")
        try:
            episodes = fetch_feed(feed)
        except (requests.RequestException, ET.ParseError) as e:
            log(f"  skipped ({e})")
            continue
        found = match_episode(info, episodes)
        if found:
            ep, score = found
            log(f"Matched “{ep.title}” ({ep.pub_date or 'no date'}, score {score:.2f})")
            return ep.audio_url

    sys.exit(
        "Couldn't find this episode in the show's public feed. It may be Spotify-exclusive, "
        "or the titles differ. Try --feed <rss-url>, --title, or --audio <file>."
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("episode", nargs="?", help="Spotify episode URL, URI or ID")
    p.add_argument("--audio", type=Path, help="transcribe this local audio file instead of downloading")
    p.add_argument("--feed", help="RSS feed URL of the show (skips the directory search)")
    p.add_argument("--show", help="override the show name used to find the feed")
    p.add_argument("--title", help="override the episode title used for matching")
    p.add_argument("--model", default="small", help="Whisper model: tiny, base, small, medium, large-v3 (default: small)")
    p.add_argument("--language", help="language code such as en or cs (default: auto-detect)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--out-dir", type=Path, default=Path("transcripts"))
    p.add_argument("--keep-audio", action="store_true", help="keep the downloaded audio next to the transcript")
    args = p.parse_args(argv)

    if not args.episode and not args.audio:
        p.error("give a Spotify episode link, or --audio <file>")

    info = EpisodeInfo(title=args.title or "")
    if args.episode:
        episode_id = parse_episode_id(args.episode)
        if not (args.audio or (args.title and (args.show or args.feed))):
            info = get_episode_info(episode_id)
            log(f"Episode: {info.title}" + (f" — {info.show}" if info.show else ""))
        info.title = args.title or info.title
        info.show = args.show or info.show
    if not info.title:
        info.title = args.audio.stem if args.audio else "episode"

    base = args.out_dir / slugify(info.title)
    with tempfile.TemporaryDirectory() as tmp:
        if args.audio:
            audio = args.audio
        else:
            url = resolve_audio_url(args, info)
            ext = Path(url.split("?")[0]).suffix or ".mp3"
            audio = Path(args.out_dir if args.keep_audio else tmp) / f"{slugify(info.title)}{ext}"
            audio.parent.mkdir(parents=True, exist_ok=True)
            log("Downloading audio…")
            download(url, audio)

        log(f"Transcribing with Whisper '{args.model}' (this can take a while on CPU)…")
        segments = transcribe(audio, args.model, args.language, args.device)

    meta = {
        "title": info.title,
        "show": info.show,
        "release_date": info.release_date.isoformat() if info.release_date else None,
        "source": args.episode or str(args.audio),
        "model": args.model,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    for path in write_outputs(segments, base, meta):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
