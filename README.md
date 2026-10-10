# spotify-transcribe

Turn a Spotify podcast link into a timestamped transcript.

```bash
python spotify_transcribe.py "https://open.spotify.com/episode/2tUttFvmQYPFrJqgw9VWLj"
```

Output goes to `transcripts/<episode-title>.{txt,srt,json}`:

- **.txt** – readable transcript, one line per segment with `[HH:MM:SS]` timestamps
- **.srt** – subtitles
- **.json** – metadata and every segment with start/end times

## How it works

Spotify streams podcast audio in a protected format, so the tool doesn't touch
Spotify's audio. Most podcasts also publish a public RSS feed with the
original audio file, and the tool uses that:

1. Reads the episode's title, show, date and length from Spotify.
2. Finds the show's RSS feed in the Apple Podcasts directory.
3. Matches the episode in the feed by title, date and length, then downloads
   the publisher's audio.
4. Transcribes it **locally** with [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
   Nothing gets uploaded to a transcription service.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads the Whisper model, which is 75 MB to 3 GB depending on `--model`.

### Optional: Spotify API credentials

Without credentials the tool reads the public episode page, which usually works.
For more reliable metadata, create a free app at
<https://developer.spotify.com/dashboard> and set:

```bash
export SPOTIFY_CLIENT_ID=...
export SPOTIFY_CLIENT_SECRET=...
```

## Options

| Flag | Purpose |
| --- | --- |
| `--model` | `tiny`, `base`, `small` (default), `medium`, `large-v3`. Larger models are more accurate and slower. |
| `--language` | Force a language such as `en` or `cs`. Auto-detected by default. |
| `--device` | `auto`, `cpu` or `cuda`. |
| `--show` / `--title` | Override the show name or episode title if the lookup gets them wrong. |
| `--feed` | Give the show's RSS feed URL directly and skip the directory search. |
| `--audio FILE` | Transcribe a local audio file you already have. |
| `--keep-audio` | Keep the downloaded audio in the output folder. |
| `--out-dir` | Output folder (default `transcripts/`). |

## Limitations

- **Spotify-exclusive shows** don't have a public feed, so the lookup fails for
  them. Use the transcript Spotify shows in its app, or `--audio` with a
  recording you have the right to use.
- On CPU, `small` runs at roughly real time to 3× real time. Use `--model base`
  for speed, or a CUDA GPU with `--model large-v3` for the best accuracy.
- Transcripts are for personal use. The podcast's copyright still applies.

## Development

```bash
pip install -r requirements-dev.txt
pytest
```
