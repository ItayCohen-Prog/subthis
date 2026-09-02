# subthis

Turns a video (or audio file) into an SRT subtitle file with short, accurately worded captions, at most three words per cue. Built for Hebrew with English tech terms mixed in: it transcribes twice through the OpenAI API (`gpt-transcribe` for wording, `whisper-1` for word timing), aligns the two, and canonicalizes product names like OpenAI, Claude and ChatGPT.

Works on Linux, macOS and Windows. No Python dependencies outside the standard library.

## Requirements

- [ffmpeg](https://ffmpeg.org/) (`ffmpeg` and `ffprobe` on PATH). `subthis setup` tells you how to install it if it's missing.
- An OpenAI API key.

## Install

With [uv](https://docs.astral.sh/uv/) (installs Python too if needed):

```sh
uv tool install /path/to/subthis
```

Or with pipx:

```sh
pipx install /path/to/subthis
```

Both put a `subthis` command on your PATH on any OS. To install from a git remote instead: `uv tool install git+<repo-url>`.

## Setup

```sh
subthis setup
```

Checks for ffmpeg, asks for your OpenAI API key (input hidden), verifies it against the API, and stores it in the config directory:

- Linux/macOS: `~/.config/subthis/`
- Windows: `%APPDATA%\subthis\`

It also creates `terms.txt` there, where you can add professional terms and spelling corrections:

```
MyCompany
DaVinci Resolve = דה וינצ'י ריזולב | Davinci Resolve
```

`OPENAI_API_KEY` in the environment overrides the stored key.

## Use

```sh
subthis lecture.mp4                 # writes lecture.srt next to the input
subthis talk.mp4 -o subs/talk.srt   # explicit output path
subthis clip.mp4 --max-words 2      # shorter cues
subthis clip.mp4 --term "Omarchy"   # one-off extra term
subthis clip.mp4 --language he --language en
```

Long videos are chunked and merged automatically; uploads stay under OpenAI's 25 MB limit.

## Tests

```sh
python -m unittest discover -s tests
```
