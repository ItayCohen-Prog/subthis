# subthis

Website and docs: https://subthis.webivize.com (English and Hebrew)

Turns a video (or audio file) into an SRT subtitle file with short, accurately worded captions, at most three words per cue. Built for Hebrew with English tech terms mixed in: it transcribes twice through the OpenAI API (`gpt-transcribe` for wording, `whisper-1` for word timing), aligns the two, and canonicalizes product names like OpenAI, Claude and ChatGPT.

Works on Linux, macOS and Windows. No Python dependencies outside the standard library.

## Requirements

- [ffmpeg](https://ffmpeg.org/) (`ffmpeg` and `ffprobe` on PATH). `subthis setup` tells you how to install it if it's missing.
- An OpenAI API key.

## Install

With [uv](https://docs.astral.sh/uv/) (installs Python too if needed):

```sh
uv tool install subthis
```

Or with pipx:

```sh
pipx install subthis
```

Both pull [subthis from PyPI](https://pypi.org/project/subthis/) and put a `subthis` command on your PATH on any OS. To try it without installing: `uvx subthis video.mp4`.

If the `subthis` command isn't found after installing, run `uv tool update-shell` (uv) or `pipx ensurepath` (pipx) and open a new terminal.

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
subthis clip.mp4 --language he --language en
```

## Teaching it your terms

Terms are names and jargon subthis should spell correctly. Spaces separate terms; wrap a multi-word term in single quotes:

```sh
subthis clip.mp4 --term "Omarchy 'Wispr Flow'"   # this run only
subthis config terms "Omarchy 'Wispr Flow'"      # every video, saved globally
subthis config terms                             # interactive editor
```

Bare `subthis config terms` opens a keyboard-driven editor of your saved terms: arrows move, space selects, `m` marks or unmarks all, `r` removes the selected terms (with confirmation), `a` adds new ones, `q` saves and quits. The key guide stays at the bottom of the screen.

For a folder of related videos, put a `subthis-terms.txt` file next to them (one term per line, or `Canonical = alias one | alias two` for spelling corrections); it is picked up automatically for any video in that folder.

## Changing your key

```sh
subthis config key
```

Prompts for a new OpenAI key and verifies it (validity and account credit) before saving, same as setup.

## Tuning the captions

```sh
subthis config captions            # see every setting, its value, and what it does
subthis config captions words 2    # example: two words per line from now on
subthis config captions reset      # back to the defaults
```

Settings (defaults follow subtitle-industry standards): `words` per line (1-3), `pause` (a silence this long starts a new line), `hang` (how long a line outlives its last word), `min` (minimum time on screen), `gap` (empty space before the next line, default 0; all in seconds), `punctuation keep|remove`, and `silence hold|cut` (hold keeps a line on screen through silences). Saved values apply to every future video; `--max-words` still overrides `words` for a single run.

## Finding your finished file

When subthis finishes it prints the full path of the subtitle file. With

```sh
subthis config open on
```

it also opens your file manager with that file selected (Explorer on Windows, Finder on macOS, and the FileManager1 interface on Linux desktops, falling back to opening the folder).

## Updates

When a newer version is on PyPI, subthis offers to update at the start of a run and then continues exactly where you were. Declining prints the command to update later.

Long videos are chunked and merged automatically; uploads stay under OpenAI's 25 MB limit.

## Tests

```sh
python -m unittest discover -s tests
```
