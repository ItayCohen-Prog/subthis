# subthis documentation

subthis turns a video (or audio file) into an SRT subtitle file with short, accurately worded captions, at most three words per cue. Built for Hebrew speech with English names and jargon mixed in. Open source (MIT), Python with no dependencies outside the standard library.

- Website: https://subthis.webivize.com
- Source: https://github.com/ItayCohen-Prog/subthis
- Package: https://pypi.org/project/subthis/

## 1. Install

subthis needs uv (installs subthis and Python itself) and ffmpeg (reads audio from video).

Windows (PowerShell):

    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    winget install --id Gyan.FFmpeg -e
    uv tool install subthis

macOS (Terminal; install Homebrew from https://brew.sh first if missing):

    curl -LsSf https://astral.sh/uv/install.sh | sh
    brew install ffmpeg
    uv tool install subthis

Linux:

    curl -LsSf https://astral.sh/uv/install.sh | sh
    sudo apt install ffmpeg   # or dnf / pacman
    uv tool install subthis

Open a new terminal after installing uv. If `subthis` is not found afterwards, run `uv tool update-shell` (or `pipx ensurepath`) and open a new terminal.

## 2. Setup

    subthis setup

A guided one-time wizard: checks ffmpeg (with per-OS install hints), explains each step before opening the browser (pressing Enter opens OpenAI's key page; the site handles account signup itself), tests the pasted key immediately (invalid/revoked keys re-prompt; an account without credit gets click-by-click guidance for adding the 5-dollar minimum of prepaid credit and a re-check), and saves the key to the config directory (`~/.config/subthis` on Linux/macOS, `%APPDATA%\subthis` on Windows). An `OPENAI_API_KEY` environment variable overrides the stored key.

## 3. Make subtitles

    subthis lecture.mp4                          # writes lecture.srt next to the video
    subthis talk.mp4 -o subs/talk.srt            # choose the output path
    subthis clip.mp4 --max-words 2               # shorter cues (1, 2 or 3)
    subthis clip.mp4 --language he --language en # expected languages (default: he, en)
    subthis clip.mp4 --force                     # overwrite an existing .srt

The full path of the finished file is printed at the end. Long videos are chunked, transcribed, and merged automatically. Each chunk is transcribed twice via the OpenAI API: gpt-transcribe for accurate wording and whisper-1 for word-level timing; the two are aligned so accurate words carry precise timestamps.

## 4. Teach it terms

Terms are names and jargon subthis should spell correctly. Syntax everywhere: spaces separate terms, single quotes group a multi-word term.

    subthis clip.mp4 --term "Omarchy 'Wispr Flow'"   # this run only
    subthis config terms "Omarchy 'Wispr Flow'"      # saved globally
    subthis config terms                             # interactive editor

The interactive editor: arrows move, space selects, `m` marks/unmarks all, `r` removes selected (with confirmation), `a` adds, `q` saves and quits.

Per-folder terms: put a `subthis-terms.txt` file next to your videos; it is picked up automatically. One term per line; spelling corrections use `Canonical = alias one | alias two`.

## 5. Config

| Command | Effect |
|---|---|
| `subthis config key` | Replace the saved OpenAI key; verified (validity + credit) before saving. Keys are never accepted as CLI arguments. |
| `subthis config terms` | Interactive terms editor; with a quoted argument, append directly. |
| `subthis config open on` / `off` | Also open the file manager with the finished file selected. |
| `subthis config captions` | View and tune caption behavior (saved for all future videos; defaults follow industry standards): `words` per line (1-3), `pause` / `hang` / `min` / `gap` in seconds (gap defaults to 0), `punctuation keep\|remove`, `silence hold\|cut`, `reset`. Example: `subthis config captions words 2`. |

## 6. Updates

At the start of an interactive run subthis checks PyPI (3-second budget, silent offline). If newer: it asks once; yes updates and reruns the same command, no prints `uv tool upgrade subthis` for later. On Windows the update command is always printed for a fresh window, since a running program cannot replace itself.

## 7. Troubleshooting

- "subthis: command not found" after install: run `uv tool update-shell` (or `pipx ensurepath`), open a new terminal.
- "OpenAI no longer accepts your saved key": the key was revoked; run `subthis config key` with a fresh key from https://platform.openai.com/api-keys
- "Your OpenAI account is out of credit": add credit (5 dollar minimum) at https://platform.openai.com/settings/organization/billing/overview
- "Required command not found: ffmpeg": install ffmpeg for your OS (section 1).
- "The video may have no audio track": the input has no sound.

## 8. CLI reference

| Command / flag | Meaning |
|---|---|
| `subthis <video>` | Transcribe to `<video>.srt` |
| `subthis setup` | Guided one-time setup |
| `subthis config key\|terms\|open` | Change key / edit terms / toggle reveal |
| `subthis help` or bare `subthis` | Full help with OS-specific quick start |
| `subthis docs` | Open the documentation in a browser |
| `-o, --output PATH` | Output SRT path (default: next to input) |
| `--max-words {1,2,3}` | Max words per cue (default 3) |
| `--term "A 'B C'"` | Extra terms this run; single quotes group; repeatable |
| `--terms-file PATH` | Alternative global terms file |
| `--language XX` | Expected ISO-639-1 language; repeatable (default he, en) |
| `--force` | Overwrite existing subtitle file |
| `--version` | Print version |

Exit codes: 0 success; 1 handled error (stderr line prefixed `subthis:`); 130 cancelled (Ctrl+C).

## 9. For AI agents

subthis is non-interactive-safe: piped stdin disables all prompts, colors, and browser openings; the update check is skipped without a TTY and can be disabled with `SUBTHIS_SKIP_UPDATE=1`. Minimal flow:

    uv tool install subthis
    OPENAI_API_KEY=sk-... subthis input.mp4 -o output.srt --force
    # stdout ends with the absolute path of the written file
