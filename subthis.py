#!/usr/bin/env python3
"""Create short, accurately worded SRT captions from a video."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import urllib.error
import urllib.request
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Sequence


__version__ = "1.3.0"

API_URL = "https://api.openai.com/v1/audio/transcriptions"
KEY_CHECK_URL = "https://api.openai.com/v1/models/whisper-1"
QUOTA_PROBE_URL = "https://api.openai.com/v1/chat/completions"
QUOTA_PROBE_MODEL = "gpt-5-nano"
API_KEYS_URL = "https://platform.openai.com/api-keys"
BILLING_URL = "https://platform.openai.com/settings/organization/billing/overview"
SIGNUP_URL = "https://platform.openai.com/signup"


def _config_dir() -> Path:
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", "").strip()
        if appdata:
            return Path(appdata) / "subthis"
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "subthis"


CONFIG_DIR = _config_dir()
ENV_FILE = CONFIG_DIR / ".env"
TERMS_FILE = CONFIG_DIR / "terms.txt"
ACCURATE_MODEL = "gpt-transcribe"
TIMING_MODEL = "whisper-1"
MAX_UPLOAD_BYTES = 24 * 1024 * 1024
CHUNK_SECONDS = 60 * 60
CHUNK_OVERLAP_SECONDS = 1.5


DEFAULT_ALIASES: dict[str, list[str]] = {
    "OpenAI": ["OpenAI", "Open AI", "אופן איי איי", "אופן איי-איי", "אופן איי"],
    "Claude": ["Claude", "Clod", "קלוד", "קלאוד"],
    "ChatGPT": ["ChatGPT", "Chat GPT", "צ'אט ג'יפיטי", "צ׳אט ג׳יפיטי", "צ'ט ג'יפיטי"],
    "Codex": ["Codex", "קודקס"],
    "Anthropic": ["Anthropic", "אנתרופיק"],
    "Gemini": ["Gemini", "ג'מיני", "ג׳מיני"],
    "Cursor": ["Cursor", "קרסר", "קורסר"],
    "GitHub": ["GitHub", "Git Hub", "גיטהאב", "גיט האב"],
    "WordPress": ["WordPress", "Word Press", "וורדפרס"],
    "JavaScript": ["JavaScript", "Java Script", "ג'אווה סקריפט", "ג׳אווה סקריפט"],
    "TypeScript": ["TypeScript", "Type Script", "טייפסקריפט"],
    "Python": ["Python", "פייתון", "פייטון"],
    "Linux": ["Linux", "לינוקס"],
    "API": ["API", "A P I", "איי פי איי"],
    "CLI": ["CLI", "C L I", "סי אל איי"],
}

DEFAULT_KEYWORDS = [
    "OpenAI",
    "Claude",
    "ChatGPT",
    "Codex",
    "Anthropic",
    "Gemini",
    "Cursor",
    "GitHub",
    "API",
    "CLI",
]


class SubthisError(RuntimeError):
    """A safe, user-facing command error."""


@dataclasses.dataclass(frozen=True)
class TimedWord:
    text: str
    start: float
    end: float


@dataclasses.dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str


@dataclasses.dataclass(frozen=True)
class AudioChunk:
    path: Path
    offset: float
    duration: float


@dataclasses.dataclass(frozen=True)
class Config:
    api_key: str
    aliases: dict[str, list[str]]
    terms: list[str]
    languages: list[str]
    max_words: int


def _normalized_token(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text).casefold()
    return "".join(
        char
        for char in normalized
        if char.isalnum() and unicodedata.category(char) != "Mn"
    )


def canonicalize_terms(text: str, aliases: dict[str, list[str]]) -> str:
    replacements: list[tuple[str, str]] = []
    for canonical, spellings in aliases.items():
        for spelling in set([canonical, *spellings]):
            cleaned = spelling.strip()
            if cleaned:
                replacements.append((cleaned, canonical))
    replacements.sort(key=lambda item: len(item[0]), reverse=True)

    result = text
    for spelling, canonical in replacements:
        pattern = rf"(?<!\w){re.escape(spelling)}(?!\w)"
        result = re.sub(pattern, lambda _match: canonical, result, flags=re.IGNORECASE)
    return result


def strip_caption_punctuation(text: str) -> str:
    without_punctuation = "".join(
        char for char in text if not unicodedata.category(char).startswith("P")
    )
    return " ".join(without_punctuation.split())


def _distribute_words(tokens: Sequence[str], start: float, end: float) -> list[TimedWord]:
    if not tokens:
        return []
    start = max(0.0, start)
    end = max(start, end)
    width = (end - start) / len(tokens)
    return [
        TimedWord(token, start + index * width, start + (index + 1) * width)
        for index, token in enumerate(tokens)
    ]


def align_accurate_words(text: str, timed_words: Sequence[TimedWord]) -> list[TimedWord]:
    accurate_tokens = text.split()
    if not accurate_tokens:
        return []
    if not timed_words:
        raise SubthisError("The timing pass returned no words for a non-empty transcript.")

    accurate_normalized = [_normalized_token(token) for token in accurate_tokens]
    timing_normalized = [_normalized_token(word.text) for word in timed_words]
    matcher = difflib.SequenceMatcher(
        None,
        accurate_normalized,
        timing_normalized,
        autojunk=False,
    )
    aligned: list[TimedWord] = []

    for tag, a_start, a_end, w_start, w_end in matcher.get_opcodes():
        tokens = accurate_tokens[a_start:a_end]
        if tag == "equal":
            aligned.extend(
                TimedWord(token, source.start, source.end)
                for token, source in zip(tokens, timed_words[w_start:w_end])
            )
            continue
        if tag == "delete":
            continue
        if w_start < w_end:
            interval_start = timed_words[w_start].start
            interval_end = timed_words[w_end - 1].end
        else:
            interval_start = aligned[-1].end if aligned else timed_words[0].start
            interval_end = (
                timed_words[w_start].start
                if w_start < len(timed_words)
                else max(interval_start, timed_words[-1].end)
            )
        aligned.extend(_distribute_words(tokens, interval_start, interval_end))

    monotonic: list[TimedWord] = []
    previous_end = 0.0
    for word in aligned:
        start = max(previous_end, word.start)
        end = max(start, word.end)
        monotonic.append(TimedWord(word.text, start, end))
        previous_end = end
    return monotonic


def make_cues(words: Sequence[TimedWord], media_end: float, max_words: int = 3) -> list[Cue]:
    if not 1 <= max_words <= 3:
        raise ValueError("max_words must be between 1 and 3")
    clean_words = [
        TimedWord(cleaned, word.start, word.end)
        for word in words
        if (cleaned := strip_caption_punctuation(word.text))
    ]
    if not clean_words:
        return []

    groups = [
        list(clean_words[index : index + max_words])
        for index in range(0, len(clean_words), max_words)
    ]
    cues: list[Cue] = []
    for index, group in enumerate(groups):
        start = group[0].start
        if index + 1 < len(groups):
            end = groups[index + 1][0].start
        else:
            end = min(media_end, group[-1].end + 0.5)
        end = max(start + 0.001, end)
        cues.append(Cue(start, end, " ".join(word.text for word in group)))
    return cues


def _srt_time(seconds: float) -> str:
    total_ms = max(0, int(seconds * 1000 + 0.5))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def render_srt(cues: Sequence[Cue]) -> str:
    blocks = [
        f"{index}\n{_srt_time(cue.start)} --> {_srt_time(cue.end)}\n"
        f"{strip_caption_punctuation(cue.text)}"
        for index, cue in enumerate(cues, start=1)
    ]
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def merge_chunk_words(chunks: Sequence[Sequence[TimedWord]]) -> list[TimedWord]:
    merged: list[TimedWord] = []
    for chunk in chunks:
        current = list(chunk)
        if not current:
            continue
        overlap = 0
        maximum = min(30, len(merged), len(current))
        for size in range(maximum, 0, -1):
            left = [_normalized_token(word.text) for word in merged[-size:]]
            right = [_normalized_token(word.text) for word in current[:size]]
            if left == right:
                overlap = size
                break
        merged.extend(current[overlap:])
    return merged


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as error:
        raise SubthisError(f"Required command not found: {command[0]}") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip().splitlines()
        message = detail[-1] if detail else "unknown media error"
        raise SubthisError(f"{command[0]} failed: {message}") from error


def probe_duration(path: Path) -> float:
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    try:
        duration = float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SubthisError("Could not determine the video duration.") from error
    if duration <= 0:
        raise SubthisError("The video duration is zero.")
    return duration


def extract_chunks(video: Path, directory: Path) -> tuple[list[AudioChunk], float]:
    duration = probe_duration(video)
    chunks: list[AudioChunk] = []
    offset = 0.0
    index = 0
    while offset < duration:
        chunk_duration = min(CHUNK_SECONDS + CHUNK_OVERLAP_SECONDS, duration - offset)
        output = directory / f"chunk-{index:04d}.ogg"
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        if offset:
            command.extend(["-ss", f"{offset:.3f}"])
        command.extend(
            [
                "-i",
                str(video),
                "-t",
                f"{chunk_duration:.3f}",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "libopus",
                "-b:a",
                "32k",
                "-application",
                "voip",
                str(output),
            ]
        )
        _run(command)
        if not output.exists() or output.stat().st_size == 0:
            raise SubthisError("FFmpeg produced an empty audio file. The video may have no audio track.")
        if output.stat().st_size > MAX_UPLOAD_BYTES:
            raise SubthisError("An extracted audio chunk exceeds OpenAI's 25 MB upload limit.")
        chunks.append(AudioChunk(output, offset, chunk_duration))
        if offset + chunk_duration >= duration:
            break
        offset += CHUNK_SECONDS
        index += 1
    return chunks, duration


def _multipart(fields: Sequence[tuple[str, str]], file_path: Path) -> tuple[bytes, str]:
    boundary = f"subthis-{uuid.uuid4().hex}"
    body = bytearray()
    for name, value in fields:
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(value.encode("utf-8"))
        body.extend(b"\r\n")
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        (
            f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
            "Content-Type: audio/ogg\r\n\r\n"
        ).encode()
    )
    body.extend(file_path.read_bytes())
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def _api_error_message(payload: bytes, status: int | None = None) -> str:
    text = payload.decode("utf-8", errors="replace")
    if status in (401, 403) or "invalid_api_key" in text:
        return (
            "OpenAI no longer accepts your saved key (it may have been revoked). "
            f"Run 'subthis setup' to enter a new one from {API_KEYS_URL}"
        )
    if "insufficient_quota" in text:
        return (
            "Your OpenAI account is out of credit, so it cannot transcribe right now. "
            f"Add credit at {BILLING_URL} and run subthis again."
        )
    prefix = f"OpenAI API error{f' {status}' if status else ''}"
    try:
        message = json.loads(text).get("error", {}).get("message")
        if isinstance(message, str) and message.strip():
            result = f"{prefix}: {message.strip()}"
            if status == 429:
                result += " (OpenAI is asking us to slow down. Wait a minute and try again.)"
            return result
    except (json.JSONDecodeError, AttributeError):
        pass
    return prefix


def request_transcription(
    api_key: str,
    file_path: Path,
    fields: Sequence[tuple[str, str]],
) -> dict[str, object]:
    body, content_type = _multipart(fields, file_path)
    request = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": content_type,
            "User-Agent": "subthis/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            payload = response.read()
    except urllib.error.HTTPError as error:
        raise SubthisError(_api_error_message(error.read(), error.code)) from error
    except urllib.error.URLError as error:
        reason = getattr(error, "reason", "connection failed")
        raise SubthisError(f"Could not reach the OpenAI API: {reason}") from error
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as error:
        raise SubthisError("OpenAI returned an invalid transcription response.") from error
    if not isinstance(decoded, dict):
        raise SubthisError("OpenAI returned an unexpected transcription response.")
    return decoded


def transcribe_accurate(chunk: AudioChunk, config: Config) -> str:
    prompt = (
        "הקלטה בעברית עם מונחים מקצועיים באנגלית. יש לכתוב שמות חברות, "
        "מוצרים וכלים באיות המקורי באנגלית כאשר כך הם נאמרים."
    )
    fields: list[tuple[str, str]] = [
        ("model", ACCURATE_MODEL),
        ("response_format", "json"),
        ("prompt", prompt),
    ]
    fields.extend(("languages[]", language) for language in config.languages)
    fields.extend(("keywords[]", term) for term in config.terms)
    response = request_transcription(config.api_key, chunk.path, fields)
    text = response.get("text")
    if not isinstance(text, str):
        raise SubthisError("The accurate transcription response did not contain text.")
    canonical = canonicalize_terms(text.strip(), config.aliases)
    return strip_caption_punctuation(canonical)


def transcribe_timing(chunk: AudioChunk, config: Config) -> list[TimedWord]:
    timing_prompt = ", ".join(config.terms[:30])
    fields: list[tuple[str, str]] = [
        ("model", TIMING_MODEL),
        ("response_format", "verbose_json"),
        ("timestamp_granularities[]", "word"),
        ("language", config.languages[0] if config.languages else "he"),
    ]
    if timing_prompt:
        fields.append(("prompt", timing_prompt))
    response = request_transcription(config.api_key, chunk.path, fields)
    raw_words = response.get("words")
    if not isinstance(raw_words, list):
        raise SubthisError("The timing transcription response did not contain word timestamps.")
    words: list[TimedWord] = []
    for item in raw_words:
        if not isinstance(item, dict):
            continue
        text = item.get("word")
        start = item.get("start")
        end = item.get("end")
        if isinstance(text, str) and isinstance(start, (int, float)) and isinstance(end, (int, float)):
            words.append(TimedWord(text.strip(), float(start), float(end)))
    return words


def _load_api_key() -> str:
    environment_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if environment_key:
        return environment_key
    if ENV_FILE.is_file():
        with ENV_FILE.open(encoding="utf-8") as handle:
            for line in handle:
                match = re.match(r"\s*OPENAI_API_KEY\s*=\s*(.*?)\s*$", line)
                if match:
                    value = match.group(1).strip().strip("\"'")
                    if value:
                        return value
    raise SubthisError(
        f"No OPENAI_API_KEY found in the environment or {ENV_FILE}. Run: subthis setup"
    )


TERMS_TEMPLATE = """\
# Add one professional term per line so subthis sends it as a literal hint.
# Add spelling corrections with: Canonical name = alias one | alias two
#
# Examples:
# MyCompany
# DaVinci Resolve = דה וינצ'י ריזולב | Davinci Resolve
"""


def _ffmpeg_install_hint() -> str:
    if sys.platform == "win32":
        return "winget install ffmpeg   (or: choco install ffmpeg)"
    if sys.platform == "darwin":
        return "brew install ffmpeg"
    return "install it with your package manager, e.g. sudo pacman -S ffmpeg / sudo apt install ffmpeg"


_COLOR = False
_ANSI = {"green": "32", "red": "31", "yellow": "33", "cyan": "36", "bold": "1", "dim": "2"}


def _paint(text: str, *names: str) -> str:
    if not _COLOR:
        return text
    prefix = "".join(f"\033[{_ANSI[name]}m" for name in names)
    return f"{prefix}{text}\033[0m"


def _say_ok(text: str) -> None:
    print(_paint("  ✓ ", "green", "bold") + text)


def _say_bad(text: str) -> None:
    print(_paint("  ✗ ", "red", "bold") + text)


def _say_note(text: str) -> None:
    print(_paint("  ! ", "yellow", "bold") + text)


def _banner(title: str) -> None:
    line = "─" * (len(title) + 2)
    print(_paint(f"╭{line}╮\n│ {title} │\n╰{line}╯", "cyan"))


def _ask(prompt: str) -> str:
    try:
        return input(_paint(prompt, "bold")).strip()
    except EOFError:
        return ""


def _open_page(url: str, interactive: bool) -> None:
    print("    " + _paint(url, "cyan", "bold"))
    if interactive:
        opened = False
        with contextlib.suppress(Exception):
            opened = webbrowser.open(url)
        if opened:
            print(_paint("    (this page should now be open in your browser)", "dim"))
        else:
            print(_paint("    (copy that address into your browser)", "dim"))


def _classify_key(api_key: str) -> tuple[str, str]:
    """Test a key against OpenAI. Returns one of: ok, invalid, no_credit, unreachable."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": f"subthis/{__version__}",
    }
    request = urllib.request.Request(KEY_CHECK_URL, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30):
            pass
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            return "invalid", ""
        return "unreachable", f"HTTP {error.code}"
    except urllib.error.URLError as error:
        return "unreachable", str(getattr(error, "reason", "connection failed"))

    # The key is real. Now spend a fraction of a cent on the smallest possible
    # request, because only a real request reveals an account with no credit.
    probe = json.dumps(
        {
            "model": QUOTA_PROBE_MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 1,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        QUOTA_PROBE_URL,
        data=probe,
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30):
            pass
    except urllib.error.HTTPError as error:
        text = error.read().decode("utf-8", errors="replace")
        if "insufficient_quota" in text:
            return "no_credit", ""
        if error.code in (401, 403) and "invalid_api_key" in text:
            return "invalid", ""
        # Any other refusal (unknown probe model, key limited to specific
        # models, rate limit) says nothing bad about the key itself.
    except urllib.error.URLError:
        pass
    return "ok", ""


def _guide_first_timer(interactive: bool) -> None:
    print("\nNo problem. There are three short steps, all on OpenAI's website.\n")

    print(_paint("Step 1 of 3: create an OpenAI account", "bold"))
    print("  If you already log in to ChatGPT, use that same account and skip ahead.")
    print("  Sign up or log in here:")
    _open_page(SIGNUP_URL, interactive)
    if interactive:
        _ask("  Press Enter when you are logged in... ")

    print("\n" + _paint("Step 2 of 3: add 5 dollars of credit", "bold"))
    print(
        "  The transcription service is prepaid, like a phone card. 5 dollars is\n"
        "  the minimum and covers roughly five hours of video. On the page below:\n"
        "    1. Click 'Add payment method' and enter your card details.\n"
        "    2. Click 'Add to credit balance' and choose 5 dollars.\n"
        "    3. If it offers 'auto-reload' (topping up automatically), you can\n"
        "       switch that off.\n"
        "  The page:"
    )
    _open_page(BILLING_URL, interactive)
    if interactive:
        _ask("  Press Enter when your balance shows the credit... ")

    print("\n" + _paint("Step 3 of 3: create your key", "bold"))
    print(
        "  On the page below, click 'Create new secret key', type any name you\n"
        "  like (for example: subthis), and click 'Create secret key'. Then click\n"
        "  'Copy'. Important: the key is shown only this once, so copy it now.\n"
        "  The page:"
    )
    _open_page(API_KEYS_URL, interactive)


def _prompt_for_working_key(existing_key: str, interactive: bool) -> str:
    attempts = 5 if interactive else 1
    for _ in range(attempts):
        if existing_key:
            entered = _ask(
                "\nPaste your OpenAI key and press Enter"
                " (or press Enter alone to keep the saved one): "
            )
        else:
            entered = _ask("\nPaste your OpenAI key here and press Enter: ")
        api_key = (entered or existing_key).strip().strip("\"'")
        if not api_key:
            _say_bad(f"Nothing was entered. Your key is waiting at {API_KEYS_URL}")
            continue
        print("  Checking your key with OpenAI...")
        status, detail = _classify_key(api_key)
        if status == "ok":
            _say_ok("The key works and your account has credit.")
            return api_key
        if status == "invalid":
            _say_bad(
                "OpenAI does not accept this key. It may be mistyped, expired,\n"
                "    or revoked. Copy a fresh one from:"
            )
            _open_page(API_KEYS_URL, interactive)
            existing_key = ""
            continue
        if status == "no_credit":
            _say_bad("The key itself works, but the account behind it has no credit yet.")
            print("    Add at least 5 dollars here:")
            _open_page(BILLING_URL, interactive)
            if not interactive:
                raise SubthisError(f"The OpenAI account has no credit. Add credit at {BILLING_URL}")
            _ask("    Press Enter after adding credit and it will be checked again... ")
            existing_key = api_key
            continue
        _say_note(
            f"Could not reach OpenAI to test the key ({detail}).\n"
            "    Saving it anyway. subthis will say so clearly if it turns out not to work."
        )
        return api_key
    raise SubthisError("No working OpenAI key was entered. Run 'subthis setup' to try again.")


def _write_env_file(api_key: str) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        ENV_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(f"OPENAI_API_KEY={api_key}\n")
    with contextlib.suppress(OSError):
        os.chmod(ENV_FILE, 0o600)


def _example_command() -> str:
    if sys.platform == "win32":
        return r"subthis C:\Users\you\Videos\lesson.mp4"
    if sys.platform == "darwin":
        return "subthis ~/Movies/lesson.mp4"
    return "subthis ~/Videos/lesson.mp4"


def run_setup() -> int:
    global _COLOR
    interactive = sys.stdin is not None and sys.stdin.isatty()
    _COLOR = (
        not os.environ.get("NO_COLOR")
        and sys.stdout is not None
        and sys.stdout.isatty()
    )
    if _COLOR and sys.platform == "win32":
        os.system("")  # flips older Windows consoles into ANSI color mode

    _banner(f"Welcome to subthis  ·  v{__version__}")
    print(
        "\nsubthis turns a video into subtitles. This one-time setup takes about\n"
        "two minutes. Nothing on your computer is changed except subthis's own\n"
        f"settings folder: {CONFIG_DIR}\n"
    )

    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        _say_ok("ffmpeg is installed (subthis uses it to read the sound from your videos)")
    else:
        _say_note("A helper program called ffmpeg is missing. It reads the sound from videos.")
        print(f"    To install it: {_ffmpeg_install_hint()}")
        print("    Then run 'subthis setup' again.")

    print(
        "\nsubthis sends your video's audio to OpenAI (the company behind ChatGPT)\n"
        "to turn the speech into text. For that you need your own OpenAI key:\n"
        "a long code starting with sk- that acts like a password. It is saved\n"
        "only on this computer.\n"
    )

    existing_key = ""
    with contextlib.suppress(SubthisError):
        existing_key = _load_api_key()
    if existing_key:
        _say_ok("A key is already saved. Press Enter at the prompt below to keep it.")
    else:
        answer = _ask("Have you used the OpenAI API before? Type y (yes) or n (no): ").lower()
        if answer.startswith("n"):
            _guide_first_timer(interactive)
        else:
            print("\nCopy a key from OpenAI's key page:")
            _open_page(API_KEYS_URL, interactive)

    api_key = _prompt_for_working_key(existing_key, interactive)
    _write_env_file(api_key)
    _say_ok(f"Key saved to {ENV_FILE}")

    if not TERMS_FILE.exists():
        TERMS_FILE.write_text(TERMS_TEMPLATE, encoding="utf-8")
        _say_ok(f"Created a terms file at {TERMS_FILE}")
        print("    (optional: list names and jargon there and subthis will spell them right)")
    else:
        _say_ok(f"Keeping your existing terms file at {TERMS_FILE}")

    if shutil.which("subthis") is None:
        _say_note(
            "The 'subthis' command is not reachable from new terminals yet. To fix:\n"
            "    installed with uv:   run  uv tool update-shell\n"
            "    installed with pipx: run  pipx ensurepath\n"
            "    then open a new terminal window."
        )

    print()
    _banner("You're all set!")
    print(
        "\nTo make subtitles, type subthis, a space, and then your video file:\n"
        f"    {_paint(_example_command(), 'bold')}\n"
        "Tip: type \"subthis \" and drag the video from your files into this\n"
        "window, then press Enter. The subtitles are saved as an .srt file\n"
        "next to the video.\n"
    )
    return 0


def load_terms(path: Path, additions: Iterable[str] = ()) -> tuple[dict[str, list[str]], list[str]]:
    aliases = {canonical: list(spellings) for canonical, spellings in DEFAULT_ALIASES.items()}
    keywords = list(DEFAULT_KEYWORDS)
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            lines = list(handle)
    else:
        lines = []
    lines.extend(additions)
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        canonical, separator, raw_aliases = line.partition("=")
        canonical = canonical.strip()
        if not canonical or any(char in canonical for char in "<>\r\n"):
            continue
        extra = [part.strip() for part in raw_aliases.split("|") if part.strip()] if separator else []
        aliases.setdefault(canonical, [])
        aliases[canonical].extend(extra)
        if canonical not in keywords:
            keywords.append(canonical)
    return aliases, keywords


def transcribe_video(video: Path, config: Config) -> tuple[list[Cue], float]:
    with tempfile.TemporaryDirectory(prefix="subthis-") as temporary:
        chunks, duration = extract_chunks(video, Path(temporary))
        all_words: list[list[TimedWord]] = []
        for index, chunk in enumerate(chunks, start=1):
            print(f"Transcribing chunk {index}/{len(chunks)}...", file=sys.stderr)
            with ThreadPoolExecutor(max_workers=2) as pool:
                accurate_future = pool.submit(transcribe_accurate, chunk, config)
                timing_future = pool.submit(transcribe_timing, chunk, config)
                accurate_text = accurate_future.result()
                timing_words = timing_future.result()
            if not accurate_text and not timing_words:
                continue
            if not accurate_text:
                raise SubthisError("The accurate transcription was empty while speech timing was detected.")
            aligned = align_accurate_words(accurate_text, timing_words)
            all_words.append(
                [
                    TimedWord(word.text, word.start + chunk.offset, word.end + chunk.offset)
                    for word in aligned
                ]
            )
        merged = merge_chunk_words(all_words)
        return make_cues(merged, duration, config.max_words), duration


def _help_epilog() -> str:
    if sys.platform == "win32":
        open_terminal = (
            "press the Windows key, type powershell, press Enter to open a terminal"
        )
        example = r"subthis C:\Users\you\Videos\lesson.mp4"
        drag_source = "File Explorer"
    elif sys.platform == "darwin":
        open_terminal = (
            "press Cmd+Space, type terminal, press Enter to open a terminal"
        )
        example = "subthis ~/Movies/lesson.mp4"
        drag_source = "Finder"
    else:
        open_terminal = "open your terminal application"
        example = "subthis ~/Videos/lesson.mp4"
        drag_source = "your file manager"
    return f"""
quick start (first time):
  1. {open_terminal}
  2. type: subthis setup
     (one-time: stores your OpenAI API key)
  3. type: subthis followed by a space, then your video file
     for example: {example}

tip: instead of typing the file location, type "subthis " and drag the
video from {drag_source} into this window, then press Enter.

the captions are saved as an .srt file next to your video.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subthis",
        description="Create accurate Hebrew/English SRT captions with at most three words per cue.",
        epilog=_help_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"subthis {__version__}")
    parser.add_argument("video", type=Path, help="video or audio file to transcribe")
    parser.add_argument("-o", "--output", type=Path, help="output SRT path; defaults beside the input")
    parser.add_argument(
        "--max-words",
        type=int,
        choices=(1, 2, 3),
        default=3,
        help="maximum words per subtitle (default: 3)",
    )
    parser.add_argument("--term", action="append", default=[], help="extra canonical term; repeat as needed")
    parser.add_argument(
        "--terms-file",
        type=Path,
        default=TERMS_FILE,
        help=f"term list (default: {TERMS_FILE})",
    )
    parser.add_argument(
        "--language",
        action="append",
        dest="languages",
        help="expected ISO-639-1 language; repeat for code-switching (default: he, en)",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing subtitle file")
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] == "help":
        build_parser().print_help()
        return 0
    if arguments[0] == "setup":
        if len(arguments) > 1:
            raise SubthisError("setup takes no further arguments.")
        return run_setup()
    args = build_parser().parse_args(arguments)
    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise SubthisError(f"Input file not found: {video}")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise SubthisError("subthis requires both ffmpeg and ffprobe.")
    output = (args.output or video.with_suffix(".srt")).expanduser().resolve()
    if output == video:
        raise SubthisError("The output path must differ from the input path.")
    if output.exists() and not args.force:
        raise SubthisError(f"Output already exists: {output}. Use --force to replace it.")
    aliases, terms = load_terms(args.terms_file.expanduser(), args.term)
    config = Config(
        api_key=_load_api_key(),
        aliases=aliases,
        terms=terms,
        languages=args.languages or ["he", "en"],
        max_words=args.max_words,
    )
    cues, _duration = transcribe_video(video, config)
    if not cues:
        raise SubthisError("No speech was detected, so no subtitle file was written.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.subthis-{os.getpid()}.tmp")
    try:
        temporary_output.write_text(render_srt(cues), encoding="utf-8")
        os.replace(temporary_output, output)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary_output.unlink()
    print(f"Wrote {len(cues)} subtitles to {output}")
    return 0


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.reconfigure(errors="replace")
    try:
        return run()
    except SubthisError as error:
        print(f"subthis: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("subthis: cancelled", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
