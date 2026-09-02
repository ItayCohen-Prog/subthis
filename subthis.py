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
import shlex
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Sequence


__version__ = "1.6.0"

API_URL = "https://api.openai.com/v1/audio/transcriptions"
KEY_CHECK_URL = "https://api.openai.com/v1/models/whisper-1"
QUOTA_PROBE_URL = "https://api.openai.com/v1/chat/completions"
QUOTA_PROBE_MODEL = "gpt-5-nano"
API_KEYS_URL = "https://platform.openai.com/api-keys"
BILLING_URL = "https://platform.openai.com/settings/organization/billing/overview"
SIGNUP_URL = "https://platform.openai.com/signup"
PYPI_JSON_URL = "https://pypi.org/pypi/subthis/json"
DOCS_URL = "https://subthis.webivize.com/docs/"
PROJECT_TERMS_FILENAME = "subthis-terms.txt"


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
SETTINGS_FILE = CONFIG_DIR / "settings.json"
ACCURATE_MODEL = "gpt-transcribe"
TIMING_MODEL = "whisper-1"
MAX_UPLOAD_BYTES = 24 * 1024 * 1024
CHUNK_SECONDS = 60 * 60
CHUNK_OVERLAP_SECONDS = 1.5
PAUSE_SPLIT_SECONDS = 0.4   # a silence this long starts a new phrase
CUE_HANG_SECONDS = 0.5      # how long a cue may outlive its last word
CUE_GAP_SECONDS = 0.08      # minimum gap kept before the next cue
MIN_CUE_SECONDS = 0.83      # Netflix floor: 5/6 of a second on screen


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
    # Apostrophes and the Hebrew geresh stay when they sit inside a word:
    # stripping them turns ג'מיני into גמיני and don't into dont.
    characters = list(text)
    kept: list[str] = []
    for index, char in enumerate(characters):
        if not unicodedata.category(char).startswith("P"):
            kept.append(char)
            continue
        if char in "'׳’":
            before = characters[index - 1] if index else ""
            after = characters[index + 1] if index + 1 < len(characters) else ""
            if before.isalpha() and after.isalpha():
                kept.append(char)
    return " ".join("".join(kept).split())


def canonicalize_timed_words(
    words: Sequence[TimedWord], aliases: dict[str, list[str]]
) -> list[TimedWord]:
    """Replace alias runs in the timing stream with their canonical term.

    The timing model hears "אופן איי איי" where the accurate model writes
    "OpenAI"; folding both to the canonical spelling turns exactly the words
    the user cares about into alignment anchors instead of mismatches.
    """
    lookup: dict[tuple[str, ...], str] = {}
    for canonical, spellings in aliases.items():
        for spelling in {canonical, *spellings}:
            key = tuple(
                token for token in (_normalized_token(part) for part in spelling.split()) if token
            )
            if key:
                lookup.setdefault(key, canonical)
    if not lookup:
        return list(words)
    longest = max(len(key) for key in lookup)
    normalized = [_normalized_token(word.text) for word in words]
    result: list[TimedWord] = []
    index = 0
    while index < len(words):
        for size in range(min(longest, len(words) - index), 0, -1):
            canonical = lookup.get(tuple(normalized[index : index + size]))
            if canonical:
                result.append(
                    TimedWord(canonical, words[index].start, words[index + size - 1].end)
                )
                index += size
                break
        else:
            result.append(words[index])
            index += 1
    return result


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
        if tag == "replace" and (a_end - a_start) == (w_end - w_start):
            # Same number of words on both sides (a phonetic respelling, a
            # number written differently): keep each timing word's real span
            # instead of smearing the block evenly.
            aligned.extend(
                TimedWord(token, source.start, source.end)
                for token, source in zip(tokens, timed_words[w_start:w_end])
            )
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


def _balanced_groups(words: Sequence[TimedWord], max_words: int) -> list[list[TimedWord]]:
    count = -(-len(words) // max_words)
    base, extra = divmod(len(words), count)
    groups: list[list[TimedWord]] = []
    index = 0
    for position in range(count):
        size = base + (1 if position < extra else 0)
        groups.append(list(words[index : index + size]))
        index += size
    return groups


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

    # Split into phrases at real pauses, then balance each phrase into groups
    # (7 words become 3+2+2, never 3+3+1) so no orphan cue trails a sentence.
    phrases: list[list[TimedWord]] = [[clean_words[0]]]
    for previous, word in zip(clean_words, clean_words[1:]):
        if word.start - previous.end > PAUSE_SPLIT_SECONDS:
            phrases.append([word])
        else:
            phrases[-1].append(word)
    groups = [group for phrase in phrases for group in _balanced_groups(phrase, max_words)]

    cues: list[Cue] = []
    for index, group in enumerate(groups):
        start = group[0].start
        next_start = groups[index + 1][0].start if index + 1 < len(groups) else media_end
        latest_allowed = max(next_start - CUE_GAP_SECONDS, start + 0.001)
        end = min(latest_allowed, group[-1].end + CUE_HANG_SECONDS, media_end)
        end = max(end, start + 0.001)
        if end - start < MIN_CUE_SECONDS:
            end = max(end, min(start + MIN_CUE_SECONDS, latest_allowed, media_end))
            end = max(end, start + 0.001)
        cues.append(Cue(start, end, " ".join(word.text for word in group)))
    return cues


def _srt_time(seconds: float) -> str:
    total_ms = max(0, int(seconds * 1000 + 0.5))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def _has_rtl_text(text: str) -> bool:
    return any(unicodedata.bidirectional(char) in ("R", "AL") for char in text)


def render_srt(cues: Sequence[Cue]) -> str:
    blocks = []
    for index, cue in enumerate(cues, start=1):
        line = strip_caption_punctuation(cue.text)
        if _has_rtl_text(line):
            # SRT carries no direction metadata; a leading RLM keeps a cue
            # that starts with an English word in Hebrew reading order.
            line = "‏" + line
        blocks.append(f"{index}\n{_srt_time(cue.start)} --> {_srt_time(cue.end)}\n{line}")
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


def _non_speech_spans(segments: object) -> list[tuple[float, float]]:
    """Spans whisper itself marks as probably-not-speech (hallucination guard)."""
    spans: list[tuple[float, float]] = []
    if not isinstance(segments, list):
        return spans
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        no_speech = segment.get("no_speech_prob")
        logprob = segment.get("avg_logprob")
        start = segment.get("start")
        end = segment.get("end")
        if (
            isinstance(no_speech, (int, float))
            and isinstance(logprob, (int, float))
            and isinstance(start, (int, float))
            and isinstance(end, (int, float))
            and no_speech > 0.6
            and logprob < -1.0
        ):
            spans.append((float(start), float(end)))
    return spans


def filter_non_speech_words(
    words: Sequence[TimedWord], segments: object
) -> list[TimedWord]:
    spans = _non_speech_spans(segments)
    if not spans:
        return list(words)
    return [
        word
        for word in words
        if not any(start <= (word.start + word.end) / 2 <= end for start, end in spans)
    ]


def transcribe_timing(chunk: AudioChunk, config: Config) -> list[TimedWord]:
    timing_prompt = ", ".join(config.terms[:30])
    fields: list[tuple[str, str]] = [
        ("model", TIMING_MODEL),
        ("response_format", "verbose_json"),
        ("timestamp_granularities[]", "word"),
        ("timestamp_granularities[]", "segment"),
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
    return filter_non_speech_words(words, response.get("segments"))


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


def _init_color() -> None:
    global _COLOR
    _COLOR = (
        not os.environ.get("NO_COLOR")
        and sys.stdout is not None
        and hasattr(sys.stdout, "isatty")
        and sys.stdout.isatty()
    )
    if _COLOR and sys.platform == "win32":
        os.system("")  # flips older Windows consoles into ANSI color mode


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", text)[:3])


def _latest_pypi_version() -> str | None:
    request = urllib.request.Request(
        PYPI_JSON_URL, headers={"User-Agent": f"subthis/{__version__}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            version = json.loads(response.read()).get("info", {}).get("version")
        return version if isinstance(version, str) else None
    except Exception:
        return None


def _update_command() -> str:
    executable = str(Path(sys.executable).resolve()).lower()
    if "pipx" in executable:
        return "pipx upgrade subthis"
    return "uv tool upgrade subthis"


def _maybe_offer_update(arguments: Sequence[str]) -> None:
    if os.environ.get("SUBTHIS_SKIP_UPDATE") == "1":
        return
    if sys.stdin is None or not sys.stdin.isatty():
        return
    if sys.stdout is None or not sys.stdout.isatty():
        return
    latest = _latest_pypi_version()
    if not latest or _version_tuple(latest) <= _version_tuple(__version__):
        return
    print(
        _paint(f"A new version of subthis is out: {latest}", "cyan", "bold")
        + f" (you have {__version__})."
    )
    answer = _ask("Update now? It takes a few seconds. Type y (yes) or n (no): ").lower()
    if not answer.startswith("y"):
        print("No problem. When you want it later, copy and run this command:")
        print("    " + _paint(_update_command(), "bold") + "\n")
        return
    print("Updating...")
    result = subprocess.run(
        _update_command().split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    fresh = shutil.which("subthis")
    if result.returncode != 0 or fresh is None:
        _say_note("The update did not finish, so this run continues on the current version.")
        return
    _say_ok("Updated. Continuing right where you were...\n")
    environment = {**os.environ, "SUBTHIS_SKIP_UPDATE": "1"}
    completed = subprocess.run([fresh, *arguments], env=environment)
    raise SystemExit(completed.returncode)


def _load_settings() -> dict:
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_settings(settings: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


def _reveal_in_file_manager(path: Path) -> None:
    """Best-effort: open the OS file manager with the file selected. Never raises.

    explorer.exe exits 1 even on success and needs backslashes; Linux uses the
    FileManager1 DBus interface (GNOME/KDE/Cinnamon/MATE) with a plain
    directory open as fallback, matching what Electron and VS Code do.
    """
    try:
        target = str(path.resolve())
        if sys.platform == "win32":
            subprocess.run(["explorer", "/select,", os.path.normpath(target)], timeout=10)
        elif sys.platform == "darwin":
            subprocess.run(["open", "-R", target], timeout=10)
        else:
            uri = "file://" + urllib.parse.quote(target)
            result = subprocess.run(
                [
                    "gdbus", "call", "--session",
                    "--dest", "org.freedesktop.FileManager1",
                    "--object-path", "/org/freedesktop/FileManager1",
                    "--method", "org.freedesktop.FileManager1.ShowItems",
                    f"['{uri}']", "",
                ],
                capture_output=True,
                timeout=10,
            )
            has_display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
            if result.returncode != 0 and has_display and shutil.which("xdg-open"):
                subprocess.run(
                    ["xdg-open", str(path.parent)], capture_output=True, timeout=10
                )
    except Exception:
        pass


def _parse_term_string(text: str) -> list[str]:
    try:
        return [term for term in shlex.split(text) if term.strip()]
    except ValueError as error:
        raise SubthisError(
            f"Could not read the terms ({error}). Check for an unclosed quote."
        ) from error


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


def run_config(rest: Sequence[str]) -> int:
    interactive = sys.stdin is not None and sys.stdin.isatty()
    if not rest or rest[0] not in ("key", "terms", "open"):
        raise SubthisError(
            "Usage:\n"
            "  subthis config key     change your saved OpenAI key\n"
            "  subthis config terms   review, add, and remove your saved terms\n"
            "  subthis config terms \"term 'multi word term'\"   add terms directly\n"
            "  subthis config open on|off   open the folder when subtitles are done"
        )
    if rest[0] == "open":
        value = rest[1].lower() if len(rest) > 1 else ""
        if value not in ("on", "off"):
            state = "on" if _load_settings().get("open_when_done") else "off"
            print(f"Opening the folder when done is currently: {state}")
            print("Turn it on or off with: subthis config open on   (or off)")
            return 0
        settings = _load_settings()
        settings["open_when_done"] = value == "on"
        _save_settings(settings)
        if value == "on":
            _say_ok("subthis will now open the folder with your file when it finishes.")
        else:
            _say_ok("subthis will only print the file location when it finishes.")
        return 0
    if rest[0] == "key":
        if len(rest) > 1:
            raise SubthisError(
                "For safety the key is not read from the command line.\n"
                "Run 'subthis config key' and paste it at the prompt."
            )
        print("Let's replace your saved OpenAI key. Copy the new one from:")
        _open_page(API_KEYS_URL, interactive)
        api_key = _prompt_for_working_key("", interactive)
        _write_env_file(api_key)
        _say_ok(f"New key saved to {ENV_FILE}")
        return 0

    text = " ".join(rest[1:]).strip()
    if not text:
        if interactive:
            return _terms_editor()
        print(
            "Type the terms to add, separated by spaces. Wrap a multi-word term\n"
            "in single quotes, like this: OpenAI 'API Platform'"
        )
        text = _ask("Terms: ")
    terms = _parse_term_string(text)
    if not terms:
        raise SubthisError("No terms were given.")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not TERMS_FILE.exists():
        TERMS_FILE.write_text(TERMS_TEMPLATE, encoding="utf-8")
    with TERMS_FILE.open("a", encoding="utf-8") as handle:
        for term in terms:
            handle.write(term + "\n")
    _say_ok(f"Added {len(terms)} term(s) to {TERMS_FILE}:")
    for term in terms:
        print("    " + term)
    print("  These now apply to every video you transcribe.")
    return 0


def _read_key() -> str:
    """Read one keypress without waiting for Enter, on any OS."""
    if sys.platform == "win32":
        import msvcrt

        char = msvcrt.getwch()
        if char in ("\x00", "\xe0"):
            return {"H": "up", "P": "down"}.get(msvcrt.getwch(), "")
        if char == "\r":
            return "enter"
        if char == "\x03":
            raise KeyboardInterrupt
        return char
    import select
    import termios
    import tty

    descriptor = sys.stdin.fileno()
    saved = termios.tcgetattr(descriptor)
    try:
        tty.setcbreak(descriptor)
        char = sys.stdin.read(1)
        if char == "\x1b":
            if select.select([sys.stdin], [], [], 0.05)[0]:
                if sys.stdin.read(1) == "[" and select.select([sys.stdin], [], [], 0.05)[0]:
                    return {"A": "up", "B": "down"}.get(sys.stdin.read(1), "")
            return "esc"
        if char in ("\r", "\n"):
            return "enter"
        return char
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, saved)


def _term_file_lines() -> list[str]:
    if TERMS_FILE.is_file():
        return TERMS_FILE.read_text(encoding="utf-8").splitlines()
    return TERMS_TEMPLATE.splitlines()


def _entry_indexes(lines: Sequence[str]) -> list[int]:
    return [
        index
        for index, line in enumerate(lines)
        if line.strip() and not line.strip().startswith("#")
    ]


def _draw_terms_screen(
    lines: Sequence[str],
    entries: Sequence[int],
    cursor: int,
    selected: set[int],
    message: str,
) -> None:
    columns, rows = shutil.get_terminal_size()
    output = ["\033[2J\033[H"]
    output.append(_paint("Your saved terms", "cyan", "bold") + "\n")
    output.append(
        _paint("These apply to every video you transcribe.", "dim") + "\n\n"
    )
    body_rows = max(3, rows - 7)
    top = max(0, min(cursor - body_rows // 2, len(entries) - body_rows))
    if not entries:
        output.append("  (no terms saved yet; press a to add your first one)\n")
    for position in range(top, min(top + body_rows, len(entries))):
        line_index = entries[position]
        pointer = _paint("❯ ", "cyan", "bold") if position == cursor else "  "
        box = _paint("[x] ", "green") if line_index in selected else "[ ] "
        text = lines[line_index].strip()[: columns - 8]
        output.append(pointer + box + text + "\n")
    output.append(f"\033[{rows - 1};1H")
    if message:
        output.append(_paint(message[: columns - 1], "yellow") + "\n")
    else:
        output.append("\n")
    guide = "↑/↓ move · space select · m mark/unmark all · a add · r remove · q save and quit"
    output.append(_paint(guide[: columns - 1], "dim"))
    sys.stdout.write("".join(output))
    sys.stdout.flush()


def _terms_editor() -> int:
    lines = _term_file_lines()
    cursor = 0
    selected: set[int] = set()
    message = ""
    sys.stdout.write("\033[?1049h\033[?25l")  # alternate screen, hidden cursor
    try:
        while True:
            entries = _entry_indexes(lines)
            cursor = max(0, min(cursor, len(entries) - 1))
            _draw_terms_screen(lines, entries, cursor, selected, message)
            message = ""
            key = _read_key()
            if key in ("q", "esc"):
                break
            if key in ("up", "k") and entries:
                cursor = max(0, cursor - 1)
            elif key in ("down", "j") and entries:
                cursor = min(len(entries) - 1, cursor + 1)
            elif key == " " and entries:
                line_index = entries[cursor]
                selected.symmetric_difference_update({line_index})
            elif key == "m" and entries:
                if len(selected) == len(entries):
                    selected.clear()
                else:
                    selected = set(entries)
            elif key == "a":
                sys.stdout.write(f"\033[{shutil.get_terminal_size().lines - 1};1H\033[K")
                sys.stdout.write("\033[?25h")
                sys.stdout.flush()
                try:
                    text = _ask(
                        "New term(s), spaces separate, 'single quotes' group: "
                    )
                    new_terms = _parse_term_string(text) if text else []
                except SubthisError as error:
                    new_terms = []
                    message = str(error)
                sys.stdout.write("\033[?25l")
                lines.extend(new_terms)
                if new_terms:
                    message = f"Added: {', '.join(new_terms)}"
            elif key == "r":
                if not selected:
                    message = "Nothing is selected. Move with ↑/↓ and press space first."
                    continue
                sys.stdout.write(f"\033[{shutil.get_terminal_size().lines - 1};1H\033[K")
                sys.stdout.write("\033[?25h")
                sys.stdout.flush()
                answer = _ask(
                    f"Delete {len(selected)} term(s)? Type y (yes) or n (no): "
                ).lower()
                sys.stdout.write("\033[?25l")
                if answer.startswith("y"):
                    lines = [
                        line for index, line in enumerate(lines) if index not in selected
                    ]
                    message = "Deleted."
                    selected = set()
                else:
                    message = "Nothing was deleted."
    finally:
        sys.stdout.write("\033[?25h\033[?1049l")  # cursor back, normal screen
        sys.stdout.flush()
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    TERMS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    remaining = len(_entry_indexes(lines))
    _say_ok(f"Saved {remaining} term(s) to {TERMS_FILE}")
    return 0


def _example_command() -> str:
    if sys.platform == "win32":
        return r"subthis C:\Users\you\Videos\lesson.mp4"
    if sys.platform == "darwin":
        return "subthis ~/Movies/lesson.mp4"
    return "subthis ~/Videos/lesson.mp4"


def run_setup() -> int:
    interactive = sys.stdin is not None and sys.stdin.isatty()
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
            timing_words = canonicalize_timed_words(timing_words, config.aliases)
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

teaching subthis your special words (names, brands, jargon):
  for one run:        subthis video.mp4 --term "OpenAI 'API Platform'"
                      (spaces separate terms; single quotes group a
                       multi-word term into one)
  for every video:    subthis config terms "OpenAI 'API Platform'"
  for one folder:     put the terms in a file named {PROJECT_TERMS_FILENAME}
                      next to your videos, one term per line

other commands:
  subthis docs             open the full documentation in your browser
  subthis setup            first-time setup (API key, checks)
  subthis config key       replace your saved OpenAI key
  subthis config terms     review, add, and remove saved terms
  subthis config open on   open the folder with the file when done
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
    parser.add_argument(
        "--term",
        action="append",
        default=[],
        help=(
            "extra terms for this run, separated by spaces; wrap a multi-word term "
            "in single quotes, e.g. --term \"OpenAI 'API Platform'\""
        ),
    )
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
    _init_color()
    if not arguments or arguments[0] == "help":
        build_parser().print_help()
        return 0
    if arguments[0] not in ("--version", "-h", "--help"):
        _maybe_offer_update(arguments)
    if arguments[0] == "setup":
        if len(arguments) > 1:
            raise SubthisError("setup takes no further arguments.")
        return run_setup()
    if arguments[0] == "config":
        return run_config(arguments[1:])
    if arguments[0] == "docs":
        print("The subthis documentation lives at:")
        _open_page(DOCS_URL, sys.stdin is not None and sys.stdin.isatty())
        return 0
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
    additions: list[str] = []
    project_terms = video.parent / PROJECT_TERMS_FILENAME
    if project_terms.is_file():
        print(f"Using the terms from {project_terms}", file=sys.stderr)
        additions.extend(project_terms.read_text(encoding="utf-8").splitlines())
    for term_string in args.term:
        additions.extend(_parse_term_string(term_string))
    aliases, terms = load_terms(args.terms_file.expanduser(), additions)
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
    print()
    _say_ok(f"Done! {len(cues)} subtitles were created.")
    print("  Your subtitle file is here:")
    print("    " + _paint(str(output), "bold"))
    if (
        _load_settings().get("open_when_done")
        and sys.stdout is not None
        and sys.stdout.isatty()
    ):
        print("  Opening its folder...")
        _reveal_in_file_manager(output)
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
