#!/usr/bin/env python3
"""Create short, accurately worded SRT captions from a video."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import difflib
import http.client
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Iterable, Sequence


__version__ = "1.8.1"

API_URL = "https://api.openai.com/v1/audio/transcriptions"
KEY_CHECK_URL = "https://api.openai.com/v1/models/whisper-1"
QUOTA_PROBE_URL = "https://api.openai.com/v1/chat/completions"
QUOTA_PROBE_MODEL = "gpt-5-nano"
API_KEYS_URL = "https://platform.openai.com/api-keys"
BILLING_URL = "https://platform.openai.com/settings/organization/billing/overview"
PYPI_JSON_URL = "https://pypi.org/pypi/subthis/json"
DOCS_URL = "https://subthis.webivize.com/docs/"
PROJECT_TERMS_FILENAME = "subthis-terms.txt"


def _config_dir() -> Path:
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", "").strip()
        if appdata:
            return Path(appdata) / "subthis"
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    # The XDG spec says a relative value must be ignored.
    base = Path(xdg) if xdg and Path(xdg).is_absolute() else Path.home() / ".config"
    return base / "subthis"


CONFIG_DIR = _config_dir()
ENV_FILE = CONFIG_DIR / ".env"
TERMS_FILE = CONFIG_DIR / "terms.txt"
SETTINGS_FILE = CONFIG_DIR / "settings.json"
ACCURATE_MODEL = "gpt-transcribe"
TIMING_MODEL = "whisper-1"
MAX_UPLOAD_BYTES = 24_000_000
CHUNK_SECONDS = 20 * 60         # gpt-4o-transcribe family caps a request at 1500s of audio
CHUNK_OVERLAP_SECONDS = 1.5
SHORT_TAIL_SECONDS = 120        # a leftover this short joins the previous chunk instead
PAUSE_SPLIT_SECONDS = 0.5   # stable-ts split_by_gap default
CUE_HANG_SECONDS = 0.5      # Netflix: out-time at least half a second past the audio
CUE_GAP_SECONDS = 0.0       # gap kept between consecutive cues (Netflix suggests 2 frames)
MIN_CUE_SECONDS = 5 / 6     # Netflix minimum event duration: 20 frames at 24fps


DEFAULT_ALIASES: dict[str, list[str]] = {
    "OpenAI": ["OpenAI", "Open AI", "אופן איי איי", "אופן איי-איי", "אופן איי"],
    "Claude": ["Claude", "Clod", "קלוד"],  # not קלאוד: that is how Hebrew says "cloud"
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


CAPTION_DEFAULTS: dict[str, object] = {
    "words": 3,                       # words per caption line (1-3)
    "pause": PAUSE_SPLIT_SECONDS,     # silence that starts a new line
    "hang": CUE_HANG_SECONDS,         # how long a line outlives its last word
    "min": MIN_CUE_SECONDS,           # minimum time a line stays on screen
    "gap": CUE_GAP_SECONDS,           # empty space kept between one line and the next
    "punctuation": "remove",          # remove | keep
    "silence": "cut",                 # cut | hold (hold = line stays up through silences)
}


@dataclasses.dataclass(frozen=True)
class Config:
    api_key: str
    aliases: dict[str, list[str]]
    terms: list[str]
    languages: list[str]
    max_words: int
    pause_split: float = PAUSE_SPLIT_SECONDS
    cue_hang: float = CUE_HANG_SECONDS
    min_cue: float = MIN_CUE_SECONDS
    cue_gap: float = CUE_GAP_SECONDS
    hold_through_silence: bool = False
    keep_punctuation: bool = False


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


def make_cues(
    words: Sequence[TimedWord],
    media_end: float,
    max_words: int = 3,
    *,
    pause_split: float = PAUSE_SPLIT_SECONDS,
    hang: float = CUE_HANG_SECONDS,
    min_cue: float = MIN_CUE_SECONDS,
    gap: float = CUE_GAP_SECONDS,
    hold_through_silence: bool = False,
    keep_punctuation: bool = False,
) -> list[Cue]:
    if not 1 <= max_words <= 3:
        raise ValueError("max_words must be between 1 and 3")
    clean_words = [
        TimedWord(cleaned, word.start, word.end)
        for word in words
        if (
            cleaned := (
                " ".join(word.text.split())
                if keep_punctuation
                else strip_caption_punctuation(word.text)
            )
        )
    ]
    if not clean_words:
        return []

    # Split into phrases at real pauses, then balance each phrase into groups
    # (7 words become 3+2+2, never 3+3+1) so no orphan cue trails a sentence.
    phrases: list[list[TimedWord]] = [[clean_words[0]]]
    for previous, word in zip(clean_words, clean_words[1:]):
        if word.start - previous.end > pause_split:
            phrases.append([word])
        else:
            phrases[-1].append(word)
    groups = [group for phrase in phrases for group in _balanced_groups(phrase, max_words)]

    cues: list[Cue] = []
    for index, group in enumerate(groups):
        start = group[0].start
        next_start = groups[index + 1][0].start if index + 1 < len(groups) else media_end
        latest_allowed = max(next_start - gap, start + 0.001)
        if hold_through_silence:
            end = min(latest_allowed, media_end)
        else:
            end = min(latest_allowed, group[-1].end + hang, media_end)
        end = max(end, start + 0.001)
        if end - start < min_cue:
            end = max(end, min(start + min_cue, latest_allowed, media_end))
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


def render_srt(cues: Sequence[Cue], keep_punctuation: bool = False) -> str:
    blocks = []
    for index, cue in enumerate(cues, start=1):
        line = cue.text if keep_punctuation else strip_caption_punctuation(cue.text)
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


_FFMPEG_EXPLANATIONS = (
    ("does not contain any stream", "This file has no sound track to transcribe."),
    ("Unknown encoder", "Your ffmpeg is missing an audio encoder."),
    ("moov atom not found", "This video file is incomplete or damaged (a recording that was cut off?)."),
    ("Invalid data found", "This does not look like a video or audio file ffmpeg can read."),
    ("Permission denied", "ffmpeg was not allowed to read this file. Check the file's permissions."),
    ("No such file", "ffmpeg could not find the file (was it moved or renamed?)."),
)


_EXTRA_TOOL_DIRS = (
    "/opt/homebrew/bin",   # Homebrew on Apple Silicon, before the shell is restarted
    "/usr/local/bin",      # Homebrew on Intel Macs
    "/opt/local/bin",      # MacPorts
    "/snap/bin",
)


def _find_tool(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for directory in _EXTRA_TOOL_DIRS:
        candidate = Path(directory) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    resolved = [_find_tool(command[0]) or command[0], *command[1:]]
    try:
        return subprocess.run(
            resolved,
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as error:
        raise SubthisError(
            f"Required command not found: {command[0]}. To install it: {_ffmpeg_install_hint()}"
        ) from error
    except OSError as error:  # e.g. a broken shortcut or a wrong-architecture binary
        raise SubthisError(f"{command[0]} could not be started ({error}). Reinstall it: {_ffmpeg_install_hint()}") from error
    except subprocess.CalledProcessError as error:
        detail = [line for line in error.stderr.strip().splitlines() if line.strip()]
        for needle, explanation in _FFMPEG_EXPLANATIONS:
            if any(needle in line for line in detail):
                raise SubthisError(explanation) from error
        message = " | ".join(detail[-3:]) if detail else "unknown media error"
        raise SubthisError(f"{command[0]} failed: {message}") from error


def _has_audio_stream(path: Path) -> bool:
    result = _run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "json", str(path)]
    )
    try:
        return bool(json.loads(result.stdout).get("streams"))
    except (ValueError, AttributeError):
        return True  # unknown; let extraction decide


def probe_duration(path: Path) -> float:
    """Length of the first audio stream, falling back to the container length."""
    result = _run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=duration:format=duration",
            "-of", "json", str(path),
        ]
    )
    duration = 0.0
    try:
        info = json.loads(result.stdout)
        candidates = [stream.get("duration") for stream in info.get("streams", [])]
        candidates.append(info.get("format", {}).get("duration"))
        for candidate in candidates:
            with contextlib.suppress(TypeError, ValueError):
                if candidate is not None and float(candidate) > 0:
                    duration = float(candidate)
                    break
    except (ValueError, AttributeError) as error:
        raise SubthisError("Could not read the length of this file.") from error
    if duration <= 0:
        raise SubthisError(
            "Could not read the length of this file. Browser recordings sometimes lack it;\n"
            "re-saving the file through any video editor or converter fixes that."
        )
    return duration


# Opus is smallest; AAC is built into every ffmpeg so it works when libopus is absent.
_AUDIO_ENCODERS: tuple[tuple[str, list[str]], ...] = (
    (".ogg", ["-c:a", "libopus", "-b:a", "32k", "-application", "voip"]),
    (".m4a", ["-c:a", "aac", "-b:a", "48k"]),
)


def _extract_audio(video: Path, offset: float, chunk_duration: float, base: Path) -> Path:
    last_error: SubthisError | None = None
    for suffix, encoder in _AUDIO_ENCODERS:
        output = base.with_suffix(suffix)
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
        if offset:
            command.extend(["-ss", f"{offset:.3f}"])
        command.extend(["-i", str(video), "-t", f"{chunk_duration:.3f}"])
        command.extend(["-map", "0:a:0", "-vn", "-sn", "-dn", "-ac", "1", "-ar", "16000"])
        command.extend(encoder)
        command.append(str(output))
        try:
            _run(command)
            return output
        except SubthisError as error:
            last_error = error
            with contextlib.suppress(FileNotFoundError):
                output.unlink()
    assert last_error is not None
    raise last_error


def extract_chunks(video: Path, directory: Path) -> tuple[list[AudioChunk], float]:
    if not _has_audio_stream(video):
        raise SubthisError("This file has no sound track, so there is nothing to transcribe.")
    duration = probe_duration(video)
    chunks: list[AudioChunk] = []
    offset = 0.0
    index = 0
    while offset < duration:
        chunk_duration = min(CHUNK_SECONDS + CHUNK_OVERLAP_SECONDS, duration - offset)
        if 0 < duration - (offset + CHUNK_SECONDS) < SHORT_TAIL_SECONDS:
            chunk_duration = duration - offset  # swallow a short tail (often outro music)
        output = _extract_audio(video, offset, chunk_duration, directory / f"chunk-{index:04d}.ogg")
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


_MIME_TYPES = {".ogg": "audio/ogg", ".m4a": "audio/mp4", ".mp3": "audio/mpeg", ".wav": "audio/wav"}


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
            f"Content-Type: {_MIME_TYPES.get(file_path.suffix.lower(), 'application/octet-stream')}\r\n\r\n"
        ).encode()
    )
    body.extend(file_path.read_bytes())
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def _api_error_message(payload: bytes, status: int | None = None) -> str:
    text = payload.decode("utf-8", errors="replace")
    if status in (401, 403) or "invalid_api_key" in text:
        if _key_comes_from_environment():
            return (
                "OpenAI does not accept the key in your OPENAI_API_KEY environment variable, "
                "which overrides the key subthis saved. Remove or fix that variable "
                "(it is usually set in your shell profile) and run again."
            )
        return (
            "OpenAI no longer accepts your saved key (it may have been revoked). "
            f"Run 'subthis config key' to enter a new one from {API_KEYS_URL}"
        )
    if "insufficient_quota" in text:
        return (
            "Your OpenAI account is out of credit, so it cannot transcribe right now. "
            f"Add credit at {BILLING_URL} and run subthis again."
        )
    prefix = f"OpenAI API error{f' {status}' if status else ''}"
    message = None
    with contextlib.suppress(ValueError, AttributeError):
        message = json.loads(text).get("error", {}).get("message")
    result = f"{prefix}: {message.strip()}" if isinstance(message, str) and message.strip() else prefix
    if status == 429:
        result += " (OpenAI is asking us to slow down. Wait a minute and try again.)"
    elif status == 407:
        result = "A proxy on this network wants a login before letting subthis reach OpenAI. Ask your IT person."
    elif status is not None and status >= 500:
        result += " (OpenAI had a temporary problem. Running the same command again usually works.)"
    return result


_RETRYABLE_STATUSES = (408, 429, 500, 502, 503, 504, 520, 522, 524)


def _network_reason(error: BaseException) -> str:
    text = str(getattr(error, "reason", None) or error)
    if "CERTIFICATE_VERIFY_FAILED" in text:
        return (
            "a secure connection could not be verified. Something on this network or computer\n"
            "(antivirus, a corporate proxy, an old system) intercepts secure traffic; ask your IT\n"
            "person, or try another network."
        )
    if "Name or service not known" in text or "nodename nor servname" in text or "getaddrinfo" in text:
        return "no internet connection (the OpenAI address could not be looked up)."
    if "timed out" in text.lower():
        return "the connection timed out. Check your internet connection and try again."
    return text


def _post_with_retry(request: urllib.request.Request, timeout: float, attempts: int = 3) -> bytes:
    """A blip at hour two of a three-hour job must not throw the job away."""
    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            payload = error.read()
            retryable = error.code in _RETRYABLE_STATUSES and b"insufficient_quota" not in payload
            if not retryable or attempt == attempts:
                raise SubthisError(_api_error_message(payload, error.code)) from error
            problem = f"OpenAI answered with error {error.code}"
        except (OSError, http.client.HTTPException) as error:  # URLError, timeouts, resets, short reads
            if attempt == attempts:
                raise SubthisError(f"Could not reach the OpenAI API: {_network_reason(error)}") from error
            problem = "the connection to OpenAI dropped"
        print(f"{problem}; trying again in {delay:.0f}s...", file=sys.stderr)
        time.sleep(delay)
        delay *= 2
    raise SubthisError("Could not reach the OpenAI API.")


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
            "User-Agent": f"subthis/{__version__}",
        },
        method="POST",
    )
    payload = _post_with_retry(request, timeout=1800)
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
    # Punctuation stays attached here; alignment normalizes tokens itself and
    # make_cues strips or keeps it according to the caption settings.
    return " ".join(canonical.split())


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
    ]
    if len(config.languages) == 1:
        # Forcing Hebrew onto an English-only video produces junk tokens;
        # with several languages configured, let whisper detect.
        fields.append(("language", config.languages[0]))
    if timing_prompt:
        fields.append(("prompt", timing_prompt))
    response = request_transcription(config.api_key, chunk.path, fields)
    raw_words = response.get("words")
    if raw_words is None and not str(response.get("text", "")).strip():
        return []  # silent chunk: whisper omits the words list entirely
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
        with ENV_FILE.open(encoding="utf-8-sig") as handle:
            for line in handle:
                match = re.match(r"\s*OPENAI_API_KEY\s*=\s*(.*?)\s*$", line)
                if match:
                    value = match.group(1).strip().strip("\"'")
                    if value:
                        return value
    raise SubthisError(
        f"No OpenAI key is saved yet ({ENV_FILE} does not have one). Run: subthis setup"
    )


def _key_comes_from_environment() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


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
        return (
            "winget install --id Gyan.FFmpeg -e\n"
            "    then close this window and open a new PowerShell window (the new\n"
            "    program is only visible to windows opened after the install)"
        )
    if sys.platform == "darwin":
        return (
            "brew install ffmpeg   then open a new Terminal window\n"
            "    (no Homebrew yet? install it first with the command on https://brew.sh)"
        )
    return (
        "sudo apt install ffmpeg  /  sudo dnf install ffmpeg  /  sudo pacman -S ffmpeg\n"
        "    (pick the one for your distribution), then open a new terminal"
    )


_COLOR = False
_ANSI = {"green": "32", "red": "31", "yellow": "33", "cyan": "36", "bold": "1", "dim": "2"}
# Pretty symbols, with plain-ASCII stand-ins for consoles that cannot show them
# (a Windows console on code page 1252 or 862, a LANG=C Linux session).
_SYMBOLS_UTF8 = {"ok": "✓", "bad": "✗", "pointer": "❯", "arrows": "↑/↓", "dot": "·", "box": "╭─╮│╰╯"}
_SYMBOLS_ASCII = {"ok": "OK", "bad": "X", "pointer": ">", "arrows": "up/down", "dot": "-", "box": "+-+|++"}
_SYM = dict(_SYMBOLS_UTF8)


def _paint(text: str, *names: str) -> str:
    if not _COLOR:
        return text
    prefix = "".join(f"\033[{_ANSI[name]}m" for name in names)
    return f"{prefix}{text}\033[0m"


def _say_ok(text: str) -> None:
    print(_paint(f"  {_SYM['ok']} ", "green", "bold") + text)


def _say_bad(text: str) -> None:
    print(_paint(f"  {_SYM['bad']} ", "red", "bold") + text)


def _say_note(text: str) -> None:
    print(_paint("  ! ", "yellow", "bold") + text)


def _banner(title: str) -> None:
    tl, hz, tr, vt, bl, br = _SYM["box"]
    line = hz * (len(title) + 2)
    print(_paint(f"{tl}{line}{tr}\n{vt} {title} {vt}\n{bl}{line}{br}", "cyan"))


def _stdout_is_utf8() -> bool:
    encoding = (getattr(sys.stdout, "encoding", None) or "").lower().replace("-", "").replace("_", "")
    return encoding in ("utf8", "utf8sig")


def _is_interactive() -> bool:
    """Both ends must be a terminal: with stdout piped, prompts would vanish into the pipe."""
    return all(
        stream is not None and hasattr(stream, "isatty") and stream.isatty()
        for stream in (sys.stdin, sys.stdout)
    )


def _flush_pending_input() -> None:
    """Discard keystrokes typed before a prompt appeared.

    Someone pressing Enter during a pause (the update check, a network
    call) must not have that Enter answer the next question for them.
    """
    with contextlib.suppress(Exception):
        if sys.stdin is None or not sys.stdin.isatty():
            return
        if sys.platform == "win32":
            import msvcrt

            while msvcrt.kbhit():
                msvcrt.getwch()
        else:
            import termios

            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)


def _ask(prompt: str) -> str:
    _flush_pending_input()
    try:
        return input(_paint(prompt, "bold")).strip()
    except EOFError:
        return ""


def _open_page(url: str, interactive: bool) -> None:
    print("    " + _paint(url, "cyan", "bold"))
    headless_linux = (
        sys.platform not in ("win32", "darwin")
        and not os.environ.get("DISPLAY")
        and not os.environ.get("WAYLAND_DISPLAY")
    )
    if interactive and headless_linux:
        # webbrowser would fall back to a text browser inside this terminal
        # (w3m/lynx) and take over the screen; the printed address is better.
        print(_paint("    (copy that address into a browser on any device)", "dim"))
        return
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
    global _VT_OK
    if sys.platform == "win32":
        _VT_OK = _enable_windows_vt()
        if not _VT_OK:
            _COLOR = False
    # The legacy Windows console (PowerShell outside Windows Terminal) has no
    # font fallback, so ✓ ✗ ❯ render as boxes there.
    legacy_windows_console = sys.platform == "win32" and not os.environ.get("WT_SESSION")
    _SYM.clear()
    _SYM.update(_SYMBOLS_ASCII if legacy_windows_console or not _stdout_is_utf8() else _SYMBOLS_UTF8)


_VT_OK = True


def _enable_windows_vt() -> bool:
    """Turn on ANSI escape processing in the Windows console; say whether it worked."""
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


def _normalize_pasted_key(text: str) -> str:
    """Accept a key however it was copied: with quotes, a line of an .env
    file, an 'export', a 'Bearer' prefix, or trailing junk lines."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    key = lines[0] if lines else ""
    for _ in range(3):
        key = key.strip().strip("\"'")
        for prefix in ("export ", "OPENAI_API_KEY=", "OPENAI_API_KEY =", "Bearer ", "bearer "):
            if key.startswith(prefix):
                key = key[len(prefix):]
    return key.strip().strip("\"'")


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
    """Pick the updater that actually installed this copy."""
    prefix = str(Path(sys.prefix).resolve()).lower()
    if "pipx" in prefix and shutil.which("pipx"):
        return "pipx upgrade subthis"
    if shutil.which("uv"):
        return "uv tool upgrade subthis"
    if shutil.which("pipx"):
        return "pipx upgrade subthis"
    return f'"{sys.executable}" -m pip install --upgrade subthis'


def _maybe_offer_update(arguments: Sequence[str]) -> None:
    if os.environ.get("SUBTHIS_SKIP_UPDATE") == "1":
        return
    if sys.stdin is None or not sys.stdin.isatty():
        return
    if sys.stdout is None or not sys.stdout.isatty():
        return
    settings = _load_settings()
    last_check = settings.get("last_update_check")
    if isinstance(last_check, (int, float)) and time.time() - last_check < 24 * 3600:
        return  # once a day is plenty
    print(_paint("checking for a newer version...", "dim"), flush=True)
    latest = _latest_pypi_version()
    if latest:
        settings["last_update_check"] = time.time()
        with contextlib.suppress(OSError):
            _save_settings(settings)
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
    command = _update_command()
    if sys.platform == "win32":
        # Windows will not let the updater replace subthis.exe while it runs.
        print(
            "On Windows the update has to run while subthis is closed. Copy this command,\n"
            "close this window, open a new PowerShell window, and run it there:\n"
            "    " + _paint(command, "bold") + "\n"
            "Continuing on the current version for now.\n"
        )
        return
    print("Updating...")
    try:
        result = subprocess.run(
            shlex.split(command, posix=sys.platform != "win32"),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as error:
        _say_note(f"Could not start the updater ({error}). Run this yourself later:\n    {command}")
        return
    fresh = shutil.which("subthis")
    if result.returncode != 0 or fresh is None:
        tail = " | ".join(line for line in result.stdout.strip().splitlines()[-3:] if line.strip())
        _say_note(
            "The update did not finish, so this run continues on the current version.\n"
            f"    Updater said: {tail or 'nothing'}\n"
            f"    To try by hand (close other subthis windows first on Windows): {command}"
        )
        return
    _say_ok("Updated. Continuing right where you were...\n")
    environment = {**os.environ, "SUBTHIS_SKIP_UPDATE": "1"}
    # Let the child own Ctrl+C so the user does not see "cancelled" twice.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    completed = subprocess.run([fresh, *arguments], env=environment)
    raise SystemExit(completed.returncode)


def _load_settings() -> dict:
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8-sig"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_atomically(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                # Windows: antivirus or a cloud-sync client may hold the file
                # for a moment right after it was written.
                if attempt == 4:
                    raise
                time.sleep(0.3 * (attempt + 1))
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _save_settings(settings: dict) -> None:
    _write_atomically(SETTINGS_FILE, json.dumps(settings, indent=2) + "\n")


def _reveal_in_file_manager(path: Path) -> None:
    """Best-effort: open the OS file manager with the file selected. Never raises.

    explorer.exe exits 1 even on success and needs backslashes; Linux uses the
    FileManager1 DBus interface (GNOME/KDE/Cinnamon/MATE) with a plain
    directory open as fallback, matching what Electron and VS Code do.
    """
    try:
        target = str(path.resolve())
        if sys.platform == "win32":
            if "," in target:
                # Explorer splits its own command line on commas.
                os.startfile(str(path.parent))  # type: ignore[attr-defined]
            else:
                subprocess.run(["explorer", "/select,", os.path.normpath(target)], timeout=10)
        elif sys.platform == "darwin":
            subprocess.run(["open", "-R", target], timeout=10)
        else:
            uri = "file://" + urllib.parse.quote(target)
            shown = False
            with contextlib.suppress(Exception):  # gdbus missing or no session bus
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
                shown = result.returncode == 0
            has_display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
            if not shown and has_display and shutil.which("xdg-open"):
                subprocess.run(
                    ["xdg-open", str(path.parent)], capture_output=True, timeout=10
                )
    except Exception:
        pass


def _parse_term_string(text: str) -> list[str]:
    # Notes/TextEdit/Word swap in curly quotes; treat them as the straight ones.
    text = text.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'}))
    # An apostrophe inside a word (צ'אט, ג'יפיטי, don't) is part of the word,
    # not a quote that groups terms; only quotes at a word's edge group.
    placeholder = "\x00"
    text = re.sub(r"(?<=\S)'(?=\S)", placeholder, text)
    try:
        return [term.replace(placeholder, "'") for term in shlex.split(text) if term.strip()]
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
        body = error.read().decode("utf-8", errors="replace")
        if error.code == 401 or (error.code == 403 and "invalid_api_key" in body):
            return "invalid", ""
        if error.code == 403:
            return "ok", ""  # a restricted key without model-read permission is still a key
        return "unreachable", f"HTTP {error.code}"
    except OSError as error:
        return "unreachable", _network_reason(error)

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
        api_key = _normalize_pasted_key(entered) if entered else existing_key
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
            print("    " + _paint(API_KEYS_URL, "cyan", "bold"))
            existing_key = ""
            continue
        if status == "no_credit":
            _say_bad("The key itself works, but the account behind it has no credit yet.")
            print(
                "    The transcription service is prepaid, like a phone card. In the\n"
                "    billing page that opens next:\n"
                "      1. Click 'Add payment method' and enter your card details.\n"
                "      2. Click 'Add to credit balance' and choose 5 dollars (the\n"
                "         minimum; it covers roughly five hours of video).\n"
                "      3. If it offers auto-reload, you can switch that off."
            )
            if not interactive:
                raise SubthisError(f"The OpenAI account has no credit. Add credit at {BILLING_URL}")
            _ask("    Press Enter to open the billing page... ")
            _open_page(BILLING_URL, interactive)
            _ask("    Press Enter once your balance shows the credit, and it will be checked again... ")
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
    content = f"OPENAI_API_KEY={api_key}\n"
    if sys.platform == "win32":
        # %APPDATA% is already private to the user; POSIX modes mean nothing here,
        # and a raw fd would be opened in text mode and double the line endings.
        ENV_FILE.write_text(content, encoding="utf-8")
        return
    descriptor = os.open(ENV_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(content)
    with contextlib.suppress(OSError):
        os.chmod(ENV_FILE, 0o600)


def _caption_settings() -> dict[str, object]:
    merged = dict(CAPTION_DEFAULTS)
    saved = _load_settings().get("captions")
    if isinstance(saved, dict):
        for name, value in saved.items():
            if name in merged:
                merged[name] = value
    try:
        merged["words"] = min(3, max(1, int(merged["words"])))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        merged["words"] = CAPTION_DEFAULTS["words"]
    for name, floor in (("pause", 0.05), ("hang", 0.0), ("min", 0.0), ("gap", 0.0)):
        try:
            number = float(str(merged[name]).replace(",", "."))
            merged[name] = min(10.0, max(floor, number))
        except (TypeError, ValueError):
            merged[name] = CAPTION_DEFAULTS[name]
    if merged["punctuation"] not in ("keep", "remove"):
        merged["punctuation"] = CAPTION_DEFAULTS["punctuation"]
    if merged["silence"] not in ("hold", "cut"):
        merged["silence"] = CAPTION_DEFAULTS["silence"]
    return merged


_CAPTION_HELP: dict[str, str] = {
    "words": "words per caption line, 1 to 3",
    "pause": "a silence this long (seconds) starts a new line",
    "hang": "how long a line stays after its last word (seconds)",
    "min": "minimum time a line stays on screen (seconds)",
    "gap": "empty space kept between one line and the next (seconds)",
    "punctuation": "'remove' cleans captions, 'keep' leaves punctuation in",
    "silence": "'cut' ends a line after the hang time, 'hold' keeps it up until the next line",
}


def _config_captions(rest: Sequence[str]) -> int:
    if rest and rest[0] == "reset":
        settings = _load_settings()
        settings.pop("captions", None)
        _save_settings(settings)
        _say_ok("Caption settings are back to the defaults.")
        return 0
    current = _caption_settings()
    if not rest:
        print("Caption settings (change one with: subthis config captions <name> <value>):\n")
        saved = _load_settings().get("captions")
        overridden = set(saved.keys()) if isinstance(saved, dict) else set()
        for name, value in current.items():
            shown = f"{value:.2f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)
            origin = "(changed)" if name in overridden else "(default)"
            print(f"  {name:<12} {shown:<8} {origin:<10} {_CAPTION_HELP[name]}")
        print("\n  reset        put everything back to the defaults")
        return 0
    name = rest[0]
    if name not in CAPTION_DEFAULTS:
        raise SubthisError(
            "Unknown caption setting: " + name + "\n"
            "Settings: words, pause, hang, min, gap, punctuation, silence (or reset)."
        )
    if len(rest) < 2:
        value = current[name]
        shown = f"{value:.2f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)
        print(f"{name} is {shown}  ({_CAPTION_HELP[name]})")
        print(f"Change it with: subthis config captions {name} <value>")
        return 0
    raw = rest[1].strip().lower()
    value: object
    if name == "words":
        try:
            value = int(raw)
        except ValueError:
            raise SubthisError("words must be 1, 2 or 3.") from None
        if value not in (1, 2, 3):
            raise SubthisError("words must be 1, 2 or 3.")
    elif name in ("pause", "hang", "min", "gap"):
        try:
            value = float(raw.replace(",", "."))  # 0,5 is how half of Europe types it
        except ValueError:
            raise SubthisError(f"{name} must be a number of seconds, like 0.5") from None
        if not 0 <= value <= 10 or (name == "pause" and value <= 0):
            raise SubthisError(f"{name} must be between {'just above 0' if name == 'pause' else '0'} and 10 seconds.")
    elif name == "punctuation":
        if raw not in ("keep", "remove"):
            raise SubthisError("punctuation must be 'keep' or 'remove'.")
        value = raw
    else:
        if raw not in ("hold", "cut"):
            raise SubthisError("silence must be 'hold' or 'cut'.")
        value = raw
    settings = _load_settings()
    captions = settings.get("captions")
    if not isinstance(captions, dict):
        captions = {}
    captions[name] = value
    settings["captions"] = captions
    _save_settings(settings)
    shown = f"{value:.2f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)
    _say_ok(f"Saved: {name} = {shown}. This applies to every video from now on.")
    return 0


def run_config(rest: Sequence[str]) -> int:
    interactive = _is_interactive()
    if not rest or rest[0] not in ("key", "terms", "open", "captions"):
        raise SubthisError(
            "Usage:\n"
            "  subthis config key     change your saved OpenAI key\n"
            "  subthis config terms   review, add, and remove your saved terms\n"
            "  subthis config terms \"term 'multi word term'\"   add terms directly\n"
            "  subthis config open on|off   open the folder when subtitles are done\n"
            "  subthis config captions      view and tune caption timing and style"
        )
    if rest[0] == "captions":
        return _config_captions(rest[1:])
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
        if interactive and _VT_OK:
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

    def read_byte(wait: float | None) -> bytes:
        # Straight from the fd: the text wrapper would slurp "ESC [ A" into
        # its own buffer and make an arrow key look like a bare Escape.
        if wait is not None and not select.select([descriptor], [], [], wait)[0]:
            return b""
        return os.read(descriptor, 1)

    try:
        tty.setcbreak(descriptor, termios.TCSANOW)
        first = read_byte(None)
        if first == b"\x1b":
            second = read_byte(0.05)
            if second in (b"[", b"O"):
                third = read_byte(0.05)
                return {b"A": "up", b"B": "down"}.get(third, "")
            return "esc"
        if first in (b"\r", b"\n"):
            return "enter"
        if first == b"\x03":
            raise KeyboardInterrupt
        # Collect the rest of a multi-byte UTF-8 character (e.g. a Hebrew key).
        needed = 0
        lead = first[0] if first else 0
        if lead >= 0xF0:
            needed = 3
        elif lead >= 0xE0:
            needed = 2
        elif lead >= 0xC0:
            needed = 1
        for _ in range(needed):
            first += read_byte(0.05)
        return first.decode("utf-8", errors="ignore")
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, saved)


def _term_file_lines() -> list[str]:
    if TERMS_FILE.is_file():
        text = TERMS_FILE.read_text(encoding="utf-8-sig", errors="replace")
        if text.lstrip().startswith("{\\rtf"):
            raise SubthisError(
                f"{TERMS_FILE} was saved as rich text (RTF). Save it as plain text instead\n"
                "(in TextEdit: Format > Make Plain Text)."
            )
        return text.splitlines()
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
        pointer = _paint(f"{_SYM['pointer']} ", "cyan", "bold") if position == cursor else "  "
        box = _paint("[x] ", "green") if line_index in selected else "[ ] "
        text = lines[line_index].strip()[: columns - 8]
        output.append(pointer + box + text + "\n")
    output.append(f"\033[{rows - 1};1H")
    if message:
        output.append(_paint(message[: columns - 1], "yellow") + "\n")
    else:
        output.append("\n")
    dot = _SYM["dot"]
    guide = (
        f"{_SYM['arrows']} move {dot} space select {dot} m mark/unmark all "
        f"{dot} a add {dot} r remove {dot} q save and quit"
    )
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
            try:
                key = _read_key()
            except KeyboardInterrupt:
                message = "Press q to save and quit (Ctrl+C would throw away your changes; press it again to do that)."
                try:
                    _draw_terms_screen(lines, entries, cursor, selected, message)
                    key = _read_key()
                except KeyboardInterrupt:
                    raise
            # A Hebrew keyboard layout sends these for the same physical keys.
            key = {"/": "q", "ש": "a", "ר": "r", "צ": "m", "ח": "j", "ל": "k"}.get(key, key)
            if key == "q":
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


def _write_srt_somewhere(output: Path, srt_text: str) -> Path:
    """Write the subtitles, never losing them: fall back to a sibling name,
    then to the Desktop or home folder, and say where they went."""
    candidates = [output]
    stem, suffix = output.stem, output.suffix or ".srt"
    candidates.append(output.with_name(f"{stem}-subthis{suffix}"))
    for folder in (Path.home() / "Desktop", Path.home(), Path.cwd()):
        candidates.append(folder / f"{stem}{suffix}")
    last_error: OSError | None = None
    for index, candidate in enumerate(candidates):
        try:
            # SRT with a BOM: Windows Media Player and many TVs otherwise guess
            # the encoding of Hebrew text wrong; VLC/mpv/YouTube accept both.
            _write_atomically(candidate, "﻿" + srt_text)
        except OSError as error:
            last_error = error
            continue
        if index:
            _say_note(f"Could not write to {output.parent}, so the file went elsewhere:")
        return candidate
    raise SubthisError(
        f"The subtitles were made but could not be saved anywhere ({last_error}).\n"
        "Free up disk space or check folder permissions, then run again."
    )


def _example_command() -> str:
    if sys.platform == "win32":
        return r"subthis C:\Users\you\Videos\lesson.mp4"
    if sys.platform == "darwin":
        return "subthis ~/Movies/lesson.mp4"
    return "subthis ~/Videos/lesson.mp4"


def run_setup() -> int:
    interactive = _is_interactive()
    _banner(f"Welcome to subthis  ·  v{__version__}")
    print(
        "\nsubthis turns a video into subtitles. This one-time setup takes about\n"
        "two minutes. Nothing on your computer is changed except subthis's own\n"
        f"settings folder: {CONFIG_DIR}\n"
    )

    if _key_comes_from_environment():
        _say_note(
            "An OPENAI_API_KEY variable is set in this terminal's environment. It will win\n"
            "    over whatever you save here; if it is stale, remove it from your shell profile."
        )
    if _find_tool("ffmpeg") and _find_tool("ffprobe"):
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
        print(
            "Here's what happens next:\n"
            "  1. OpenAI's key page opens in your browser. No account yet? The site\n"
            "     will have you sign up first (same login as ChatGPT, and it's quick).\n"
            "  2. On the key page, click 'Create new secret key', give it any name,\n"
            "     click 'Create secret key', then 'Copy'. It is shown only this once.\n"
            "  3. Come back here and paste it.\n"
        )
        if interactive:
            _ask("Press Enter to open the key page in your browser... ")
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
        with path.open(encoding="utf-8-sig", errors="replace") as handle:
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


def _wait_with_heartbeat(futures: Sequence[Future], label: str) -> None:
    started = time.monotonic()
    last_note = started
    while True:
        # Short waits keep Ctrl+C responsive on Windows, where a long lock
        # wait cannot be interrupted.
        done, pending = wait(futures, timeout=0.5)
        if not pending:
            return
        now = time.monotonic()
        if now - last_note >= 30:
            last_note = now
            elapsed = int(now - started)
            print(f"  still working on {label} ({elapsed // 60}m{elapsed % 60:02d}s so far)...", file=sys.stderr)


def _trim_overlap(words: list[TimedWord], chunk: AudioChunk, previous_end: float | None) -> list[TimedWord]:
    """Keep each word from exactly one chunk: the boundary sits mid-overlap."""
    if previous_end is None:
        return words
    boundary = previous_end - CHUNK_OVERLAP_SECONDS / 2
    return [word for word in words if word.start >= boundary]


def transcribe_video(video: Path, config: Config) -> tuple[list[Cue], float]:
    with tempfile.TemporaryDirectory(prefix="subthis-", ignore_cleanup_errors=True) as temporary:
        chunks, duration = extract_chunks(video, Path(temporary))
        if len(chunks) > 1:
            print(
                f"Long video: {len(chunks)} parts of up to {CHUNK_SECONDS // 60} minutes each. "
                "Each part takes roughly a minute per 10 minutes of speech.",
                file=sys.stderr,
            )
        all_words: list[list[TimedWord]] = []
        previous_end: float | None = None
        for index, chunk in enumerate(chunks, start=1):
            label = f"part {index} of {len(chunks)}" if len(chunks) > 1 else "your video"
            print(f"Transcribing {label}...", file=sys.stderr)
            pool = ThreadPoolExecutor(max_workers=2)
            try:
                accurate_future = pool.submit(transcribe_accurate, chunk, config)
                timing_future = pool.submit(transcribe_timing, chunk, config)
                _wait_with_heartbeat([accurate_future, timing_future], label)
                accurate_text = accurate_future.result()
                timing_words = timing_future.result()
            except KeyboardInterrupt:
                # Don't sit through a minutes-long upload after Ctrl+C.
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            finally:
                pool.shutdown(wait=False)
            chunk_end = chunk.offset + chunk.duration
            if not accurate_text and not timing_words:
                previous_end = chunk_end
                continue
            if not accurate_text:
                # The accurate model heard nothing but whisper timed something
                # (music, mumbling): keep whisper's words rather than fail.
                aligned = list(timing_words)
            elif not timing_words:
                # The opposite: real words with no timing; spread them evenly.
                aligned = _distribute_words(accurate_text.split(), 0.0, chunk.duration)
            else:
                timing_words = canonicalize_timed_words(timing_words, config.aliases)
                aligned = align_accurate_words(accurate_text, timing_words)
            shifted = [
                TimedWord(word.text, word.start + chunk.offset, word.end + chunk.offset)
                for word in aligned
            ]
            all_words.append(_trim_overlap(shifted, chunk, previous_end))
            previous_end = chunk_end
        merged = merge_chunk_words(all_words)
        cues = make_cues(
            merged,
            duration,
            config.max_words,
            pause_split=config.pause_split,
            hang=config.cue_hang,
            min_cue=config.min_cue,
            gap=config.cue_gap,
            hold_through_silence=config.hold_through_silence,
            keep_punctuation=config.keep_punctuation,
        )
        return cues, duration


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
  subthis config captions  tune caption timing and style
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
        default=None,
        help="maximum words per subtitle (default: 3, or your 'subthis config captions words' setting)",
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
    if arguments and arguments[0].lower() in ("help", "setup", "config", "docs"):
        arguments[0] = arguments[0].lower()  # Setup / SETUP are the same command
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
    if os.environ.get("SUDO_USER") and sys.platform != "win32":
        raise SubthisError(
            "Please run subthis without sudo. With sudo it would save settings and files\n"
            "as the wrong user, and lose your saved key on the next normal run."
        )
    raw_input_path = str(args.video)
    # argparse already turned the text into a Path, which collapses "//",
    # so look for "scheme:/" (two or more letters, so C:/ stays a drive).
    if re.match(r"^[a-z][a-z0-9+.-]+:/", raw_input_path, re.IGNORECASE):
        raise SubthisError(
            "subthis works on video files saved on this computer, not on links.\n"
            "Download the video first, then run subthis on the downloaded file."
        )
    given = args.video.expanduser()
    if given.is_dir():
        raise SubthisError(f"That is a folder, not a file: {given}\nDrag the video file itself.")
    video = given.absolute()
    if not video.is_file():
        hint = ""
        if sys.platform == "win32" and len(str(video)) > 240:
            hint = "\nThe path is very long; Windows may refuse it. Move the video to a shorter folder."
        raise SubthisError(f"Input file not found: {video}{hint}")
    if sys.platform == "win32":
        attributes = getattr(os.stat(video), "st_file_attributes", 0)
        if attributes & 0x400000:  # FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS
            raise SubthisError(
                "This video is stored online only (OneDrive). Right-click it in Explorer,\n"
                "choose 'Always keep on this device', wait for the download, then run again."
            )
    if not _find_tool("ffmpeg") or not _find_tool("ffprobe"):
        raise SubthisError(
            "subthis needs ffmpeg and ffprobe to read the sound from your video.\n"
            f"To install: {_ffmpeg_install_hint()}"
        )
    if args.output is not None:
        output = args.output.expanduser().absolute()
        if output.is_dir():
            output = output / video.with_suffix(".srt").name
    else:
        output = video.with_suffix(".srt")
    if output.resolve() == video.resolve():
        raise SubthisError("The output path must differ from the input path.")
    if output.exists() and not args.force:
        raise SubthisError(f"Output already exists: {output}. Use --force to replace it.")
    # Prove the destination is writable before spending money on the API.
    output.parent.mkdir(parents=True, exist_ok=True)
    probe_file = output.with_name(f".{output.name}.subthis-{os.getpid()}.probe")
    try:
        probe_file.write_text("", encoding="utf-8")
    except OSError as error:
        raise SubthisError(
            f"Cannot write into {output.parent} ({error.strerror or error}).\n"
            "Choose another place with:  -o /path/to/subtitles.srt"
        ) from error
    finally:
        with contextlib.suppress(OSError):
            probe_file.unlink()
    additions: list[str] = []
    # Explorer hides extensions, so "subthis-terms.txt" often becomes
    # "subthis-terms.txt.txt"; accept both.
    for candidate in (PROJECT_TERMS_FILENAME, PROJECT_TERMS_FILENAME + ".txt"):
        project_terms = video.parent / candidate
        if project_terms.is_file():
            print(f"Using the terms from {project_terms}", file=sys.stderr)
            additions.extend(
                project_terms.read_text(encoding="utf-8-sig", errors="replace").splitlines()
            )
            break
    for term_string in args.term:
        additions.extend(_parse_term_string(term_string))
    terms_file = args.terms_file.expanduser()
    if terms_file != TERMS_FILE and not terms_file.is_file():
        raise SubthisError(f"The terms file you pointed at does not exist: {terms_file}")
    aliases, terms = load_terms(terms_file, additions)
    captions = _caption_settings()
    config = Config(
        api_key=_load_api_key(),
        aliases=aliases,
        terms=terms,
        languages=args.languages or ["he", "en"],
        max_words=args.max_words if args.max_words is not None else int(captions["words"]),
        pause_split=float(captions["pause"]),
        cue_hang=float(captions["hang"]),
        min_cue=float(captions["min"]),
        cue_gap=float(captions["gap"]),
        hold_through_silence=captions["silence"] == "hold",
        keep_punctuation=captions["punctuation"] == "keep",
    )
    cues, _duration = transcribe_video(video, config)
    if not cues:
        raise SubthisError("No speech was detected, so no subtitle file was written.")
    srt_text = render_srt(cues, keep_punctuation=config.keep_punctuation)
    output = _write_srt_somewhere(output, srt_text)
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
        for stream in (sys.stdout, sys.stderr):
            with contextlib.suppress(Exception):
                stream.flush()
        # Worker threads may still be mid-upload; a normal exit would wait
        # for them (minutes). Leave now; the temp dir was already cleaned.
        os._exit(130)
    except BrokenPipeError:
        # stdout was closed early (e.g. piped into `head`); leave quietly.
        with contextlib.suppress(Exception):
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 1
    except OSError as error:
        if os.environ.get("SUBTHIS_DEBUG"):
            raise
        location = f" ({error.filename})" if getattr(error, "filename", None) else ""
        if getattr(error, "winerror", None) in (32, 33):
            print(
                f"subthis: another program has this file open{location}. Close it (a video player,\n"
                "an editor, or a sync client) and try again.",
                file=sys.stderr,
            )
            return 1
        print(
            f"subthis: a file or folder problem stopped the run{location}: {error.strerror or error}",
            file=sys.stderr,
        )
        return 1
    except Exception as error:  # a friendly line beats a traceback for non-technical users
        if os.environ.get("SUBTHIS_DEBUG"):
            raise
        print(
            f"subthis: something unexpected went wrong ({type(error).__name__}: {error}).\n"
            "Please report it at https://github.com/ItayCohen-Prog/subthis/issues\n"
            "(run again with SUBTHIS_DEBUG=1 to see the full details).",
            file=sys.stderr,
        )
        return 1


def _launched_by_double_click() -> bool:
    """True when this console exists only for us (no shell will keep it open)."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        processes = (ctypes.c_uint * 8)()
        count = ctypes.windll.kernel32.GetConsoleProcessList(processes, 8)  # type: ignore[attr-defined]
        return 0 < count <= 2
    except Exception:
        return False


def entrypoint() -> int:
    code = main()
    if _launched_by_double_click():
        # Otherwise the window vanishes before the person can read anything.
        with contextlib.suppress(Exception):
            input("\nPress Enter to close this window...")
    return code


if __name__ == "__main__":
    raise SystemExit(entrypoint())
