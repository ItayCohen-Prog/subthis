#!/usr/bin/env python3
"""Create short, accurately worded SRT captions from a video."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import difflib
import getpass
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Sequence


__version__ = "1.1.0"

API_URL = "https://api.openai.com/v1/audio/transcriptions"
KEY_CHECK_URL = "https://api.openai.com/v1/models/whisper-1"


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
    prefix = f"OpenAI API error{f' {status}' if status else ''}"
    try:
        decoded = json.loads(payload.decode("utf-8", errors="replace"))
        message = decoded.get("error", {}).get("message")
        if isinstance(message, str) and message.strip():
            return f"{prefix}: {message.strip()}"
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


def _verify_api_key(api_key: str) -> str | None:
    """Return a warning message when verification was inconclusive; raise on a bad key."""
    request = urllib.request.Request(
        KEY_CHECK_URL,
        headers={"Authorization": f"Bearer {api_key}", "User-Agent": "subthis/1.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30):
            return None
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            raise SubthisError(
                "OpenAI rejected this API key. Check it at https://platform.openai.com/api-keys"
            ) from error
        return f"could not verify the key (HTTP {error.code}); saved it anyway"
    except urllib.error.URLError as error:
        reason = getattr(error, "reason", "connection failed")
        return f"could not reach OpenAI to verify the key ({reason}); saved it anyway"


def _write_env_file(api_key: str) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        ENV_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(f"OPENAI_API_KEY={api_key}\n")
    with contextlib.suppress(OSError):
        os.chmod(ENV_FILE, 0o600)


def run_setup() -> int:
    print(f"subthis {__version__} setup\n")

    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        print("ffmpeg: found")
    else:
        print("ffmpeg: NOT FOUND — subthis needs ffmpeg and ffprobe to read video.")
        print(f"        To install: {_ffmpeg_install_hint()}")

    existing_key = ""
    with contextlib.suppress(SubthisError):
        existing_key = _load_api_key()
    prompt = (
        "OpenAI API key (Enter keeps the saved key): "
        if existing_key
        else "OpenAI API key: "
    )
    try:
        entered = getpass.getpass(prompt).strip()
    except EOFError:
        entered = ""
    api_key = entered or existing_key
    if not api_key:
        raise SubthisError("No API key entered. Get one at https://platform.openai.com/api-keys")

    warning = _verify_api_key(api_key)
    if warning:
        print(f"Warning: {warning}", file=sys.stderr)
    else:
        print("API key verified with OpenAI.")

    _write_env_file(api_key)
    print(f"Saved API key to {ENV_FILE}")

    if not TERMS_FILE.exists():
        TERMS_FILE.write_text(TERMS_TEMPLATE, encoding="utf-8")
        print(f"Created terms file at {TERMS_FILE}")
    else:
        print(f"Keeping existing terms file at {TERMS_FILE}")

    print("\nSetup complete. Try: subthis video.mp4")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subthis",
        description="Create accurate Hebrew/English SRT captions with at most three words per cue.",
        epilog="Run 'subthis setup' once after installing to store your OpenAI API key.",
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
    if arguments and arguments[0] == "setup":
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
