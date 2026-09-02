from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).parents[1] / "subthis.py"
SPEC = importlib.util.spec_from_file_location("subthis", MODULE_PATH)
subthis = importlib.util.module_from_spec(SPEC)
sys.modules["subthis"] = subthis
assert SPEC.loader is not None
SPEC.loader.exec_module(subthis)


class CanonicalTermsTests(unittest.TestCase):
    def test_replaces_hebrew_and_english_aliases_with_canonical_terms(self) -> None:
        text = "בדקתי אופן איי איי וגם קלוד ואז chat gpt"

        actual = subthis.canonicalize_terms(text, subthis.DEFAULT_ALIASES)

        self.assertEqual(actual, "בדקתי OpenAI וגם Claude ואז ChatGPT")

    def test_does_not_replace_alias_inside_another_word(self) -> None:
        actual = subthis.canonicalize_terms("הענן cloudiness", subthis.DEFAULT_ALIASES)

        self.assertEqual(actual, "הענן cloudiness")

    def test_does_not_turn_the_real_word_cloud_into_claude(self) -> None:
        actual = subthis.canonicalize_terms("cloud computing", subthis.DEFAULT_ALIASES)

        self.assertEqual(actual, "cloud computing")


class AlignmentTests(unittest.TestCase):
    def test_transfers_timing_when_accurate_model_merges_phonetic_words(self) -> None:
        timed = [
            subthis.TimedWord("אנחנו", 0.0, 0.3),
            subthis.TimedWord("עובדים", 0.3, 0.7),
            subthis.TimedWord("עם", 0.7, 0.9),
            subthis.TimedWord("אופן", 0.9, 1.1),
            subthis.TimedWord("איי", 1.1, 1.3),
            subthis.TimedWord("איי", 1.3, 1.5),
            subthis.TimedWord("היום", 1.5, 1.9),
        ]

        actual = subthis.align_accurate_words("אנחנו עובדים עם OpenAI היום", timed)

        self.assertEqual([word.text for word in actual], ["אנחנו", "עובדים", "עם", "OpenAI", "היום"])
        self.assertAlmostEqual(actual[3].start, 0.9)
        self.assertAlmostEqual(actual[3].end, 1.5)
        self.assertAlmostEqual(actual[4].start, 1.5)

    def test_alignment_remains_monotonic_when_no_tokens_match(self) -> None:
        timed = [
            subthis.TimedWord("one", 2.0, 2.4),
            subthis.TimedWord("two", 2.5, 3.0),
        ]

        actual = subthis.align_accurate_words("שלום עולם חדש", timed)

        self.assertEqual(len(actual), 3)
        self.assertEqual(actual[0].start, 2.0)
        self.assertEqual(actual[-1].end, 3.0)
        self.assertTrue(all(a.end <= b.start for a, b in zip(actual, actual[1:])))


class CueTests(unittest.TestCase):
    def test_removes_all_unicode_punctuation_from_caption_text(self) -> None:
        actual = subthis.strip_caption_punctuation(
            "שלום, OpenAI! מה נשמע? Next.js — כן. צ׳אט-בוט"
        )

        self.assertEqual(actual, "שלום OpenAI מה נשמע Nextjs כן צאטבוט")

    def test_groups_at_most_three_words_and_holds_cue_across_pause(self) -> None:
        words = [
            subthis.TimedWord("אחד", 0.0, 0.2),
            subthis.TimedWord("שתיים", 0.25, 0.5),
            subthis.TimedWord("שלוש", 0.55, 0.8),
            subthis.TimedWord("ארבע", 3.0, 3.2),
            subthis.TimedWord("חמש", 3.25, 3.5),
            subthis.TimedWord("שש", 3.55, 3.8),
            subthis.TimedWord("שבע", 4.0, 4.3),
        ]

        cues = subthis.make_cues(words, media_end=10.0, max_words=3)

        self.assertEqual([cue.text for cue in cues], ["אחד שתיים שלוש", "ארבע חמש שש", "שבע"])
        self.assertEqual(cues[0].end, 3.0)
        self.assertEqual(cues[1].end, 4.0)
        self.assertEqual(cues[2].end, 4.8)
        self.assertTrue(all(len(cue.text.split()) <= 3 for cue in cues))

    def test_make_cues_never_emits_punctuation(self) -> None:
        words = [
            subthis.TimedWord("שלום,", 0.0, 0.3),
            subthis.TimedWord("OpenAI!", 0.4, 0.8),
            subthis.TimedWord("באמת?", 0.9, 1.2),
        ]

        cues = subthis.make_cues(words, media_end=2.0, max_words=3)

        self.assertEqual(cues[0].text, "שלום OpenAI באמת")

    def test_srt_rounds_milliseconds_without_overlapping(self) -> None:
        cues = [
            subthis.Cue(0.0, 1.2346, "שלום OpenAI"),
            subthis.Cue(1.2346, 2.0, "מה נשמע"),
        ]

        actual = subthis.render_srt(cues)

        self.assertEqual(
            actual,
            "1\n00:00:00,000 --> 00:00:01,235\nשלום OpenAI\n\n"
            "2\n00:00:01,235 --> 00:00:02,000\nמה נשמע\n",
        )


class ChunkMergeTests(unittest.TestCase):
    def test_discards_duplicate_words_from_overlapping_chunks(self) -> None:
        first = [
            subthis.TimedWord("hello", 0.0, 0.4),
            subthis.TimedWord("OpenAI", 0.5, 1.0),
        ]
        second = [
            subthis.TimedWord("OpenAI", 0.5, 1.0),
            subthis.TimedWord("again", 1.1, 1.5),
        ]

        actual = subthis.merge_chunk_words([first, second])

        self.assertEqual([word.text for word in actual], ["hello", "OpenAI", "again"])


class ConfigDirTests(unittest.TestCase):
    def test_uses_xdg_config_home_when_set(self) -> None:
        with mock.patch.object(subthis.sys, "platform", "linux"), mock.patch.dict(
            os.environ, {"XDG_CONFIG_HOME": "/custom/config"}
        ):
            self.assertEqual(subthis._config_dir(), Path("/custom/config/subthis"))

    def test_defaults_to_dot_config_without_xdg(self) -> None:
        with mock.patch.object(subthis.sys, "platform", "linux"), mock.patch.dict(
            os.environ, {"XDG_CONFIG_HOME": ""}
        ):
            self.assertEqual(subthis._config_dir(), Path.home() / ".config" / "subthis")

    def test_uses_appdata_on_windows(self) -> None:
        with mock.patch.object(subthis.sys, "platform", "win32"), mock.patch.dict(
            os.environ, {"APPDATA": r"C:\Users\bram\AppData\Roaming"}
        ):
            actual = subthis._config_dir()
        self.assertEqual(actual.name, "subthis")
        self.assertIn("AppData", str(actual))


class HelpTests(unittest.TestCase):
    def test_bare_invocation_and_help_word_show_help_and_exit_zero(self) -> None:
        for arguments in ([], ["help"]):
            with mock.patch("sys.stdout") as stdout:
                self.assertEqual(subthis.run(arguments), 0)
            self.assertTrue(stdout.write.called)

    def test_epilog_mentions_only_the_current_platform(self) -> None:
        cases = {
            "win32": ("powershell", ["Cmd+Space", "Finder"]),
            "darwin": ("Cmd+Space", ["powershell", "File Explorer"]),
            "linux": ("terminal application", ["powershell", "Cmd+Space"]),
        }
        for platform, (expected, absent) in cases.items():
            with mock.patch.object(subthis.sys, "platform", platform):
                epilog = subthis._help_epilog()
            self.assertIn(expected, epilog, platform)
            for text in absent:
                self.assertNotIn(text, epilog, platform)


class SetupDispatchTests(unittest.TestCase):
    def test_setup_rejects_extra_arguments(self) -> None:
        with self.assertRaises(subthis.SubthisError):
            subthis.run(["setup", "extra"])

    def test_setup_dispatches_to_run_setup(self) -> None:
        with mock.patch.object(subthis, "run_setup", return_value=0) as run_setup:
            self.assertEqual(subthis.run(["setup"]), 0)
        run_setup.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
