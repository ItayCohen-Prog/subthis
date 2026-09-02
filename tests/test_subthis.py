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
    def test_removes_punctuation_but_keeps_in_word_geresh_and_apostrophes(self) -> None:
        actual = subthis.strip_caption_punctuation(
            "שלום, OpenAI! מה נשמע? Next.js — כן. צ׳אט-בוט"
        )

        self.assertEqual(actual, "שלום OpenAI מה נשמע Nextjs כן צ׳אטבוט")
        self.assertEqual(subthis.strip_caption_punctuation("ג'מיני, don't!"), "ג'מיני don't")

    def test_splits_at_pauses_balances_groups_and_never_bridges_silence(self) -> None:
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

        # the 2.2s silence starts a new phrase, and the 4-word phrase balances
        # into 2+2 instead of leaving a lone orphan word
        self.assertEqual([cue.text for cue in cues], ["אחד שתיים שלוש", "ארבע חמש", "שש שבע"])
        self.assertAlmostEqual(cues[0].end, 1.3)  # hang cap, not lingering to 3.0
        self.assertAlmostEqual(cues[1].end, 3.47)
        self.assertAlmostEqual(cues[2].end, 4.8)
        self.assertTrue(all(len(cue.text.split()) <= 3 for cue in cues))

    def test_short_cue_borrows_time_to_reach_minimum_duration(self) -> None:
        words = [subthis.TimedWord("רגע", 1.0, 1.2)]

        cues = subthis.make_cues(words, media_end=10.0, max_words=3)

        self.assertAlmostEqual(cues[0].end - cues[0].start, subthis.MIN_CUE_SECONDS)

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
            "1\n00:00:00,000 --> 00:00:01,235\n‏שלום OpenAI\n\n"
            "2\n00:00:01,235 --> 00:00:02,000\n‏מה נשמע\n",
        )

    def test_rtl_mark_prefixes_hebrew_cues_but_not_english_ones(self) -> None:
        rendered = subthis.render_srt(
            [
                subthis.Cue(0.0, 1.0, "Claude עושה דברים"),
                subthis.Cue(1.0, 2.0, "pure English here"),
            ]
        )

        lines = rendered.splitlines()
        self.assertTrue(lines[2].startswith("‏"))
        self.assertEqual(lines[6], "pure English here")


class CaptionSettingsTests(unittest.TestCase):
    def test_hold_through_silence_keeps_cue_up_until_the_next_one(self) -> None:
        words = [
            subthis.TimedWord("לפני", 0.0, 0.4),
            subthis.TimedWord("אחרי", 5.0, 5.4),
        ]

        cues = subthis.make_cues(words, media_end=10.0, hold_through_silence=True)

        self.assertAlmostEqual(cues[0].end, 5.0 - subthis.CUE_GAP_SECONDS)

    def test_keep_punctuation_passes_it_through_to_the_srt(self) -> None:
        words = [subthis.TimedWord("שלום,", 0.0, 0.5), subthis.TimedWord("OpenAI!", 0.6, 1.1)]

        cues = subthis.make_cues(words, media_end=2.0, keep_punctuation=True)
        rendered = subthis.render_srt(cues, keep_punctuation=True)

        self.assertIn("שלום, OpenAI!", rendered)

    def test_config_captions_saves_validates_and_resets(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(subthis, "CONFIG_DIR", Path(tmp)), mock.patch.object(
                subthis, "SETTINGS_FILE", Path(tmp) / "settings.json"
            ):
                self.assertEqual(subthis.run_config(["captions", "words", "2"]), 0)
                self.assertEqual(subthis.run_config(["captions", "silence", "hold"]), 0)
                merged = subthis._caption_settings()
                self.assertEqual(merged["words"], 2)
                self.assertEqual(merged["silence"], "hold")
                self.assertEqual(merged["pause"], subthis.PAUSE_SPLIT_SECONDS)

                with self.assertRaises(subthis.SubthisError):
                    subthis.run_config(["captions", "words", "9"])
                with self.assertRaises(subthis.SubthisError):
                    subthis.run_config(["captions", "punctuation", "maybe"])
                with self.assertRaises(subthis.SubthisError):
                    subthis.run_config(["captions", "pause", "0"])

                self.assertEqual(subthis.run_config(["captions", "reset"]), 0)
                self.assertEqual(subthis._caption_settings(), subthis.CAPTION_DEFAULTS)


class TimedWordCanonicalizationTests(unittest.TestCase):
    def test_merges_multiword_alias_run_into_one_anchor_with_combined_timing(self) -> None:
        words = [
            subthis.TimedWord("עם", 0.5, 0.7),
            subthis.TimedWord("אופן", 0.9, 1.1),
            subthis.TimedWord("איי", 1.1, 1.3),
            subthis.TimedWord("איי", 1.3, 1.5),
            subthis.TimedWord("היום", 1.5, 1.9),
        ]

        actual = subthis.canonicalize_timed_words(words, subthis.DEFAULT_ALIASES)

        self.assertEqual([w.text for w in actual], ["עם", "OpenAI", "היום"])
        self.assertAlmostEqual(actual[1].start, 0.9)
        self.assertAlmostEqual(actual[1].end, 1.5)

    def test_single_word_alias_becomes_canonical(self) -> None:
        actual = subthis.canonicalize_timed_words(
            [subthis.TimedWord("קלוד", 2.0, 2.4)], subthis.DEFAULT_ALIASES
        )

        self.assertEqual(actual[0].text, "Claude")


class ReplaceAlignmentTests(unittest.TestCase):
    def test_equal_length_replace_keeps_real_word_timings(self) -> None:
        timed = [
            subthis.TimedWord("עשרים", 1.0, 1.4),
            subthis.TimedWord("וחמש", 1.6, 2.0),
        ]

        actual = subthis.align_accurate_words("25 שקלים", timed)

        self.assertEqual([w.text for w in actual], ["25", "שקלים"])
        self.assertAlmostEqual(actual[0].start, 1.0)
        self.assertAlmostEqual(actual[0].end, 1.4)
        self.assertAlmostEqual(actual[1].start, 1.6)
        self.assertAlmostEqual(actual[1].end, 2.0)


class NonSpeechFilterTests(unittest.TestCase):
    def test_drops_words_inside_flagged_segments_and_keeps_the_rest(self) -> None:
        words = [
            subthis.TimedWord("אמיתי", 1.0, 1.4),
            subthis.TimedWord("הזיה", 6.0, 6.4),
        ]
        segments = [
            {"start": 0.0, "end": 4.0, "no_speech_prob": 0.1, "avg_logprob": -0.3},
            {"start": 5.0, "end": 8.0, "no_speech_prob": 0.9, "avg_logprob": -1.6},
        ]

        actual = subthis.filter_non_speech_words(words, segments)

        self.assertEqual([w.text for w in actual], ["אמיתי"])

    def test_missing_or_malformed_segments_keep_all_words(self) -> None:
        words = [subthis.TimedWord("מילה", 0.0, 0.5)]

        self.assertEqual(subthis.filter_non_speech_words(words, None), words)
        self.assertEqual(subthis.filter_non_speech_words(words, ["junk", 5]), words)


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


class ApiErrorMessageTests(unittest.TestCase):
    def test_401_names_a_revoked_key_and_points_at_setup(self) -> None:
        message = subthis._api_error_message(b'{"error":{"message":"bad"}}', 401)

        self.assertIn("revoked", message)
        self.assertIn("subthis setup", message)
        self.assertIn(subthis.API_KEYS_URL, message)

    def test_insufficient_quota_points_at_billing(self) -> None:
        payload = b'{"error":{"message":"...","code":"insufficient_quota"}}'

        message = subthis._api_error_message(payload, 429)

        self.assertIn("out of credit", message)
        self.assertIn(subthis.BILLING_URL, message)

    def test_other_errors_keep_the_api_message(self) -> None:
        message = subthis._api_error_message(b'{"error":{"message":"boom"}}', 500)

        self.assertIn("boom", message)


class PromptForKeyTests(unittest.TestCase):
    def test_rejects_invalid_key_then_accepts_a_working_one(self) -> None:
        with mock.patch.object(
            subthis, "_classify_key", side_effect=[("invalid", ""), ("ok", "")]
        ), mock.patch.object(subthis, "_ask", side_effect=["sk-bad", "sk-good"]), mock.patch.object(
            subthis, "_open_page"
        ):
            self.assertEqual(subthis._prompt_for_working_key("", True), "sk-good")

    def test_gives_up_without_a_key_when_not_interactive(self) -> None:
        with mock.patch.object(subthis, "_ask", return_value=""):
            with self.assertRaises(subthis.SubthisError):
                subthis._prompt_for_working_key("", False)

    def test_no_credit_aborts_when_not_interactive(self) -> None:
        with mock.patch.object(
            subthis, "_classify_key", return_value=("no_credit", "")
        ), mock.patch.object(subthis, "_ask", return_value="sk-real"), mock.patch.object(
            subthis, "_open_page"
        ):
            with self.assertRaises(subthis.SubthisError) as caught:
                subthis._prompt_for_working_key("", False)
        self.assertIn(subthis.BILLING_URL, str(caught.exception))


class TermParsingTests(unittest.TestCase):
    def test_splits_on_spaces_and_groups_single_quoted_phrases(self) -> None:
        actual = subthis._parse_term_string("OpenAI 'API Platform' ChatGPT")

        self.assertEqual(actual, ["OpenAI", "API Platform", "ChatGPT"])

    def test_unclosed_quote_raises_a_clear_error(self) -> None:
        with self.assertRaises(subthis.SubthisError) as caught:
            subthis._parse_term_string("OpenAI 'API Platform")
        self.assertIn("quote", str(caught.exception))


class UpdateOfferTests(unittest.TestCase):
    def test_version_tuple_orders_versions(self) -> None:
        self.assertLess(subthis._version_tuple("1.4.0"), subthis._version_tuple("1.10.0"))
        self.assertLessEqual(subthis._version_tuple("1.4.0"), subthis._version_tuple("1.4.0"))

    def test_skip_env_var_prevents_any_network_call(self) -> None:
        with mock.patch.dict(os.environ, {"SUBTHIS_SKIP_UPDATE": "1"}), mock.patch.object(
            subthis, "_latest_pypi_version"
        ) as fetch:
            subthis._maybe_offer_update(["video.mp4"])
        fetch.assert_not_called()

    def test_declining_prints_the_update_command_and_continues(self) -> None:
        tty = mock.Mock()
        tty.isatty.return_value = True
        with mock.patch.object(subthis.sys, "stdin", tty), mock.patch.object(
            subthis.sys, "stdout", tty
        ), mock.patch.object(
            subthis, "_latest_pypi_version", return_value="99.0.0"
        ), mock.patch.object(subthis, "_ask", return_value="n"), mock.patch(
            "builtins.print"
        ) as printed:
            subthis._maybe_offer_update(["video.mp4"])
        output = "\n".join(str(call) for call in printed.call_args_list)
        self.assertIn(subthis._update_command(), output)


class ConfigCommandTests(unittest.TestCase):
    def test_config_key_refuses_a_key_on_the_command_line(self) -> None:
        with self.assertRaises(subthis.SubthisError):
            subthis.run_config(["key", "sk-something"])

    def test_config_terms_appends_parsed_terms_to_the_global_file(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            terms_file = Path(tmp) / "terms.txt"
            with mock.patch.object(subthis, "CONFIG_DIR", Path(tmp)), mock.patch.object(
                subthis, "TERMS_FILE", terms_file
            ):
                self.assertEqual(subthis.run_config(["terms", "OpenAI 'API Platform'"]), 0)
            content = terms_file.read_text(encoding="utf-8")
        self.assertIn("OpenAI\n", content)
        self.assertIn("API Platform\n", content)

    def test_config_without_subcommand_shows_usage(self) -> None:
        with self.assertRaises(subthis.SubthisError) as caught:
            subthis.run_config([])
        self.assertIn("subthis config key", str(caught.exception))


class SettingsAndRevealTests(unittest.TestCase):
    def test_config_open_saves_and_reports_the_setting(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(subthis, "CONFIG_DIR", Path(tmp)), mock.patch.object(
                subthis, "SETTINGS_FILE", Path(tmp) / "settings.json"
            ):
                self.assertEqual(subthis.run_config(["open", "on"]), 0)
                self.assertTrue(subthis._load_settings()["open_when_done"])
                self.assertEqual(subthis.run_config(["open", "off"]), 0)
                self.assertFalse(subthis._load_settings()["open_when_done"])

    def test_reveal_never_raises_even_when_everything_fails(self) -> None:
        with mock.patch.object(subthis.subprocess, "run", side_effect=OSError("boom")):
            subthis._reveal_in_file_manager(Path("/nonexistent/file.srt"))

    def test_entry_indexes_skip_comments_and_blank_lines(self) -> None:
        lines = ["# comment", "", "OpenAI", "  ", "Wispr Flow = alias", "# more"]

        self.assertEqual(subthis._entry_indexes(lines), [2, 4])


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
