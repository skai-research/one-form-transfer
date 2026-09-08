"""Add IPA and romanized versions of text columns to a Hugging Face dataset.

IPA is produced with Phonemizer (eSpeak NG backend). Words containing digits or letters
from a script other than the language's own are passed through untranscribed, and
stress, length and other diacritics are removed. Romanization uses uroman.
"""
import argparse
import os
import re
import unicodedata

from datasets import load_dataset, load_from_disk
from phonemizer.backend.espeak.espeak import EspeakBackend
from phonemizer.punctuation import Punctuation

ESPEAK_VOICE = {"en": "en-us", "es": "es", "ru": "ru", "pl": "pl", "hi": "hi", "ur": "ur",
                "ta": "ta", "ml": "ml", "ar": "ar", "fr": "fr-fr", "bn": "bn", "el": "el"}
ALIASES = {"eng": "en", "english": "en", "spa": "es", "spanish": "es", "rus": "ru", "russian": "ru",
           "pol": "pl", "polish": "pl", "hin": "hi", "hindi": "hi", "urd": "ur", "urdu": "ur",
           "tam": "ta", "tamil": "ta", "mal": "ml", "malayalam": "ml", "ara": "ar", "arabic": "ar",
           "fra": "fr", "french": "fr", "ben": "bn", "bengali": "bn", "ell": "el", "greek": "el"}
SCRIPT = {"en": "latin", "es": "latin", "fr": "latin", "pl": "latin", "ru": "cyrillic",
          "el": "greek", "ar": "arabic", "ur": "arabic", "hi": "devanagari", "bn": "bengali",
          "ta": "tamil", "ml": "malayalam"}
SCRIPT_RANGES = {
    "latin": [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F), (0x1E00, 0x1EFF)],
    "cyrillic": [(0x0400, 0x052F), (0x1C80, 0x1C8F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)],
    "greek": [(0x0370, 0x03FF), (0x1F00, 0x1FFF)],
    "arabic": [(0x0600, 0x06FF), (0x0750, 0x077F), (0x0870, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "devanagari": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    "bengali": [(0x0980, 0x09FF)],
    "tamil": [(0x0B80, 0x0BFF)],
    "malayalam": [(0x0D00, 0x0D7F)],
}
PUNCTUATION = Punctuation.default_marks() + "।॥"
PLACEHOLDER_CHARS = ",`~"


def language_code(name):
    key = name.lower().replace("_", "-").split("-")[0]
    code = ALIASES.get(key, key)
    if code not in ESPEAK_VOICE:
        raise ValueError(f"unsupported language {name!r}; known: {sorted(ESPEAK_VOICE)}")
    return code


def strip_ipa(text):
    for mark in "ˈˌːˑ":
        text = text.replace(mark, "")
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def placeholder(i):
    base, out = len(PLACEHOLDER_CHARS), []
    while True:
        out.append(PLACEHOLDER_CHARS[i % base])
        i = i // base - 1
        if i < 0:
            return "".join(reversed(out))


class PreservingEspeakBackend(EspeakBackend):
    """eSpeak backend that passes words containing digits or letters of a foreign script
    through the transcription unchanged."""

    def __init__(self, code):
        self.ranges = SCRIPT_RANGES[SCRIPT[code]]
        super().__init__(ESPEAK_VOICE[code], punctuation_marks=PUNCTUATION + "<>/" + PLACEHOLDER_CHARS,
                         preserve_punctuation=True, with_stress=True, language_switch="keep-flags",
                         words_mismatch="ignore")

    def _preserve(self, word):
        for ch in word:
            cat = unicodedata.category(ch)
            if cat == "Nd" or (cat[0] == "L" and not any(lo <= ord(ch) <= hi for lo, hi in self.ranges)):
                return True
        return False

    def transcribe(self, text):
        mapping = {}

        def shield(match):
            word = match.group(0)
            if not self._preserve(word):
                return word
            tag = mapping.setdefault(word, placeholder(len(mapping)))
            return f"<<<{tag}> {word} </{tag}>>>"

        out = self.phonemize([re.sub(r"\S+", shield, text)])
        out = out[0] if out else ""  # the backend returns nothing for empty input
        for word, tag in mapping.items():
            prefix, suffix = f"<<<{tag}>", f"</{tag}>>>"
            if prefix not in out or suffix not in out:
                raise RuntimeError(f"placeholder {tag!r} lost while transcribing {text[:80]!r}")
            out = re.sub(re.escape(prefix) + ".*?" + re.escape(suffix), lambda _: word, out)
        return out


_backends, _romanizer = {}, None


def to_ipa(text, code):
    if code not in _backends:
        _backends[code] = PreservingEspeakBackend(code)
    return strip_ipa(_backends[code].transcribe(text))


def to_romanized(text):
    global _romanizer
    if _romanizer is None:
        import uroman
        _romanizer = uroman.Uroman()
    return _romanizer.romanize_string(text)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="Hub id or a local save_to_disk directory")
    p.add_argument("--config", default=None)
    p.add_argument("--splits", nargs="+", default=None, help="default: all splits")
    p.add_argument("--columns", nargs="+", required=True, help="text columns to convert")
    lang = p.add_mutually_exclusive_group(required=True)
    lang.add_argument("--language", help="language of the whole dataset, e.g. en, hi, ur-PK, tamil")
    lang.add_argument("--language-column", help="column holding a per-row language code")
    p.add_argument("--output", required=True, help="directory for save_to_disk")
    p.add_argument("--push", default=None, help="optionally push to this Hub repo (config = --config)")
    p.add_argument("--num-proc", type=int, default=os.cpu_count())
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    ds = load_from_disk(args.dataset) if os.path.isdir(args.dataset) else load_dataset(args.dataset, args.config)
    if args.splits:
        ds = type(ds)({s: ds[s] for s in args.splits})
    fixed = language_code(args.language) if args.language else None

    def convert(batch):
        n = len(batch[args.columns[0]])
        codes = [fixed] * n if fixed else [language_code(x) for x in batch[args.language_column]]
        out = {}
        for col in args.columns:
            texts = [t or "" for t in batch[col]]
            out[f"{col}_ipa_stripped"] = [to_ipa(t, c) for t, c in zip(texts, codes)]
            out[f"{col}_romanized"] = [to_romanized(t) for t in texts]
        return out

    ds = ds.map(convert, batched=True, batch_size=args.batch_size, num_proc=args.num_proc,
                desc="transcribing")
    ds.save_to_disk(args.output)
    if args.push:
        ds.push_to_hub(args.push, config_name=args.config or "default")


if __name__ == "__main__":
    main()
