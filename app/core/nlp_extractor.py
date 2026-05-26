"""
nlp_extractor.py
----------------
Extracts symptom keywords from free-text user input.

The symptom lexicon is now generated dynamically from the CSV dataset
at initialisation time instead of being hardcoded.

Pipeline:
  1. Read all symptom strings from symptom_disease.csv
  2. Auto-generate synonym variants for each canonical symptom
     (e.g. "burning urination" → ["burn urinating", "burning when peeing", ...])
  3. Apply longest-match extraction with negation detection
"""

import re
import csv
import os
from typing import NamedTuple
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except:
    SPACY_AVAILABLE = False
    nlp = None
try:
    from rapidfuzz import process as fuzz_process, fuzz
    _FUZZY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FUZZY_AVAILABLE = False

# Minimum similarity score (0-100) to accept a fuzzy match
FUZZY_THRESHOLD = 92


# ---------------------------------------------------------------------------
# 1. Synonym generator — turns dataset symptom names into lexicon entries
# ---------------------------------------------------------------------------

_MANUAL_SYNONYMS: dict[str, list[str]] = {
    "headache":               ["headache", "head ache", "head pain", "head hurts", "head is pounding",
                               "head is throbbing", "head hurting", "my head hurts", "cephalgia"],
    "throbbing pain":         ["throbbing", "pulsating", "pounding pain", "pulsing pain"],
    "one-sided pain":         ["one side", "one-sided", "left side of head", "right side of head", "half my head"],
    "pressure around forehead":["pressure", "band around head", "tight head", "squeezing",
                                "pressure in head", "head feels tight"],
    "visual aura":            ["aura", "visual disturbance", "flashing lights", "zigzag lines",
                               "blurred vision before headache", "visual changes"],
    "dizziness":              ["dizzy", "dizziness", "lightheaded", "light-headed", "vertigo",
                               "spinning", "unsteady", "off balance"],
    "sensitivity to light":   ["sensitive to light", "light hurts eyes", "photophobia",
                               "light sensitivity", "bright light painful", "can't stand light"],
    "sensitivity to sound":   ["sensitive to sound", "noise hurts", "phonophobia",
                               "sound sensitivity", "loud noises bother me"],
    "runny nose":             ["runny nose", "runny", "nose is running", "nasal discharge",
                               "nose dripping", "dripping nose"],
    "sneezing":               ["sneezing", "sneeze", "sneezes", "keep sneezing"],
    "congestion":             ["congested", "stuffy nose", "blocked nose", "nasal congestion",
                               "can't breathe through nose", "stuffed up"],
    "sore throat":            ["sore throat", "throat pain", "throat hurts", "throat is sore",
                               "painful swallowing", "throat ache", "scratchy throat"],
    "difficulty swallowing":  ["hard to swallow", "difficulty swallowing", "painful swallowing",
                               "swallowing hurts", "can't swallow"],
    "cough":                  ["cough", "coughing", "dry cough", "wet cough", "chesty cough"],
    "shortness of breath":    ["short of breath", "breathless", "can't breathe",
                               "difficulty breathing", "out of breath", "hard to breathe", "breathing difficulty"],
    "chest pain":             ["chest pain", "chest hurts", "pain in chest", "chest ache", "chest discomfort"],
    "chest tightness":        ["chest tight", "tightness in chest", "chest feels tight", "chest pressure"],
    "heartburn":              ["heartburn", "heart burn", "burning in chest", "chest burning",
                               "acid in throat", "burning sensation chest"],
    "racing heart":           ["racing heart", "heart pounding", "palpitations", "heart beating fast",
                               "heart racing", "rapid heartbeat"],
    "nausea":                 ["nausea", "nauseous", "feel sick", "feeling sick", "want to vomit",
                               "queasy", "stomach feels sick", "feel nauseous"],
    "vomiting":               ["vomiting", "vomited", "threw up", "throwing up", "been sick",
                               "puking", "vomit"],
    "diarrhoea":              ["diarrhoea", "diarrhea", "loose stool", "watery stool", "loose stools",
                               "runny stool", "frequent stool"],
    "stomach cramps":         ["stomach cramps", "stomach pain", "abdominal pain", "belly pain",
                               "stomach ache", "tummy ache", "abdominal cramps", "gut pain"],
    "stomach pain":           ["stomach pain", "abdominal pain", "belly ache", "tummy pain"],
    "burning urination":      ["burning when urinating", "burning urination", "pain when peeing",
                               "stinging urine", "burning pee", "it burns when i pee",
                               "burns when i urinate", "pain urinating"],
    "frequent urination":     ["frequent urination", "urinating often", "need to urinate often",
                               "peeing a lot", "going to toilet often", "urgency to urinate",
                               "going bathroom often", "urinating frequently", "peeing frequently"],
    "body aches":             ["body aches", "muscle aches", "all over aches", "aching",
                               "sore all over", "body pain", "aching body"],
    "back pain":              ["back pain", "backache", "back ache", "lower back pain",
                               "my back hurts", "back is sore", "lumbar pain"],
    "joint pain":             ["joint pain", "joints hurt", "arthralgia", "achy joints"],
    "muscle pain":            ["muscle pain", "muscle soreness", "sore muscles", "muscle ache",
                               "muscles hurt", "myalgia"],
    "fever":                  ["fever", "temperature", "high temperature", "febrile", "feel hot",
                               "running a temperature", "38 degrees", "39 degrees", "feverish",
                               "high fever", "fever and chills"],
    "chills":                 ["chills", "shivering", "shakes", "feeling cold", "rigors", "can't get warm"],
    "fatigue":                ["tired", "fatigue", "exhausted", "no energy", "lethargy", "weak",
                               "feeling run down", "wiped out", "tiredness", "exhaustion"],
    "sweating":               ["sweating", "sweaty", "night sweats", "excessive sweating", "perspiring"],
    "loss of appetite":       ["no appetite", "not hungry", "lost appetite", "don't want to eat",
                               "can't eat", "reduced appetite"],
    "rash":                   ["rash", "skin rash", "red rash", "hives", "itchy rash", "spots on skin"],
    "itchy skin":             ["itchy skin", "skin itch", "skin itching", "itchiness", "pruritus"],
    "dry skin":               ["dry skin", "skin is dry", "flaky skin"],
    "red eyes":               ["red eyes", "pink eye", "bloodshot", "eyes are red", "eye redness"],
    "itchy eyes":             ["itchy eyes", "eye itch", "eyes itch", "itching eyes"],
    "watery eyes":            ["watery eyes", "eyes watering", "tearing", "tears"],
    "blurred vision":         ["blurred vision", "blurry vision", "vision blurred", "fuzzy vision"],
    "thirst":                 ["thirsty", "thirst", "very thirsty", "drinking lots", "increased thirst"],
    "weight loss":            ["losing weight", "weight loss", "lost weight unintentionally", "unexplained weight loss"],
    "weight gain":            ["weight gain", "gaining weight", "putting on weight"],
    "trembling":              ["trembling", "shaking", "tremor", "hands shaking", "shaky"],
    "anxiety":                ["anxious", "anxiety", "feeling anxious", "nervousness", "on edge"],
    "depression":             ["depressed", "depression", "feeling depressed", "low mood", "hopeless"],
    "confusion":              ["confused", "confusion", "disoriented", "not thinking clearly"],
    "spinning sensation":     ["spinning sensation", "room spinning", "world spinning", "feel like spinning"],
    "yellow skin":            ["yellow skin", "yellowing skin", "jaundice", "skin turned yellow"],
    "yellow eyes":            ["yellow eyes", "eyes are yellow", "whites of eyes yellow"],
    "dark urine":             ["dark urine", "brown urine", "cola coloured urine", "dark yellow urine"],
    "pale skin":              ["pale skin", "pallor", "skin looks pale", "washed out"],
    "cold intolerance":       ["cold intolerance", "sensitive to cold", "always cold", "intolerant of cold"],
    "hair loss":              ["hair loss", "losing hair", "hair falling out", "alopecia", "baldness"],
    "constipation":           ["constipated", "constipation", "can't go to toilet", "hard stool"],
    "wheezing":               ["wheezing", "wheeze", "whistling breath", "chest wheezing"],
    "facial pain":            ["facial pain", "face pain", "face hurts", "facial discomfort"],
    "nasal congestion":       ["nasal congestion", "blocked nose", "stuffy nose", "congested nose"],
    "post-nasal drip":        ["post-nasal drip", "drip down throat", "mucus down throat"],
    "productive cough":       ["productive cough", "cough with phlegm", "cough bringing up mucus", "wet cough"],
    "high fever":             ["high fever", "very high temperature", "burning fever", "temperature of 39",
                               "temperature of 40"],
    "severe headache":        ["severe headache", "bad headache", "really bad headache", "excruciating headache"],
    "eye pain":               ["eye pain", "pain in eye", "painful eyes", "pain behind eyes"],
    "skin rash":              ["skin rash", "rash on skin", "red patches on skin", "skin eruption"],
}


def _auto_synonyms(canonical: str) -> list[str]:
    variants = {canonical}
    variants.add(canonical.replace("_", " "))
    if canonical.endswith("s") and len(canonical) > 3:
        variants.add(canonical[:-1])
    return list(variants)


def build_lexicon_from_csv(csv_path: str) -> dict[str, list[str]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    canonical_symptoms: set[str] = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for i in range(1, 18):
                val = row.get(f"symptom_{i}", "").strip().lower()
                if val:
                    canonical_symptoms.add(val)

    lexicon: dict[str, list[str]] = {}

    for canonical, phrases in _MANUAL_SYNONYMS.items():
        lexicon[canonical] = list(dict.fromkeys(p.lower() for p in phrases))

    for sym in canonical_symptoms:
        if sym not in lexicon:
            lexicon[sym] = _auto_synonyms(sym)
        else:
            if sym not in lexicon[sym]:
                lexicon[sym].insert(0, sym)

    return lexicon


# ---------------------------------------------------------------------------
# 2. Extractor class
# ---------------------------------------------------------------------------

class ExtractionResult(NamedTuple):
    symptoms:    list
    raw_mentions: list
    negated:     list
    noise: list


class SymptomExtractor:
    def __init__(self, csv_path: str | None = None):
        if csv_path and os.path.exists(csv_path):
            lexicon = build_lexicon_from_csv(csv_path)
        else:
            lexicon = {k: list(dict.fromkeys(v)) for k, v in _MANUAL_SYNONYMS.items()}

        self.phrase_to_symptom: dict[str, str] = {}
        for canonical, phrases in lexicon.items():
            for phrase in phrases:
                self.phrase_to_symptom[phrase.lower()] = canonical

        self.sorted_phrases = sorted(
            self.phrase_to_symptom.keys(), key=len, reverse=True
        )

        self._all_phrases: list[str] = list(self.phrase_to_symptom.keys())

        self.negation_patterns = re.compile(
            r"\b(no|not|without|don't have|doesn't have|haven't|hasn't|never|"
            r"no sign of|denies|absence of)\b",
            re.IGNORECASE
        )

        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.nlp = None
        else:
            self.nlp = None

        print(f"[NLP] Lexicon loaded: {len(lexicon)} canonical symptoms, "
              f"{len(self.phrase_to_symptom)} total phrases")

    def _is_negated(self, doc, start_char: int, end_char: int) -> bool:
        if not doc:
            return False

        symptom_tokens = [t for t in doc if t.idx >= start_char and t.idx < end_char]
        if not symptom_tokens:
            symptom_tokens = [t for t in doc if t.idx + len(t.text) > start_char and t.idx < end_char]

        negation_words = {"no", "not", "without", "never", "deny", "denies", "absence", "negative"}

        for token in symptom_tokens:
            if any(child.dep_ == "neg" or child.lemma_.lower() in negation_words for child in token.children):
                return True

            curr = token
            while curr != curr.head:
                curr = curr.head
                if curr.lemma_.lower() in negation_words:
                    return True
                if any(child.dep_ == "neg" or child.lemma_.lower() in negation_words for child in curr.children):
                    return True
                if curr.pos_ == "VERB":
                    break

        return False

    def _fuzzy_match_token(self, token: str) -> tuple[str, str] | None:
        if not _FUZZY_AVAILABLE or len(token) < 3:
            return None
        result = fuzz_process.extractOne(
            token,
            self._all_phrases,
            scorer=fuzz.WRatio,
            score_cutoff=FUZZY_THRESHOLD,
        )
        if result is None:
            return None
        matched_phrase, score, _ = result
        return matched_phrase, self.phrase_to_symptom[matched_phrase]

    def extract(self, text: str) -> ExtractionResult:
        text_lower = text.lower()
        doc = self.nlp(text) if self.nlp else None

        found_symptoms:   list[str] = []
        negated_symptoms: list[str] = []
        raw_mentions:     list[str] = []
        matched_positions: set[int] = set()

        for phrase in self.sorted_phrases:
            start = 0
            while True:
                idx = text_lower.find(phrase, start)
                if idx == -1:
                    break
                positions = set(range(idx, idx + len(phrase)))
                if positions & matched_positions:
                    start = idx + 1
                    continue

                canonical = self.phrase_to_symptom[phrase]
                raw_mentions.append(text[idx: idx + len(phrase)])
                matched_positions |= positions

                if self._is_negated(doc, idx, idx + len(phrase)):
                    if canonical not in negated_symptoms:
                        negated_symptoms.append(canonical)
                else:
                    if canonical not in found_symptoms:
                        found_symptoms.append(canonical)

                start = idx + 1

        if _FUZZY_AVAILABLE:
            words = re.findall(r"[a-z']+", text_lower)
            candidates: list[tuple[int, str]] = []
            for n in (3, 2, 1):
                for i in range(len(words) - n + 1):
                    ngram = " ".join(words[i: i + n])
                    approx_pos = text_lower.find(ngram)
                    if approx_pos == -1:
                        continue
                    positions = set(range(approx_pos, approx_pos + len(ngram)))
                    if positions & matched_positions:
                        continue
                    candidates.append((approx_pos, ngram))

            seen_positions: set[int] = set()
            for approx_pos, ngram in candidates:
                positions = set(range(approx_pos, approx_pos + len(ngram)))
                if positions & seen_positions:
                    continue

                hit = self._fuzzy_match_token(ngram)
                if hit is None:
                    continue

                matched_phrase, canonical = hit
                seen_positions |= positions
                matched_positions |= positions
                raw_mentions.append(ngram)

                if self._is_negated(doc, approx_pos, approx_pos + len(ngram)):
                    if canonical not in negated_symptoms:
                        negated_symptoms.append(canonical)
                else:
                    if canonical not in found_symptoms:
                        found_symptoms.append(canonical)

        STOPWORDS = {"and", "or", "the", "a", "i", "have", "has", "had"}

        input_words = set(
            word for word in re.findall(r"[a-z']+", text_lower)
            if word not in STOPWORDS
        )

        cleaned_symptoms = []
        for symptom in found_symptoms:
            symptom_words = set(symptom.split())
            if symptom_words & input_words or len(symptom_words) == 1:
                cleaned_symptoms.append(symptom)

        original_words = set(re.findall(r"[a-z']+", text_lower))
        matched_words = set()
        for phrase in raw_mentions:
            matched_words.update(phrase.lower().split())

        noise_words = [
            word for word in (original_words - matched_words)
            if word not in STOPWORDS and len(word) > 3
        ]

        found_symptoms = cleaned_symptoms
        return ExtractionResult(
            symptoms=found_symptoms,
            raw_mentions=raw_mentions,
            negated=negated_symptoms,
            noise=list(noise_words)
        )


# ---------------------------------------------------------------------------
# 3. Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pathlib
    _here = pathlib.Path(__file__).parent.parent.parent
    csv_p = str(_here / "data" / "symptom_disease.csv")

    extractor = SymptomExtractor(csv_path=csv_p)

    print("\n" + "="*40)
    print("      SYMPTOM EXTRACTION TESTER")
    print("="*40)
    print("Type your symptoms (or 'quit' to exit)")

    while True:
        try:
            t = input("\n[Input]: ").strip()
            if not t or t.lower() in ["quit", "exit", "q"]:
                print("Exiting...")
                break

            r = extractor.extract(t)
            print(f"  [Found]:   {r.symptoms}")
            if r.negated:
                print(f"  [Negated]: {r.negated}")
            if not r.symptoms and not r.negated:
                print("  [Result]:  No symptoms detected.")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
