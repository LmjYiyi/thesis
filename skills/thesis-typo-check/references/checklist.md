# Thesis Typo Check Checklist

Use this checklist when proofreading a chapter, abstract, or review response with a typo-focused scope.

## High-Frequency Checks

### Character-Level Errors

- Wrong homophone or near-sound character.
- Wrong near-shape character.
- Missing character in fixed phrases or technical expressions.
- Repeated character or repeated short phrase.
- Missing measure word, particle, or conjunction that breaks the sentence.

### Punctuation and Pairing

- Mismatched `()`, `""`, `<>`, `[]`.
- Duplicate punctuation such as `..`, `,,`, `;;`.
- Full-width and half-width punctuation mixed in one local phrase.
- Broken enumerations with inconsistent marks.

### Mixed Chinese-English Text

- Missing or inconsistent spacing around English abbreviations and units when the local style requires spacing.
- Inconsistent case such as `matlab` vs `MATLAB`.
- Inconsistent hyphenation such as `Metropolis Hastings` vs `Metropolis-Hastings`.
- English plural or article errors only when clearly local and mechanical.

### Thesis-Specific Mechanical Issues

- Chapter or section titles with missing words.
- Figure/table references with obviously broken wording.
- Repeated transition phrases caused by patch edits.
- Review replies where one sentence accidentally copies the previous one.
- Markdown text where list bullets or headings swallow the first character of the sentence after editing.

## Domain Terminology Guardrails

Prefer these forms unless the source material clearly establishes another house style:

- `LFMCW`
- `Drude`
- `Lorentz`
- `Metropolis-Hastings`
- `Levenberg-Marquardt`
- `ESPRIT`
- `MATLAB`
- `Word`
- `Markdown`
- `GHz`, `MHz`, `kHz`, `dB`

Do not "correct" the following without evidence:

- Variable symbols such as `n_e`, `f_p`, `omega`, `nu_e`
- Equation numbers, citation labels, and cross-references
- File names, script names, and command lines
- Deliberate capitalization inside formulas or code snippets

## Review Heuristics

Use `certain` when:

- The original text contains an obvious wrong character.
- A duplicated or missing character is clear from the sentence itself.
- Punctuation pairing is visibly broken.
- A technical term conflicts with a nearby repeated use in the same section.

Use `suspected` when:

- The sentence may be missing a word, but multiple insertions are possible.
- A term looks inconsistent, but the global house style is unknown.
- The phrase may be awkward rather than wrong.

## Recommended Review Order

1. Scan titles, headings, captions, and opening sentences first.
2. Scan around recent rewrites, merged paragraphs, and copied review responses.
3. Scan mixed Chinese-English phrases and terminology.
4. Scan punctuation and paired symbols last.

## Suggested Report Style

When there are only a few issues, use numbered items:

```text
1. [Chapter 3, paragraph 2]
   Original: ...
   Suggestion: ...
   Reason: repeated character / wrong character / punctuation mismatch
   Confidence: high
```

When there are many issues, use a compact table:

```text
Location | Original | Suggestion | Reason | Confidence
```
