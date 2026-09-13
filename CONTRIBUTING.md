# Contributing to the Universal Quantum Seed

Thank you for your interest in contributing! The most valuable contributions are improving word coverage across languages: adding more synonyms, shorter alternatives, regional variants, and colloquial terms so that seed recovery is as intuitive as possible for users worldwide.

## How You Can Help

### 1. Verify Existing Words

Check that the words in your language file are correct, commonly understood, and unambiguous. If a word is wrong or could confuse users, open an issue or submit a fix.

### 2. Add More Synonyms and Abbreviations

Each visual position should have multiple accepted words. The more, the better. Good additions include:

- **Synonyms**: different words for the same concept (e.g., "dog", "puppy", "hound")
- **Short forms**: abbreviations or informal terms people actually use in daily speech
- **Regional variants**: words from different dialects or regions (e.g., Latin American vs European Spanish)
- **Colloquial terms**: slang, casual language, commonly borrowed words from other languages
- **Plural forms**: if natural in the language

### 3. Add New Languages

If your language isn't in the [42 currently supported languages](README.md#supported-languages), you can add it by creating a new language file.

## Language File Format

Each language file lives in `languages/` and follows this format:

```python
LABEL = "English"

SEED_WORDS = {
    0: ['eye', 'eyes', 'sight', 'vision'],
    1: ['ear', 'ears', 'hearing'],
    2: ['nose', 'nostrils', 'snout'],
    ...
    255: ['clock', 'clocks', 'timer', 'watch', 'hour'],
}
```

### Rules

- **Keys**: Visual index `0` through `255` (all 256 must be present)
- **Values**: List of accepted words, lowercase
- **First word** in each list is the primary/display word for that language
- **No duplicates** within a single index
- Words must be **real words** that a native speaker would recognize
- Keep words **as short as possible**: aim for the shortest natural word first

## How to Submit Changes

### Quick Edits (GitHub Web)

1. Navigate to the language file in `languages/`
2. Click the pencil icon to edit
3. Add your words to the appropriate indexes
4. Submit a pull request with a brief description

### Full Workflow

1. Fork the repository
2. Edit the language file(s) in `languages/`
3. Run the compiler:
   ```
   python tools/compile.py
   ```
4. If collisions are found, either fix them manually or run the auto-fixer:
   ```
   python tools/fix_collisions.py
   ```
5. Submit a pull request

### What the Collision Checker Does

The compiler detects when the same word maps to different visual indexes across languages. For example, if "ring" maps to both "circle" (198) and "ring" (241), that's a collision. The word must be removed from one index to keep lookups unambiguous.

Run `python tools/compile.py` before submitting. It must report `Collisions: NONE (clean!)`.

## Guidelines for Adding Words

### Do

- Add words that a native speaker would immediately associate with the visual
- Include informal and spoken forms, not just formal/literary words
- Add commonly-used English loanwords if they're part of everyday speech in your language
- Test that your words don't collide with other indexes by running the compiler

### Don't

- Don't add obscure or archaic words that most speakers wouldn't know
- Don't add words that are ambiguous (could mean two very different visuals)
- Don't add offensive or inappropriate words
- Don't remove existing words unless they're incorrect

## Reporting Bugs

If you find an issue (a wrong translation, a word that maps to the wrong visual, or a technical problem), please [open an issue](../../issues) with:

1. **Language** affected
2. **Index** number (0-255) and the visual it represents
3. **The problem**: wrong word, missing word, collision, etc.
4. **Suggested fix** if you have one

## Code of Conduct

Be respectful and constructive. This project serves users worldwide, and every language contribution helps make seed recovery more accessible.

## License

By contributing, you agree that your contributions will be licensed under the [PolyForm Shield License 1.0.0](LICENSE).
