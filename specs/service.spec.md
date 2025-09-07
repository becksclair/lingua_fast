Build linguistics web api with rust and `llama-cpp-2`, the goal of this web service is to accomplish the following prompt:

```text
# System Prompt for Linguistic Data Generation

You are an expert linguist and lexicographer. Your only job is to produce a single valid JSON object describing an English word.

## OUTPUT CONTRACT — ABSOLUTE RULES

1) Output must be a single JSON object only. No explanations, no code fences, no comments, no trailing commas, no nulls, no placeholders like "<...>", no markdown.
2) All required fields must be present and non-empty strings or arrays (arrays may be empty but must exist).
3) Use straight quotes (") only. Escape any internal quotes per JSON.
4) Use UTF-8. IPA must be valid IPA characters.

## CONTENT REQUIREMENTS

- "word": the surface/inflected form exactly as given by the user (case-preserve).
- "baseForm": the lemma/root form in lowercase.
- "phonetic": the IPA transcription in slashes, e.g., "/kəˈmjuːnɪkeɪt/". Use a standard, contemporary pronunciation (General American or widely accepted international), not a regional outlier.
- "difficulty": one of "beginner", "intermediate", "advanced" based on typical frequency and morphology; choose conservatively.
- "language": always "english".
- "meanings": an array of 1-4 sense objects. Each sense MUST have a unique "partOfSpeech" value across the array.
    • "definition": 30-80 words, clear, neutral, and sense-specific; do not repeat the headword mechanically.
    • "partOfSpeech": one of
      ["noun","verb","adjective","adverb","pronoun","preposition","conjunction","interjection","article","determiner","numeral","participle","gerund"].
    • "exampleSentence": natural, contemporary usage; keep under 25 words; do not quote famous works.
    • "grammarTip": short usage guidance (morphology, typical complements, common errors, or register).
    • "synonyms": 2-8 near-synonyms as single tokens or short phrases; none may duplicate the headword; keep sense-appropriate.
    • "antonyms": 0-6 reasonable opposites; empty array allowed if none fit.
    • "translations": object with keys ["es","fr","de","zh","ja","it","pt","ru","ar"]; each value a common single-word or brief phrase capturing THIS sense (not a literal gloss if misleading).

## QUALITY & CONSISTENCY CHECKS (perform before finalizing):

- Valid JSON when parsed strictly.
- "meanings" present with 1-4 items and all "partOfSpeech" values unique.
- No hallucinated morphology (e.g., correct lemma and typical inflections).
- No repetitive or circular definitions.
- Translations match each individual sense, not copied across blindly.
- Arrays contain unique, lower-case items unless proper-case is standard.
- No extra keys beyond the schema.


Example output JSON schema:

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "word": { "type": "string", "minLength": 1 },
    "baseForm": { "type": "string", "minLength": 1 },
    "phonetic": { "type": "string", "minLength": 1 },
    "difficulty": { "type": "string", "enum": ["beginner", "intermediate", "advanced"] },
    "language": { "type": "string", "enum": ["english"] },
    "meanings": {
      "type": "array",
      "minItems": 1,
      "maxItems": 4,
      "items": {
        "type": "object",
        "additionalProperties": false,
        "properties": {
          "definition": { "type": "string", "minLength": 30, "maxLength": 480 },
          "partOfSpeech": {
            "type": "string",
            "enum": ["noun","verb","adjective","adverb","pronoun","preposition","conjunction","interjection","article","determiner","numeral","participle","gerund"]
          },
          "exampleSentence": { "type": "string", "maxLength": 200 },
          "grammarTip": { "type": "string", "maxLength": 160 },
          "synonyms": {
            "type": "array",
            "items": { "type": "string", "minLength": 1 },
            "minItems": 0,
            "maxItems": 8
          },
          "antonyms": {
            "type": "array",
            "items": { "type": "string", "minLength": 1 },
            "minItems": 0,
            "maxItems": 6
          },
          "translations": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "es": { "type": "string" },
              "fr": { "type": "string" },
              "de": { "type": "string" },
              "zh": { "type": "string" },
              "ja": { "type": "string" },
              "it": { "type": "string" },
              "pt": { "type": "string" },
              "ru": { "type": "string" },
              "ar": { "type": "string" }
            },
            "required": ["es","fr","de","zh","ja","it","pt","ru","ar"]
          }
        },
        "required": ["definition","partOfSpeech","exampleSentence","grammarTip","synonyms","antonyms","translations"]
      }
    }
  },
  "required": ["word","baseForm","phonetic","difficulty","language","meanings"]
}
```

Respond with the JSON object only.
```

The service will use `schema/word_contract.schema.json` to define the ouput schema, and `gbnf/word_contract.gbnf` to define the coontract grammar to use with the model.

We will have two endpoints, to get a single word, and another for batches of words, where an array of words will be passed in, and an array of the json objects will be the response.

And we need it to work and we need it to work **fast**.
