# Instant transcription and a streaming local agent

Design + measurements, 2026-09-17. Implemented on `feat/instant-live-agent`.

## Problem

Live transcription felt slow and the local copilot barely worked:

- **Transcription.** On Whisper large-v3-turbo a line landed ~2 s after the
  speaker stopped, and the "words as they're spoken" preview updated once in
  47 s of speech: every preview re-decoded the open utterance through a full
  30 s-padded Whisper pass, and the loop backs off to 2× the decode time.
- **Copilot on Ollama.** Live cards went through the OpenAI-compatible
  endpoint without a thinking switch. On qwen3.5:9b that request spent its
  whole token budget on hidden reasoning and returned no text (below). Even
  without thinking, each pass rebuilt a multi-thousand-token prompt, and a
  fresh prefill on this model runs at ~220–300 tok/s — seconds to tens of
  seconds per card, JSON not streamed.
- **Local files.** Documents had to be added one by one; retrieval was
  NLEmbedding cosine only (no Croatian model, weak on names and jargon).

Goal: words on screen while someone talks, and an answer suggestion within
about a second of the other side finishing a question, grounded in the
user's own files — all on this Mac.

## Measurements

Apple M4 Pro, 24 GB, macOS 26.7, Ollama 0.34.0, qwen3.5:9b (Q4_K_M).
Transcription numbers come from `scripts/dev/live-harness` feeding a 47 s
bilingual test call (macOS voices, English + Croatian, 8 utterances) at
recording pace through the real `TranscriptionEngine`; lag is measured from
the end of the committed segment to the moment `onSegment` fired.

### Transcription

| Engine | Speech end → committed line (median / p90 / max) | Preview updates in 47 s |
|---|---|---|
| Whisper large-v3-turbo, before | 2.01 / 2.13 / 2.21 s | 1 |
| Parakeet TDT v3 (FluidAudio) | **0.53 / 0.55 / 0.61 s** | 67 (before the 0.8 s preview floor) |
| Whisper large-v3-turbo, after (preview reuse; warm ANE cache, not a controlled A/B) | 1.29 / 1.43 / 1.47 s | 14 |

Parakeet's floor is the segmenter itself: 600 ms of silence ends an
utterance, then the decode takes tens of milliseconds. First load 42.8 s
(461 MB download + CoreML compile); later loads use the compiled cache.

Quality on the test call: English was exact on both engines. On Croatian
Parakeet got one sentence right and drifted into Polish/Slovak spellings on
another ("Tylko odobrva rezervácie…" for "Tko odobrava rezervacije…"), where
Whisper was closer. Whisper stays one click away in Settings.

### Local model (qwen3.5:9b)

| Request shape | First token | Notes |
|---|---|---|
| Compat endpoint, no think flag (the old live path) | — | 400 tokens of reasoning, no answer, 29 s |
| Compat endpoint, `reasoning_effort: "none"` | 0.18 s | cached prompt |
| Native, fresh 2.2k–3.2k-token prompt | 8.5–14.6 s | prefill 219–305 tok/s |
| Native, append-only session, each transcript increment | **0.36–0.90 s** | flat up to 3.2k tokens of session |
| Same session, last user message replaced | 2.46 s | rollback to a checkpoint |
| Stream cancelled, partial reply appended, next turn | 0.35 s | cancel is cache-safe |
| Fresh 1.2k prefill, `num_batch` 512 / 1024 / 2048 | 5.5 / 4.0 / 4.6 s | 1024 chosen |

Generation runs at ~28–30 tok/s. qwen3.5 is a hybrid (recurrent) model:
Ollama keeps the state of the last sequence plus sparse checkpoints, so a
prompt that merely *extends* the previous one is nearly free, and one that
changes anything earlier re-reads from the nearest checkpoint or from zero.

### End to end (audio → Parakeet → agent)

`LiveHarness agent`, same test calls, docs folder with two files:

| Turn | Question commit → first token | → answer complete |
|---|---|---|
| English question, document found | 0.42–0.87 s (median 0.50) | 1.05–1.92 s |
| Croatian question, English document (search step) | 1.60 s | 2.75 s |
| Update (NOW line / ASK / NOTE) | 0.36–0.73 s | 1.5–2.0 s |

Add the ~0.55 s commit lag for speech end → first word of the answer.

## Design

### 1. Parakeet as the live engine

`ParakeetTranscriber` wraps FluidAudio's `AsrManager` (already a dependency
for diarization). The live loop keeps its segmenter, filters and clock; only
the decoder is chosen per session. Previews run every 0.35 s (2× decode-time
backoff kept) once an utterance has 800 ms of audio — earlier previews
invented a word from the first syllable. The segmenter polls every 100 ms
instead of 250 ms. `TranscriptionBackend.parakeet` is the default for
installs that never picked an engine. Whisper no longer loads at launch for
Parakeet users; file import loads it on demand, and a Parakeet download
failure brings Whisper up so recording still works.

Custom vocabulary can't prime Parakeet (no prompt), so every engine's output
gets a spelling pass: a glossary term heard with its letters split or in the
wrong case ("data tonic", "a s d l c", "octo") is rewritten; letters must match
in order at word boundaries, so "October" stays and misheard words are left
alone.

A Whisper commit reuses the newest preview when that preview decoded the same
utterance (same start, covering the speech up to the cut and at most the
bounding pause) instead of decoding the same audio again.

### 2. LiveAgent: one append-only session per call

When the live provider is Ollama (and Settings → Copilot → Streaming live
agent is on, the default), `CallAnalysisEngine` hands the call to `LiveAgent`
instead of the paced JSON loop — running both would evict each other's cached
state in Ollama's single slot.

- **Session.** System prompt (stable for the call: format, persona, brief,
  glossary, document list) prefilled once at recording start (warm-up turn).
  Every later request appends a user turn and the model's reply. Nothing
  already sent is edited. Options (`num_ctx` 32k, `num_batch` 1024,
  `keep_alive` 30m) never change mid-call — a change reloads the model.
- **Tagged lines, streamed.** `NOW:` topic, `ASK:` a question worth asking,
  `ANSWER:` what to say, `NOTE:` a document fact, `SOURCE:`, `SEARCH:`.
  `LiveAgentReply.parse` tolerates bullets, bold, lowercase and Croatian tags,
  and hides a half-arrived tag mid-stream.
- **Scheduling.** A committed "Them" line that looks like a question (English
  and Croatian openers, or "?", which Parakeet writes) becomes an answer turn
  immediately; an update in flight is cancelled and its partial reply kept.
  Other lines batch into updates (≥12 words, ≥10 s apart, 1.5 s debounce;
  sooner past 80 words). Warm-up and compaction are never preempted.
- **Retrieval.** Questions look up the question plus its lead-in; updates look
  up what was just said, so a document can speak up unasked. Each excerpt is
  sent once per call. When nothing matches a question and the user has
  documents, a short `[search]` turn asks the model for keywords in English and
  the call's language, and the lookup runs again — this is what bridges a
  Croatian question to English files. When nothing matches at all, the
  directive says so; the prompt keeps figures, names and sources to what the
  call or a document gave (without that, the 9B model invented a price and a
  file name).
- **Deadlines and recovery.** 20 s to first token (120 s for cold turns); a
  timed-out or failed turn rolls back if it produced nothing, keeps its partial
  reply otherwise, and the agent re-checks Ollama until it answers — starting
  Ollama mid-call recovers without restarting the recording.
- **Compaction.** Past 70% of the context the model summarizes the call and
  the session restarts from the summary (one fresh prefill, ~40 min of dense
  talk apart).

Outputs land in the existing feed: answers as `suggestion` (or `ask_answer`
when typed in the Ask card), `live_ask`, `live_note`. The panel shows a NOW
card in place of the coach card and a live answer card while an answer
streams, with question → first-word latency.

The legacy JSON path on Ollama now sends `reasoning_effort: "none"`.

### 3. Knowledge folders

Settings → Knowledge → Add Folder… stores a security-scoped bookmark
(`com.apple.security.files.bookmarks.app-scope`), indexes readable files
(Markdown, text, code, PDF, DOCX/DOC/RTF/ODT, HTML), and re-checks the folder
at launch and at recording start by modification date. Build/VCS directories
and credential-looking files are skipped; 2,000 files and 80 chunks per file
at most. Folder chunks carry no embedding (fast, language-agnostic); they're
found by BM25 over accent-folded tokens (đ included), fused with embedding
similarity for hand-added documents by reciprocal rank. The index is written
off the main thread.

## Not done / next

- **Speaking answers (TTS)** — deferred by choice. The pipeline shape from the
  voice-latency work applies directly: phrase splitter on the streamed ANSWER,
  one synthesis at a time with ≤2 finished phrases queued, playback cancelled
  when the user starts talking. FluidAudio's Supertonic-3 covers Croatian
  (31 languages, ~81 ms first audio at int4 on an M5 Pro in FluidAudio's
  benchmark). The app's own audio is already excluded from the system-audio
  tap, so spoken answers wouldn't reach the transcript; on speakers the mic
  would still hear them.
- **Cross-language retrieval without the search turn** — a multilingual
  embedding model in Ollama would match Croatian questions to English files
  directly; it's an extra model download, so it's the user's call.
- **Croatian accuracy on Parakeet** — per-language token allowlists in
  FluidAudio, or a post-call Whisper polish pass run locally.
- **Speculative answers** — Parakeet punctuates, so a "?" in the preview could
  start retrieval (or the answer) before the 600 ms pause confirms the end.

## Testing

- `--profile-test` runs `LiveLatencyChecks` (backend/language hints, glossary
  respelling, reply parsing, question detection, prompt shape, BM25 + RRF,
  chunking, folder scans, model-name matching).
- `scripts/dev/live-harness` builds with Command Line Tools only:
  `selftest`, `stt` (engine latency on a file), `agent` (end to end against
  local Ollama). `make-test-call.sh` generates the bilingual call and docs.
