#!/bin/zsh
# Builds a bilingual test call for the live harness: English and Croatian
# macOS voices with pauses, questions included, plus a small docs folder the
# agent can ground answers in. Usage: ./make-test-call.sh <output-dir>
set -euo pipefail
out=${1:?usage: make-test-call.sh <output-dir>}
mkdir -p "$out/docs/specs"
cd "$out"

fmt=(--file-format=WAVE --data-format=LEI16@16000)
say -v Samantha -o u1.wav $fmt "Thanks for joining. Today I want to walk through the BigQuery migration plan for Datatonic."
say -v Daniel -o u2.wav $fmt "Sure. We have the dbt models ready for the first two domains."
say -v Samantha -o u3.wav $fmt "What will the scheduled queries cost after the cutover?"
say -v "Lana (Enhanced)" -o u4.wav $fmt "Koliko će koštati rezervacija nakon prelaska?"
say -v Daniel -o u5.wav $fmt "We also need to talk about the review process before go live."
say -v Samantha -o u6.wav $fmt "Who runs the ASDLC review board?"

python3 - <<'PY'
import wave
gap = b"\x00\x00" * int(16000 * 1.6)
frames = [gap]
for i in range(1, 7):
    w = wave.open(f"u{i}.wav"); frames += [w.readframes(w.getnframes()), gap]; w.close()
out = wave.open("meeting.wav", "wb"); out.setnchannels(1); out.setsampwidth(2); out.setframerate(16000)
out.writeframes(b"".join(frames)); out.close()
PY
rm -f u?.wav

cat > docs/bigquery-costs.md <<'MD'
# BigQuery migration — cost estimate

Scheduled queries after the cutover run on a 500-slot Enterprise reservation.
Estimated cost is about 4,100 USD per month, billed monthly.

Finance approves reservation commitments; the platform team proposes them.
MD
cat > docs/specs/asdlc-review.md <<'MD'
# ASDLC review

The ASDLC review checks security controls, data lineage and a tested rollback plan.
Reviews happen every Tuesday. A domain cannot cut over to production before it passes.
Owner of the review board: Ana Kovač.
MD
echo "Test call: $out/meeting.wav, docs: $out/docs"
