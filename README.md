# mediasniff

Classify streaming media fragments by their **content**, not their filename.

Answers *muxed / video-only / audio-only / subtitles / metadata / encrypted* for
the segment formats used by HLS, HLS-demuxed ("dual"), DASH and CMAF. Pure
stdlib Python, no ffprobe, no dependencies.

The problem it exists to solve: demuxed streams routinely serve audio and video
from near-identical URLs with the same extension. Nothing about the name or the
magic bytes tells them apart.

```
$ python3 src/mediasniff.py samples/*.dash
samples/init_audio.dash    isobmff  iso6 iso6 dash   audio-only [init segment]
samples/init_video.dash    isobmff  iso6 iso6 dash   video-only [init segment]
samples/seg_audio.dash     isobmff  iso6 iso6 msdh   audio-only
samples/seg_video.dash     isobmff  iso6 iso6 msdh   video-only
```

Four files, identical extension, identical magic bytes, identical `ftyp`/`styp`
brands. Correctly separated.

## Usage

```python
import mediasniff

report = mediasniff.sniff(first_64kb_of_a_segment)
report.verdict      # "muxed (audio+video)" | "video-only" | "audio-only" | ...
report.container    # "mpeg-ts" | "isobmff" | "adts" | "webm/matroska" | ...
report.tracks       # per-track kind, codec, PID/track_id, declared vs observed
report.confidence   # "high" | "medium" | "low"
report.evidence     # why it concluded that, in order
```

CLI: `python3 src/mediasniff.py <fragment> [fragment ...]`

**A 4 KB `Range: bytes=0-4095` request is enough for every format here.** `moof`
precedes `mdat`, and HLS requires a PAT+PMT at the head of every TS segment. You
never need to download a whole segment to type it. Verified across the corpus at
4K/16K/64K/256K prefixes.

That works because a prefix is typed from what the container *declares*. Do not
expect to confirm it by observation: a real 7 Mbit/s live TS segment in the test
set carries its first audio packet at **byte offset 1,009,372**. `sniff()`
detects truncation and stops treating an unseen track as an absent one.

## How it decides

Three tiers, in decreasing order of reliability. The tool reports which one it
used, so you can decide how much to trust a given answer.

### 1. Declared — reading a field, not guessing

| Container | Detected by | Track kind comes from |
|---|---|---|
| MPEG-TS | `0x47` at 0, 188, 376… (also 192/204-byte cells) | PAT → PMT `stream_type` + descriptors |
| fMP4 init | `ftyp`/`moov` | `moov/trak/mdia/hdlr`: `vide`/`soun`/`subt` |
| WebM | `1A45DFA3` | `Tracks/TrackEntry/TrackType` |
| FLV (RTMP) | `FLV` | header a/v flags, cross-checked against tag types |
| Packed audio | `ID3` then `0xFFF`/`0x0B77` | the container *is* the codec |
| WebVTT/TTML | `WEBVTT` / `<tt` | — |
| Annex-B | `00 00 01` start codes | NAL unit types |

This tier is deterministic. It is right unless the file is malformed.

**Container magic is checked before any content heuristic.** A CMAF fragment
carrying TTML is an ISOBMFF file with a text track, not a bare TTML file, and
reporting the latter loses the track structure.

### 2. Observed — what is actually in the bytes

`declared` and `observed` are tracked separately, because real packagers
disagree with themselves:

- The Liberty Mountain webcam in the corpus declares an ID3 metadata PID in its
  PMT that carries **zero packets**.
- Packagers that emit one PMT for every rendition will declare audio+video on an
  audio-only segment.

So TS packets are counted per PID and PES `stream_id` is read independently of
the PMT. Disagreement is reported rather than silently resolved.

### 3. Inferred — the sample table, for init-less CMAF fragments

A DASH/CMAF media segment is `styp` + `moof` + `mdat`. **Nothing in it declares
a track kind** — `hdlr` lives in the init segment you may not have. Two
approaches that look obvious both fail on their own:

- **Payload syncword sniffing fails on audio.** AAC in a CMAF `mdat` is *raw*
  access units. The ADTS syncword exists only in TS and packed audio. Sniffing
  for it classifies video fine and silently fails every AAC segment.
- **Absolute sample size lies.** In the tears-of-steel 224x100 rendition the mean
  *video* sample is 86 bytes and the mean *audio* sample is 175 bytes.

What works is the shape of the sample table in `trun`, which is
bitrate-independent:

|  | video | audio |
|---|---|---|
| size max/mean | 6–26 | 1.0–1.6 |
| size cv | 1.1–2.9 | 0.0–0.5 |
| composition offsets | present with B-frames | never |
| sync-sample flags | set (only keyframes seek) | omitted (all samples sync) |
| sample duration | timescale-dependent | 1024/2048 AAC, 1536 AC-3, 1152 MP3, 960/480 Opus |

**The peak ratio (max/mean) is the load-bearing signal, not the coefficient of
variation.** Only video has keyframes, so only video has a peak. VBR AAC reaches
cv 0.46 — high enough to look like video — while its peak ratio stays at 1.4.
Keying on cv misclassifies it; keying on peak does not.

### Where the tiers are genuinely complementary

**All-intra video** (I-frame-only trickplay renditions) is the adversarial case:
every sample is a keyframe, so there is no peak, no B-frames, and no non-sync
samples. Its sample table is statistically indistinguishable from VBR audio.

So tier 3 is deliberately made to return *inconclusive* there rather than guess,
and the `mdat` payload — a chain-validated run of length-prefixed NAL units —
settles it. Direct observation of the elementary stream **outranks** the
statistical read of the metadata; getting that precedence backwards turns an
honest "unknown" into a confident wrong answer.

Neither tier alone covers the corpus. Together they do:

- sample table → solves raw AAC (no syncword to find)
- payload syncword → solves all-intra video (no statistical tells)

## DRM and encryption

Short answer: **everything except whole-segment AES-128 classifies normally, and
the protection metadata is readable on top of that.**

Common Encryption (ISO/IEC 23001-7) is *designed* so a player can parse a
fragment without holding a key. Boxes, handlers, the sample table and the track
structure all stay in the clear; only the media samples are encrypted. So a
Widevine or PlayReady segment classifies exactly as well as a clear one, and the
protection metadata is available as a bonus:

```
$ python3 src/mediasniff.py samples/drm_avc_init.mp4
  verdict   : video-only [init segment] [cenc]   (confidence: high)
  tracks    :  video  h264  trak 1  [D]  encrypted
  evidence  :
      - protection: cenc (AES-CTR, full sample); no in-band pssh (DRM system
        declared in the manifest); KID 9eb4050d-e44b-4802-932e-27d75083e266; IV 8B
```

| Protection | Classifiable? | What is readable |
|---|---|---|
| CENC `cenc`/`cens` (AES-CTR) | yes, fully | scheme, `pssh` system IDs, `tenc` default KID, IV size |
| CENC `cbcs`/`cbc1` (AES-CBC) | yes, fully | as above, plus the crypt:skip pattern |
| HLS SAMPLE-AES (TS) | yes, fully | PMT still declares each stream; `zavc`/`zaac`/`zac3`/`zec3` identifiers |
| WebM ContentEncryption | yes, fully | algorithm, cipher mode, `ContentEncKeyID` |
| PIFF (Smooth Streaming) | yes | legacy `uuid` protection boxes recognised |
| DVB-CSA / conditional access (TS) | yes | PSI stays clear so tracks still type; scrambled PIDs flagged, `stream_id` fallback disabled |
| **HLS AES-128 (whole segment)** | **no** | nothing — reported as opaque, never guessed |

13 DRM system UUIDs are mapped to names (Widevine, PlayReady, FairPlay,
ClearKey, Marlin, Nagra, Irdeto, Verimatrix, WisePlay, …). `pssh` may
legitimately live only in the MPD or playlist rather than in-band, so its
absence is reported as "declared in the manifest", not as a parse failure.

Two practical notes:

- **Subtitle tracks are usually left in the clear** even in a DRM stream. The
  Axinom vectors protect video and audio but not their `wvtt`/`stpp` tracks.
- The **default KID is the useful handle**: it groups fragments by key, which is
  what you want for correlating segments with license requests.

## Results

`python3 tools/evaluate.py` walks every rendition of two public streams, pulls a
64 KB Range of one segment from each, classifies it **with no init segment
available**, and scores against the manifest's own declaration.

```
Apple adv_dv_atmos                30 fragments: 30 correct, 0 unresolved, 0 WRONG
  statistical tier: 17 fragments (video n=7, audio n=10)
    video scores : [-0.5, 6.5]   -- 5/7 clear the +2.0 threshold
    audio scores : [-7.0, -4.5]  -- 10/10 clear the -2.0 threshold
    separation   : +6.5 vs -4.5 -> gap 11.0 against a +/-2.0 decision band
    2 fragments undecided by sample table alone; resolved by the mdat payload tier

Unified Streaming tears-of-steel   7 fragments:  7 correct, 0 unresolved, 0 WRONG

TOTAL: 42 fragments, 0 misclassified
```

Covers AAC-LC, HE-AAC, HE-AACv2, AC-3, E-AC-3/Atmos, H.264, Dolby Vision,
WebVTT in 13 languages, all-intra trickplay, muxed TS and audio-only TS.

The separation is not marginal: decisive scores cluster at +6.5 and -4.5 to -7.0
against a ±2.0 decision band. Nothing in the corpus lands near the boundary.

## Tested against real-world streams

The curated corpus is all reference-grade packagers, which makes it a weak test:
they are well-formed by construction. `tools/wild.py` samples four independent
public directories, resolves each entry down to one media fragment, and
classifies it with no prior knowledge.

| Source | What it covers |
|---|---|
| iptv-org (~17k) | HLS/DASH TV from hundreds of different packagers |
| the same, `.mpd` only (212) | DASH is ~1% of the directory, so without filtering the fMP4 paths go untested |
| tv.garden (~6.3k) | independently curated from iptv-org, so it reaches a different set of packagers. Carries `isGeoBlocked` flags, which lifts the reachable rate to ~95% against iptv-org's ~65% |
| radio-browser (400) | Icecast/SHOUTcast. No container, no segment boundary -- a read joins **mid-frame**, the only thing that exercises resync |
| Free-TV (~2k) | a second, independently maintained TV list |

tv.garden's webcam category (~4.1k entries) is deliberately skipped: it is 100%
YouTube embeds, which are not directly fetchable bytes.

```
$ python3 tools/wild.py --source all --sample 500
500 sampled: 370 reachable, 130 not
containers : 210 mpeg-ts, 59 mpeg-audio, 43 isobmff, 19 adts, 17 opaque,
             11 elementary stream (resynced), 9 html, 1 text, 1 ogg
vs declared type: 282/286 agree (99%)
unresolved or low confidence: 0
```

Stable at **99-100% across eight independent samples** (~3,200 streams), with
zero unresolved. Targeted runs: **117/117** radio, **100/100** tv.garden,
**20/20** DASH init segments, **14/14** init-less DASH media segments. The
residual failures are URLs serving an error page -- HTML, JSON, or in one case a
literal `¯\_(ツ)_/¯` -- each reported as exactly that rather than as "unknown".

That started at **8/21**. Every point of the gap was a real bug. The ones the
first round found:

| Found in the wild | Bug |
|---|---|
| A 7 Mbit/s live TS segment | Its **first audio packet sits at byte 1,009,372**. Preferring "observed" over "declared" made every prefix report video-only. `sniff()` now detects truncation and unions declared with observed. |
| Real LL-HLS `EXT-X-PART` fragments | A 0.2 s part holds ~5 inter frames and no keyframe, so its sample sizes are flat -- statistically identical to audio. Confidently **misclassified as audio-only**. |
| The same fragments | `_nal_chain` rejected any NAL **larger than the sniff window**. Real 2 Mbit/s video has 11 KB slices against an 8 KB window, so the payload check that should have caught the above failed on exactly the content that needed it. |
| A 36-program broadcast multiplex | Only the first 4 PMTs were parsed, and **DVB-CSA conditional access** scrambles PES headers, defeating the `stream_id` fallback. |
| gzipped playlists | A single stray `0x47` in compressed data locked the TS parser on, reporting a **playlist as MPEG-TS**. |

And what the second round -- radio and DASH -- found:

| Found in the wild | Bug |
|---|---|
| Icecast MP3/AAC stations | A continuous stream joins **mid-frame**; the first syncword was 157-3857 bytes in. With no resync these fell through to the entropy heuristic and were reported as **AES-128 encrypted**. 320 kbps MP3 measures 7.99 bits/byte -- indistinguishable from ciphertext by entropy alone. |
| SHOUTcast with ICY metadata | `StreamTitle='...'` blocks are injected **into** the audio every `icy-metaint` bytes, often ~1 KB -- room for only two frames between them. No consecutive-frame check survives that. |
| An obfuscated `.ts` | The AC-3 check accepted a bare `0x0B77` syncword with **no frame validation**, so a video stream was reported as AC-3 audio. Two arbitrary bytes match that pattern every ~64 KB. |
| A live MP3 station | A length-prefixed "NAL" is four arbitrary bytes read as a length plus one as a header. Compressed audio satisfies it by chance, and the station was classified as **video**. |
| An HLS `.aac` rendition | The packager's ID3 tag length leaves an 880-byte gap before the first ADTS frame. Trusting it gave "unknown codec" at low confidence. |
| A live MP3 station | Digital silence is runs of `0x00`, so `00 00 01` appears at the head of audio often enough to matter. One Annex-B start code was enough to call it **video**; three with valid NAL headers are now required. |

### How audio is detected now

Demanding N consecutive frames from offset 0 fails against all three of
mid-frame joins, corrupt frames, and interleaved metadata. Instead, frame
lengths are computed properly (ISO 11172-3 tables for MPEG audio, ATSC A/52 for
AC-3, the ADTS header for AAC) and **coverage is accumulated per stream
identity** -- sample rate, channel layout, version -- with the best identity
winning.

That distinction matters. Both interleaved audio and random noise produce many
broken runs; what separates them is that a real stream's runs all share one
identity, while noise produces scattered ones:

| | coverage |
|---|---|
| clean AAC / MP3 | 99-100% |
| AAC with ICY metadata punched through it | 99.3% |
| real AES-128 ciphertext | 9.2% |
| random bytes (30 trials, worst case) | 12.3% |
| Annex-B H.264 misread as MPEG audio (worst non-audio case) | 39.9% |

The detector fires at 75%, which sits in the measured gap with ~35 points of
margin below and ~22 above. A test asserts both sides of that margin, so an
erosion fails locally instead of waiting for the next wild sweep to find it. Summing coverage
indiscriminately lets noise creep toward the threshold; taking only the longest
single run throws away the interleaved stream. Grouping by identity does
neither.

### The lesson

**Every one of these was a case the curated corpus was too well-behaved to
contain.** Reference clips are low-bitrate, so their NALs are small. Reference
segments are complete, not prefixes. Reference packagers do not gzip playlists,
scramble multiplexes, inject metadata into audio, or misreport their own ID3
lengths. And they always start at a frame boundary.

### The manifests lie

Most remaining "disagreements" are the manifest being wrong, verified by hand:

- Four separate broadcasters declare `CODECS="avc1.…"` -- video only -- while
  serving segments whose own URL path reads `tracks-v1a1` (video track 1, audio
  track 1) and which carry MP3 or AAC audio.
- `EXT-X-MEDIA:TYPE=AUDIO` **without** a `URI` attribute means that rendition is
  already inside the variant, so those segments are muxed. Treating any `AUDIO=`
  reference as "audio lives elsewhere" is wrong, and `tools/wild.py` had to be
  corrected for it.

Content-based classification is not just a workaround for bad filenames. It
catches manifests that misdeclare their own contents.

## Caveats

- **Whole-segment AES-128** (`EXT-X-KEY:METHOD=AES-128`) makes a fragment
  genuinely opaque. It is reported as encrypted with no track list — never
  guessed at. `SAMPLE-AES` and CENC leave the container parseable and work fine.
- **Manifests are not ground truth either.** A demuxed HLS variant's `CODECS`
  attribute lists `avc1,mp4a` while its segments contain only video — `CODECS`
  describes the playback combination, not the segment. `tools/evaluate.py` has
  to account for this to compute correct ground truth, which is itself an
  argument for classifying by content.
- **A multi-track fragment with no init segment** can be counted (one `traf` per
  track) but not attributed per track from the payload. It reports the track
  count and declines to name them.
- The EBML walk is minimal — enough for `TrackType` and `ContentEncryption`.
- **Not covered:** MPEG-H 3D audio track typing, DVB subtitle page structure,
  multi-program TS is flagged but each program is not reported separately, and a
  multi-track fragment without its init segment can be counted but not named.
- **H.264 vs HEVC is not always decidable.** One NAL header byte reads as
  plausible in both; only parameter-set units settle it. A fragment whose NALs
  are all slices and SEI reports `h264/hevc` rather than guessing.

## Live-specific shapes

| Shape | Handling |
|---|---|
| LL-HLS partial segments (`EXT-X-PART`) | A part that is not the first of its segment opens mid-`mdat` with no header at all. Classified from the payload run. |
| RTMP / FLV ingest | Header a/v flags cross-checked against actual tag types, so a muxer that declares audio but never sends it is caught. |
| Raw Annex-B elementary streams | Start-code scan, NAL unit types. |
| Byte-range segments (`EXT-X-BYTERANGE`) | A range into a TS file re-syncs on `0x47`; a range into an fMP4 lands in the headless path. |
| Smooth Streaming / PIFF | `uuid` extension boxes (`tfxd`, `tfrf`, protection headers) recognised and reported. |
| Multi-program TS | Flagged explicitly, because "muxed" across programs means something different. |

## Layout

```
src/mediasniff.py           the classifier
tests/test_mediasniff.py    107 tests: corpus, short prefixes, fuzz, regressions
tools/fetch_samples.sh      rebuild the corpus from public test vectors
tools/make_samples.py       derive packed-audio / AES-128 / WebM samples
tools/evaluate.py           measure separation across many renditions
tools/wild.py               sample real-world streams from five public directories
tests/fixtures/             checked-in captures from live streams (see its README)
```

```bash
./tools/fetch_samples.sh    # corpus is gitignored, not committed
make test
```

The corpus is deliberately adversarial: four `.dash` files with byte-identical
magic and brands (two audio, two video), two `.ts` files of identical length
whose names differ by one infix, and a WebVTT file that Apple serves under a
`.mp4`-shaped path.

## License

MIT -- see [LICENSE](LICENSE). Use it on your own streams freely.

The captures in `tests/fixtures/` are **not** covered by that license: they are
short fragments of third-party broadcast and radio streams, included solely as
test vectors because each one caught a real bug. See
[tests/fixtures/README.md](tests/fixtures/README.md) for provenance.
