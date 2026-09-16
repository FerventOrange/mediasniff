"""Measure how well sniff() separates audio from video on init-less fragments.

The declared tier (TS with a PMT, fMP4 with a moov, WebM with Tracks) is not
interesting to evaluate -- it reads a field, so it is right or the file is
malformed. The statistical tier is the one that can actually be wrong: a
moof+mdat fragment whose init segment we do not have.

This walks every rendition of a public HLS stream, pulls a 64 KB Range of one
media segment from each, classifies it with no init segment available, and
scores the result against the manifest's own declaration.

Usage:  python3 tools/evaluate.py [--limit N]
"""

import argparse
import os
import re
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import mediasniff as ms  # noqa: E402

APPLE = "https://devstreaming-cdn.apple.com/videos/streaming/examples/adv_dv_atmos/main.m3u8"
UNIFIED = ("https://demo.unified-streaming.com/k8s/features/stable/video/"
           "tears-of-steel/tears-of-steel.ism/.m3u8")

RANGE_BYTES = 65536


def fetch(url, nbytes=None):
    req = urllib.request.Request(url, headers={"User-Agent": "mediasniff/eval"})
    if nbytes:
        req.add_header("Range", f"bytes=0-{nbytes - 1}")
    with urllib.request.urlopen(req, timeout=45) as fh:
        return fh.read()


def renditions(master_url):
    """Yield (truth, name, media_playlist_url) for every rendition in a master
    playlist. `truth` comes from the tag type and its CODECS attribute."""
    text = fetch(master_url).decode("utf-8", "replace")
    base = master_url.rsplit("/", 1)[0] + "/"
    lines = text.splitlines()

    for line in lines:
        if not line.startswith("#EXT-X-MEDIA:"):
            continue
        uri = re.search(r'URI="([^"]+)"', line)
        typ = re.search(r"TYPE=([A-Z\-]+)", line)
        name = re.search(r'NAME="([^"]+)"', line)
        if not uri or not typ:
            continue
        kind = {"AUDIO": "audio", "SUBTITLES": "text",
                "CLOSED-CAPTIONS": "text"}.get(typ.group(1))
        if kind:
            yield kind, (name.group(1) if name else typ.group(1)), urllib.parse.urljoin(base, uri.group(1))

    for i, line in enumerate(lines):
        if line.startswith("#EXT-X-STREAM-INF") and i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            if nxt and not nxt.startswith("#"):
                res = re.search(r"RESOLUTION=(\S+?)(?:,|$)", line)
                tag = (res.group(1) if res else "audio-only variant")
                yield _truth_from_codecs(line), f"variant {tag}", urllib.parse.urljoin(base, nxt)
        elif line.startswith("#EXT-X-I-FRAME-STREAM-INF"):
            uri = re.search(r'URI="([^"]+)"', line)
            res = re.search(r"RESOLUTION=(\S+?)(?:,|$)", line)
            if uri:
                # All-intra trickplay: every sample is a keyframe. The hardest
                # video case there is, because it has none of the usual video
                # tells -- no B-frames, no non-sync samples.
                yield "video", f"iframe-only {res.group(1) if res else ''}", urllib.parse.urljoin(base, uri.group(1))


VIDEO_CODEC_RE = re.compile(r"\b(avc[1-4]|hvc1|hev1|dvh[1e]|vp0?[89]|av01|mp4v|vvc1)", re.I)
AUDIO_CODEC_RE = re.compile(r"\b(mp4a|ac-3|ec-3|ac-4|opus|flac|alac|dts[a-z]?)", re.I)


def _truth_from_codecs(stream_inf_line):
    """What a variant's own segments carry.

    CODECS is NOT the answer on its own: it describes the whole playback
    combination, so a demuxed variant lists avc1 AND mp4a while its segments
    hold only video. The AUDIO= / SUBTITLES= attributes are what disambiguate --
    they say the audio arrives from a separate rendition group, which means
    this variant's segments are video-only.

    This is a good argument for content-based classification in the first
    place: even a correct manifest does not tell you what is inside a segment.
    """
    codecs = re.search(r'CODECS="([^"]+)"', stream_inf_line)
    text = codecs.group(1) if codecs else ""
    has_v = bool(VIDEO_CODEC_RE.search(text))
    has_a = bool(AUDIO_CODEC_RE.search(text))
    audio_elsewhere = "AUDIO=" in stream_inf_line
    if has_v and has_a and not audio_elsewhere:
        return "muxed"
    if has_v:
        return "video"
    return "audio" if has_a else "video"


def first_segment(playlist_url):
    text = fetch(playlist_url).decode("utf-8", "replace")
    base = playlist_url.rsplit("/", 1)[0] + "/"
    seg = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#EXT-X-BYTERANGE"):
            continue
        if line and not line.startswith("#"):
            seg = line
            break
    return urllib.parse.urljoin(base, seg) if seg else None


def evaluate(master_url, label, limit):
    print(f"\n=== {label} ===")
    rows, n = [], 0
    for truth, name, pl in renditions(master_url):
        if n >= limit:
            break
        try:
            seg_url = first_segment(pl)
            if not seg_url:
                continue
            blob = fetch(seg_url, RANGE_BYTES)
        except Exception as exc:                     # noqa: BLE001
            print(f"  skip {name}: {exc}")
            continue
        n += 1

        rep = ms.sniff(blob)
        if rep.container != "isobmff" or any(t.declared for t in rep.tracks):
            rows.append((truth, _normalize(rep.verdict), name, rep.container, None,
                         f"declared ({rep.container})"))
            continue

        # statistical tier. Note sniff() also has an mdat-syncword fallback that
        # can resolve a traf this tier leaves unknown; `full` shows the outcome
        # of the whole pipeline, `score` isolates the sample-table tier alone.
        full = _normalize(rep.verdict)
        trafs = ms._scan_trafs(blob)
        for idx, info in enumerate(trafs):
            kind, _codec, why, score = ms._classify_traf(info)
            # Secondary trafs in a video segment are caption/metadata tracks the
            # manifest does not describe; report them but do not score them
            # against the rendition's own truth.
            row_truth = truth if idx == 0 else "(secondary)"
            got = kind.value if idx else (full if full != "unknown" else kind.value)
            rows.append((row_truth, got, name, "moof-only", score,
                         why + ("" if kind.value == got else f" -> {got} via mdat syncword")))

    _report(rows)
    return rows


def _normalize(verdict):
    """Map a verdict string onto the same vocabulary the manifest uses."""
    v = verdict.split("[")[0].split("<-")[0].strip()
    if v.startswith("muxed"):
        return "muxed"
    if v.startswith("video-only"):
        return "video"
    if v.startswith("audio-only"):
        return "audio"
    if v.startswith("subtitles"):
        return "text"
    if v.startswith("metadata"):
        return "data"
    return "unknown"


def _report(rows):
    rows = [r for r in rows if r[0] != "(secondary)"]
    ok = sum(1 for t, g, *_ in rows if t == g)
    unresolved = sum(1 for t, g, *_ in rows if g == "unknown")
    wrong = [r for r in rows if r[0] != r[1] and r[1] != "unknown"]
    print(f"  {len(rows)} fragments: {ok} correct, {unresolved} unresolved, {len(wrong)} WRONG")

    stat = [r for r in rows if r[4] is not None]
    if stat:
        vid = [r[4] for r in stat if r[0] == "video"]
        aud = [r[4] for r in stat if r[0] == "audio"]
        print(f"  statistical tier: {len(stat)} fragments "
              f"(video n={len(vid)}, audio n={len(aud)})")
        # Separate the fragments the sample table decided on its own from the
        # ones that only came out right because the mdat payload rescued them.
        vid_clear = [v for v in vid if v >= 2.0]
        aud_clear = [a for a in aud if a <= -2.0]
        if vid:
            print(f"    video scores : {sorted(set(round(v, 1) for v in vid))}"
                  f"  -- {len(vid_clear)}/{len(vid)} clear the +2.0 threshold")
        if aud:
            print(f"    audio scores : {sorted(set(round(a, 1) for a in aud))}"
                  f"  -- {len(aud_clear)}/{len(aud)} clear the -2.0 threshold")
        if vid_clear and aud_clear:
            print(f"    separation   : worst decisive video {min(vid_clear):+.1f} vs "
                  f"worst decisive audio {max(aud_clear):+.1f} "
                  f"-> gap {min(vid_clear) - max(aud_clear):+.1f}")
        undecided = len(vid) - len(vid_clear) + len(aud) - len(aud_clear)
        if undecided:
            print(f"    {undecided} fragment(s) undecided by sample table alone; "
                  f"resolved by the mdat payload tier")
    for t, g, name, container, score, why in rows:
        mark = "ok  " if t == g else ("??  " if g == "unknown" else "WRONG")
        sc = f"{score:+5.1f}" if score is not None else "  -- "
        print(f"    {mark} truth={t:<6} got={g:<8} {sc}  {name[:34]:<34} {why[:60]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()
    rows = evaluate(APPLE, "Apple adv_dv_atmos", args.limit)
    rows += evaluate(UNIFIED, "Unified Streaming tears-of-steel", args.limit)
    wrong = [r for r in rows if r[0] != r[1] and r[1] != "unknown"]
    print(f"\nTOTAL: {len(rows)} fragments, {len(wrong)} misclassified")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
