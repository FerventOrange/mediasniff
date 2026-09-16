"""Run the sniffer against randomly chosen real-world streams.

The curated corpus is all reference-grade packagers -- Apple, Unified Streaming,
Axinom, Shaka. They are well-formed by construction, which is exactly why they
are a weak test. This samples large public directories of freely-available
streams instead, so the bytes come from whatever encoder, CDN and packager each
broadcaster happens to run.

Sources (--source):
    iptv     HLS/DASH TV streams from the iptv-org directory (~17k entries)
    tvgarden tv.garden's channel list (~7k stream URLs), curated independently
             of iptv-org and so reaching a different set of packagers. Its
             webcam category is skipped: those are all YouTube embeds, not
             directly fetchable bytes.
    dash     the same directory filtered to .mpd. DASH is only ~1% of it, so
             without this the fMP4 paths go essentially untested.
    radio    Icecast/SHOUTcast radio. Continuous streams with no container and
             no segment boundaries, so a read joins *mid-frame* -- the only
             thing that exercises resync in the packed-audio path.
    freetv   a second, independently maintained TV list
    all      an even split across the above

Ground truth, where available, is the stream's own declaration: CODECS for HLS,
contentType/mimeType for DASH, the codec field for radio. Most of the value is
not the pass rate though -- it is the failures column.

Usage:
    python3 tools/wild.py --source all --sample 200 [--seed 1] [--save-failures]
"""

# pylint: disable=missing-function-docstring,redefined-outer-name,global-statement,wrong-import-position

import argparse
import collections
import json
import os
import random
import re
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import mediasniff as ms  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IPTV_API = "https://iptv-org.github.io/api/streams.json"
FREETV = "https://raw.githubusercontent.com/Free-TV/IPTV/master/playlist.m3u8"
RADIO_API = "https://de1.api.radio-browser.info/json/stations/topclick/400"
# tv.garden's channel list. The original TVGarden repo is archived and points
# here. Its webcams/ category is 100% YouTube embeds, which are not directly
# fetchable bytes, so only the TV list is used.
TVGARDEN = (
    "https://raw.githubusercontent.com/famelack/famelack-channels/main"
    "/tv/raw/categories/all.json"
)

UA = "Mozilla/5.0 (X11; Linux x86_64) mediasniff/wild"
SEGMENT_BYTES = 65536
TIMEOUT = 12
DASH_WANT_MEDIA = False

# Many of these hosts have broken or self-signed TLS. We are classifying bytes,
# not trusting them, and a cert failure would only bias the sample toward
# well-run CDNs -- the bias this whole exercise exists to remove.
_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE

VIDEO_CODEC_RE = re.compile(r"\b(avc[1-4]|hvc1|hev1|dvh[1e]|vp0?[89]|av01|mp4v|vvc1)", re.I)
AUDIO_CODEC_RE = re.compile(r"\b(mp4a|ac-3|ec-3|ac-4|opus|flac|alac|dts)", re.I)
DASH_NS = "{urn:mpeg:dash:schema:mpd:2011}"


def fetch(url, nbytes=None, timeout=TIMEOUT, icy=False):
    headers = {"User-Agent": UA, "Accept": "*/*"}
    if icy:
        # Deliberately NOT sending Icy-MetaData: asking for metadata makes
        # SHOUTcast interleave StreamTitle blocks *into* the audio every
        # icy-metaint bytes, which corrupts the very bytes we are classifying.
        # The ";" path below is the way past the admin page.
        headers["Accept"] = "audio/*, */*"
    req = urllib.request.Request(url, headers=headers)
    if nbytes:
        req.add_header("Range", f"bytes=0-{nbytes - 1}")
    with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as fh:
        blob = fh.read(nbytes or 4 << 20)
        # urllib does not decode Content-Encoding. Several origins gzip their
        # playlists, and without this they all show up as failures.
        if fh.headers.get("Content-Encoding", "").lower() == "gzip":
            try:
                blob = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(blob)
            except zlib.error:
                pass
        return blob, fh.geturl()


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------


def source_iptv(dash_only=False):
    blob, _ = fetch(IPTV_API, timeout=90)
    out = []
    for s in json.loads(blob):
        url = s.get("url") or ""
        if not url.startswith(("http://", "https://")):
            continue
        if dash_only and ".mpd" not in url.lower():
            continue
        out.append((s.get("title") or s.get("channel") or "?", url, "manifest"))
    return out


def source_freetv():
    blob, _ = fetch(FREETV, timeout=60)
    out, name = [], "?"
    for line in blob.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if line.startswith("#EXTINF"):
            name = line.rsplit(",", 1)[-1].strip()
        elif line.startswith(("http://", "https://")):
            out.append((name, line, "manifest"))
    return out


def source_radio():
    blob, _ = fetch(RADIO_API, timeout=60)
    out = []
    for s in json.loads(blob):
        url = s.get("url_resolved") or s.get("url") or ""
        if not url.startswith(("http://", "https://")):
            continue
        codec = (s.get("codec") or "?").upper()
        out.append((f"{s.get('name', '?').strip()[:28]} [{codec}]", url, "radio:" + codec))
    return out


def source_tvgarden(include_geoblocked=False):
    """tv.garden / famelack channel list: ~6.6k channels, ~7k stream URLs.

    Independently curated from iptv-org, so it reaches different packagers.
    Entries carry an isGeoBlocked flag; honouring it keeps the sample from
    filling up with streams that can only fail.
    """
    blob, _ = fetch(TVGARDEN, timeout=90)
    out = []
    for e in json.loads(blob):
        if e.get("isGeoBlocked") and not include_geoblocked:
            continue
        name = (e.get("name") or "?").strip()
        country = (e.get("country") or "").upper()
        for url in (e.get("sources") or {}).get("streams") or []:
            if url.startswith(("http://", "https://")):
                out.append((f"{name[:30]} ({country})", url, "manifest"))
    return out


# --------------------------------------------------------------------------
# resolution: directory entry -> one media fragment
# --------------------------------------------------------------------------


def truth_from_master(text):
    """Ground truth from an HLS master playlist's own attributes.

    The subtlety that makes manifest-based classification unreliable: AUDIO=
    on a variant does NOT by itself mean the audio lives elsewhere. Per RFC
    8216, an EXT-X-MEDIA entry with no URI attribute means that rendition is
    *already present in the variant stream* -- so those segments are muxed.
    Only a group whose members carry a URI actually moves the audio out.
    """
    external_groups = set()
    for line in text.splitlines():
        if line.startswith("#EXT-X-MEDIA:") and "TYPE=AUDIO" in line and "URI=" in line:
            gid = re.search(r'GROUP-ID="([^"]+)"', line)
            if gid:
                external_groups.add(gid.group(1))

    for line in text.splitlines():
        if not line.startswith("#EXT-X-STREAM-INF"):
            continue
        codecs = re.search(r'CODECS="([^"]+)"', line)
        if not codecs:
            continue
        has_v = bool(VIDEO_CODEC_RE.search(codecs.group(1)))
        has_a = bool(AUDIO_CODEC_RE.search(codecs.group(1)))
        group = re.search(r'AUDIO="([^"]+)"', line)
        audio_elsewhere = bool(group) and group.group(1) in external_groups
        if has_v and has_a and not audio_elsewhere:
            return "muxed"
        if has_v:
            return "video"
        if has_a:
            return "audio"
    return None


def resolve_dash(text, base, want_media=False):
    """Pick one representation out of an MPD and build a segment URL for it.

    DASH gives cleaner ground truth than HLS: contentType/mimeType on the
    AdaptationSet states exactly what the segments hold, with none of the
    rendition-group ambiguity that makes HLS CODECS unreliable.
    """
    root = ET.fromstring(text)
    base_el = root.find(DASH_NS + "BaseURL")
    if base_el is not None and base_el.text:
        base = urllib.parse.urljoin(base, base_el.text.strip())

    for period in root.iter(DASH_NS + "Period"):
        for aset in period.iter(DASH_NS + "AdaptationSet"):
            reps = aset.findall(DASH_NS + "Representation")
            if not reps:
                continue
            rep = reps[0]
            mime = aset.get("mimeType") or rep.get("mimeType") or ""
            ctype = aset.get("contentType") or mime.split("/")[0]
            truth = {"video": "video", "audio": "audio", "text": "text"}.get(ctype)
            if truth is None:
                continue

            tmpl = aset.find(DASH_NS + "SegmentTemplate")
            if tmpl is None:
                tmpl = rep.find(DASH_NS + "SegmentTemplate")
            if tmpl is None:
                continue
            tpl = tmpl.get("media") if want_media else tmpl.get("initialization")
            if not tpl:
                continue
            url = tpl.replace("$RepresentationID$", rep.get("id") or "")
            url = url.replace("$Bandwidth$", rep.get("bandwidth") or "")
            if want_media:
                # $Number$ counts from startNumber; $Time$ takes the first entry
                # of the SegmentTimeline. Both may be width-formatted ($Number%04d$).
                start = tmpl.get("startNumber") or "1"
                timeline = tmpl.find(DASH_NS + "SegmentTimeline")
                first_t = "0"
                if timeline is not None:
                    seg = timeline.find(DASH_NS + "S")
                    if seg is not None:
                        first_t = seg.get("t") or "0"

                def _sub(match, start=start, first_t=first_t):
                    var, fmt = match.group(1), match.group(2)
                    val = start if var == "Number" else first_t
                    # Keep the zero-pad flag: $Number%04d$ must render "0001",
                    # not "   1". A bare $Number%4d$ is legal printf but means
                    # space padding, never wanted in a URL, so normalise it.
                    if not fmt:
                        return val
                    return ("%" + (fmt if fmt.startswith("0") else "0" + fmt)) % int(val)

                url = re.sub(r"\$(Number|Time)(?:%(0?\d+d))?\$", _sub, url)
                if "$" in url:
                    continue
            # An init segment carries the moov, so its classification is
            # declared. A media segment carries only moof+mdat, which is the
            # tier that has to be inferred -- and therefore the one worth
            # testing against real-world encoders.
            return truth, urllib.parse.urljoin(base, url)
    return None, None


def first_uri(text, base):
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return urllib.parse.urljoin(base, line)
    return None


def probe(name, url, kind):
    row = {
        "name": name[:38],
        "url": url,
        "kind": kind,
        "status": "",
        "container": "",
        "verdict": "",
        "confidence": "",
        "truth": None,
        "blob": b"",
        "fragment_url": "",
        "truncated": False,
    }
    if kind.startswith("radio:"):
        row["truth"] = "audio"

    is_radio = kind.startswith("radio:")
    try:
        blob, final = fetch(url, SEGMENT_BYTES, icy=is_radio)
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        ssl.SSLError,
        ConnectionError,
        TimeoutError,
        OSError,
    ) as exc:
        row["status"] = f"unreachable: {type(exc).__name__}"
        return row
    except Exception as exc:  # noqa: BLE001
        row["status"] = f"error: {type(exc).__name__}"
        return row

    if is_radio and ms.sniff(blob).container == "html":
        # Still the admin page: the actual stream lives at the ";" path.
        alt = urllib.parse.urljoin(final, ";")
        try:
            blob, final = fetch(alt, SEGMENT_BYTES, icy=True)
        except Exception:  # noqa: BLE001
            pass

    for _hop in range(4):
        rep = ms.sniff(blob)

        if rep.container == "dash-mpd":
            try:
                truth, nxt = resolve_dash(
                    blob.decode("utf-8", "replace"), final, want_media=DASH_WANT_MEDIA
                )
            except ET.ParseError:
                row["status"] = "unparseable MPD"
                return row
            if not nxt:
                row["status"] = "MPD with no resolvable SegmentTemplate"
                return row
            row["truth"] = truth
            try:
                blob, final = fetch(nxt, SEGMENT_BYTES)
            except Exception as exc:  # noqa: BLE001
                row["status"] = f"dash segment unreachable: {type(exc).__name__}"
                return row
            continue

        if rep.is_playlist:
            text = blob.decode("utf-8", "replace")
            if row["truth"] is None:
                row["truth"] = truth_from_master(text)
            nxt = first_uri(text, final)
            if not nxt:
                row["status"] = "playlist with no playable URI"
                return row
            try:
                blob, final = fetch(nxt, SEGMENT_BYTES)
            except Exception as exc:  # noqa: BLE001
                row["status"] = f"child unreachable: {type(exc).__name__}"
                return row
            continue

        if not blob:
            # Zero-byte body: there is nothing to classify, so this is a fetch
            # condition rather than a classification failure.
            row["status"] = "empty response"
            return row
        row.update(
            container=rep.container,
            verdict=rep.verdict,
            confidence=rep.confidence,
            status="ok",
            blob=blob,
            fragment_url=final,
            truncated=rep.truncated,
        )
        return row

    row["status"] = "too many hops"
    return row


def normalize(verdict):
    v = verdict.split("[")[0].split("<-")[0].strip()
    for prefix, label in (
        ("muxed", "muxed"),
        ("video-only", "video"),
        ("audio-only", "audio"),
        ("subtitles", "text"),
    ):
        if v.startswith(prefix):
            return label
    return "unknown"


# --------------------------------------------------------------------------


def report(rows, save_failures):
    ok = [r for r in rows if r["status"] == "ok"]
    dead = [r for r in rows if r["status"] != "ok"]
    print(f"\n{len(rows)} sampled: {len(ok)} reachable, {len(dead)} not\n")

    print("containers:")
    for c, n in collections.Counter(r["container"] for r in ok).most_common():
        print(f"    {n:>3}  {c}")
    print("verdicts:")
    for v, n in collections.Counter(normalize(r["verdict"]) for r in ok).most_common():
        print(f"    {n:>3}  {v}")

    # An opaque AES-128 segment is not a disagreement: "cannot determine without
    # the key" is the correct answer, so score it separately.
    checked = [r for r in ok if r["truth"] and r["container"] != "opaque"]
    opaque = [r for r in ok if r["truth"] and r["container"] == "opaque"]
    wrong = [r for r in checked if normalize(r["verdict"]) != r["truth"]]
    if checked:
        pct = 100.0 * (len(checked) - len(wrong)) / len(checked)
        print(
            f"\nvs declared type: {len(checked) - len(wrong)}/{len(checked)} agree "
            f"({pct:.0f}%)" + (f"; {len(opaque)} excluded as encrypted-opaque" if opaque else "")
        )
        for r in wrong:
            print(
                f"    declared={r['truth']:<6} sniffed={normalize(r['verdict']):<7} "
                f"{r['container']:<22} {r['name']}"
            )
            print(f"        {r['fragment_url'][:104]}")

    problems = [
        r
        for r in ok
        if r["container"] not in ("opaque", "html", "json", "text")
        and (
            normalize(r["verdict"]) == "unknown"
            or r["confidence"] == "low"
            or r["container"] == "unknown"
        )
    ]
    enc = [r for r in ok if r["container"] == "opaque"]
    junk = [r for r in ok if r["container"] in ("html", "json", "text")]
    print(
        f"\nencrypted (correctly opaque): {len(enc)}    "
        f"dead links serving html/json/text: {len(junk)}"
    )
    print(f"unresolved or low confidence: {len(problems)}")
    for r in problems:
        print(
            f"    {r['container']:<20} {r['confidence']:<7} " f"{r['verdict'][:36]:<36} {r['name']}"
        )
        print(f"        {r['fragment_url'][:104]}")
        if save_failures and r["blob"]:
            out = os.path.join(ROOT, "samples", "wild")
            os.makedirs(out, exist_ok=True)
            fn = re.sub(r"\W+", "_", r["name"] or "unnamed")[:44] + ".bin"
            with open(os.path.join(out, fn), "wb") as fh:
                fh.write(r["blob"])
            print(f"        saved samples/wild/{fn} ({len(r['blob'])} bytes)")

    if dead:
        kinds = collections.Counter(r["status"].split(":")[0] for r in dead)
        print("\nunreachable: " + ", ".join(f"{k} x{v}" for k, v in kinds.most_common()))
    return len(wrong), len(problems)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source", default="all", choices=["all", "iptv", "dash", "radio", "freetv", "tvgarden"]
    )
    ap.add_argument("--sample", type=int, default=120)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--save-failures", action="store_true")
    ap.add_argument(
        "--dash-media",
        action="store_true",
        help="resolve DASH to a media segment (moof+mdat, the inferred "
        "tier) instead of an init segment (moov, the declared tier)",
    )
    args = ap.parse_args()
    global DASH_WANT_MEDIA  # noqa: PLW0603
    DASH_WANT_MEDIA = args.dash_media

    rng = random.Random(args.seed)
    buckets = []
    want = {"all": ["iptv", "dash", "radio", "freetv", "tvgarden"]}.get(args.source, [args.source])
    for src in want:
        try:
            got = {
                "iptv": lambda: source_iptv(False),
                "dash": lambda: source_iptv(True),
                "radio": source_radio,
                "freetv": source_freetv,
                "tvgarden": source_tvgarden,
            }[src]()
        except Exception as exc:  # noqa: BLE001
            print(f"  source {src} unavailable: {type(exc).__name__}")
            continue
        print(f"  {src}: {len(got)} entries")
        rng.shuffle(got)
        buckets.append(got)

    # Even split, so the rare sources (DASH, radio) are not swamped by the 17k
    # HLS entries -- the whole point is to exercise the untested paths.
    per = max(1, args.sample // max(1, len(buckets)))
    picked = [e for b in buckets for e in b[:per]]
    rng.shuffle(picked)
    print(f"\nprobing {len(picked)} streams ...")

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(probe, n, u, k) for n, u, k in picked]
        for fut in as_completed(futures):
            rows.append(fut.result())

    wrong, problems = report(rows, args.save_failures)
    return 1 if (wrong or problems) else 0


if __name__ == "__main__":
    sys.exit(main())
