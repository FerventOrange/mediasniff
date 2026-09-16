"""Classify streaming media fragments by content, not by filename.

Answers "muxed / video-only / audio-only / subtitles / metadata / encrypted"
for the segment formats used by HLS, HLS-demuxed ("dual"), DASH and CMAF:

  * MPEG-2 Transport Stream  (.ts, .m2ts) -- PAT/PMT + PES inspection
  * ISO BMFF / fMP4 / CMAF   (.mp4, .m4s, .cmfv, .cmfa) -- box + handler walk
  * HLS packed audio         (bare ADTS / MP3 / AC-3, usually ID3-prefixed)
  * WebM / Matroska          (DASH-WebM) -- EBML TrackType walk
  * WebVTT / TTML(IMSC)      -- text subtitle segments
  * AES-128 whole-segment encryption -- detected as opaque, not misreported

Pure stdlib, no ffprobe. Designed to work on a partial fragment: a few hundred
KB of the head is enough for a confident answer in every format here.

Two distinct questions are kept separate on purpose, because demuxed streams
routinely disagree on them:

  declared -- what the container's index says the fragment carries
              (TS PMT entries, moov trak handlers, WebM TrackEntries)
  observed -- what elementary data is actually present in these bytes
              (PES packets per PID, trafs in moof, codec syncwords in mdat)

A packager that emits the same PMT for every rendition will declare
"audio+video" on an audio-only segment. Observation catches that; declaration
catches the opposite case of a fragment that simply has no keyframe yet.
"""

from __future__ import annotations

import enum
import struct
import zlib
from dataclasses import dataclass, field


class Kind(str, enum.Enum):
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"
    DATA = "data"          # ID3 timed metadata, SCTE-35, emsg, etc.
    UNKNOWN = "unknown"


@dataclass
class Track:
    kind: Kind
    codec: str = ""
    track_id: int | None = None
    pid: int | None = None          # MPEG-TS only
    declared: bool = False          # named by an index (PMT / moov / Tracks)
    observed: bool = False          # elementary data actually seen in bytes
    packets: int = 0                # TS packets or samples attributed to it
    note: str = ""

    def __str__(self) -> str:
        loc = f"pid 0x{self.pid:04x}" if self.pid is not None else f"trak {self.track_id}"
        flags = "".join(c for c, on in (("D", self.declared), ("O", self.observed)) if on)
        return f"{self.kind.value:<7} {self.codec:<10} {loc:<12} [{flags}]" + (
            f"  {self.note}" if self.note else ""
        )


@dataclass
class Report:
    container: str = "unknown"
    verdict: str = "unknown"
    tracks: list[Track] = field(default_factory=list)
    brands: list[str] = field(default_factory=list)
    encrypted: str = ""             # "", "aes-128-full", "sample-aes", "cenc"
    drm: "Drm" = field(default_factory=lambda: Drm())
    is_init: bool = False           # init segment (no media payload)
    is_playlist: bool = False       # you were handed a manifest, not a fragment
    truncated: bool = False         # these bytes are a prefix, not a whole fragment
    confidence: str = "low"         # low | medium | high
    evidence: list[str] = field(default_factory=list)

    def kinds(self, *, observed_only: bool = False) -> set[Kind]:
        return {
            t.kind
            for t in self.tracks
            if (t.observed if observed_only else (t.observed or t.declared))
            and t.kind in (Kind.VIDEO, Kind.AUDIO, Kind.TEXT)
        }


# --------------------------------------------------------------------------
# MPEG-2 Transport Stream
# --------------------------------------------------------------------------

# ISO/IEC 13818-1 table 2-34, plus the ATSC/SCTE/Blu-ray private extensions
# that real packagers emit.
TS_STREAM_TYPES: dict[int, tuple[Kind, str]] = {
    0x01: (Kind.VIDEO, "mpeg1video"),
    0x02: (Kind.VIDEO, "mpeg2video"),
    0x03: (Kind.AUDIO, "mp2"),
    0x04: (Kind.AUDIO, "mp3"),
    0x05: (Kind.DATA, "sections"),
    0x06: (Kind.UNKNOWN, "pes-private"),   # resolve via descriptors
    0x0F: (Kind.AUDIO, "aac-adts"),
    0x10: (Kind.VIDEO, "mpeg4visual"),
    0x11: (Kind.AUDIO, "aac-latm"),
    0x15: (Kind.DATA, "id3-metadata"),     # HLS timed metadata rides here
    0x1B: (Kind.VIDEO, "h264"),
    0x1C: (Kind.AUDIO, "aac-raw"),
    0x1F: (Kind.VIDEO, "svc"),
    0x20: (Kind.VIDEO, "mvc"),
    0x21: (Kind.VIDEO, "jpeg2000"),
    0x24: (Kind.VIDEO, "hevc"),
    0x25: (Kind.VIDEO, "hevc-subset"),
    0x33: (Kind.VIDEO, "vvc"),
    0x81: (Kind.AUDIO, "ac3"),
    0x82: (Kind.AUDIO, "dts"),
    0x83: (Kind.AUDIO, "truehd"),
    0x84: (Kind.AUDIO, "eac3"),
    0x85: (Kind.AUDIO, "dts-hd"),
    0x87: (Kind.AUDIO, "eac3"),
    0x8A: (Kind.AUDIO, "dts"),
    0x91: (Kind.AUDIO, "ac3"),
    0x92: (Kind.TEXT, "subtitle"),
    0xC1: (Kind.AUDIO, "ac3 [SAMPLE-AES]"),
    0xC2: (Kind.AUDIO, "eac3 [SAMPLE-AES]"),
    0xCF: (Kind.AUDIO, "aac [SAMPLE-AES]"),
    0xDB: (Kind.VIDEO, "h264 [SAMPLE-AES]"),
    0x1A: (Kind.DATA, "iso14496-sections"),
    0x86: (Kind.DATA, "scte35"),   # SCTE-35 splice_info; Blu-ray reuses it for DTS-HD MA
}

# Registration descriptor (tag 0x05) format_identifiers seen in the wild.
TS_REGISTRATIONS: dict[bytes, tuple[Kind, str]] = {
    b"AC-3": (Kind.AUDIO, "ac3"),
    b"EAC3": (Kind.AUDIO, "eac3"),
    b"AC-4": (Kind.AUDIO, "ac4"),
    b"DTS1": (Kind.AUDIO, "dts"),
    b"DTS2": (Kind.AUDIO, "dts"),
    b"DTS3": (Kind.AUDIO, "dts"),
    b"Opus": (Kind.AUDIO, "opus"),
    b"mlpa": (Kind.AUDIO, "truehd"),
    b"HEVC": (Kind.VIDEO, "hevc"),
    b"AVC1": (Kind.VIDEO, "h264"),
    b"VC-1": (Kind.VIDEO, "vc1"),
    b"ID3 ": (Kind.DATA, "id3-metadata"),
    b"CUEI": (Kind.DATA, "scte35"),
    # HLS SAMPLE-AES format identifiers (Apple HLS spec, section on encryption)
    b"zavc": (Kind.VIDEO, "h264 [SAMPLE-AES]"),
    b"zaac": (Kind.AUDIO, "aac [SAMPLE-AES]"),
    b"zac3": (Kind.AUDIO, "ac3 [SAMPLE-AES]"),
    b"zec3": (Kind.AUDIO, "eac3 [SAMPLE-AES]"),
}

# Descriptor tags that identify stream_type 0x06 payloads on their own.
TS_DESCRIPTOR_TAGS: dict[int, tuple[Kind, str]] = {
    0x56: (Kind.TEXT, "teletext"),
    0x59: (Kind.TEXT, "dvb-subtitle"),
    0x6A: (Kind.AUDIO, "ac3"),
    0x7A: (Kind.AUDIO, "eac3"),
    0x7B: (Kind.AUDIO, "dts"),
    0x7C: (Kind.AUDIO, "aac"),
    0x7F: (Kind.UNKNOWN, "extension"),
}

TS_PACKET_SIZES = (188, 192, 204, 208)


def _ts_packet_ok(data: bytes, off: int) -> bool:
    """Validate a TS packet header, not just its sync byte. A lone 0x47 occurs
    in any binary at a rate of 1/256 -- gzip-compressed text triggered a false
    MPEG-TS lock before this check existed."""
    if off + 4 > len(data) or data[off] != 0x47:
        return False
    if data[off + 1] & 0x80:                       # transport_error_indicator
        return False
    if (data[off + 3] >> 4) & 0x03 == 0:           # adaptation_field_control 0 is reserved
        return False
    return True


def _ts_layout(data: bytes) -> tuple[int, int] | None:
    """Return (packet_size, first_packet_offset) for an MPEG-TS fragment."""
    for size in TS_PACKET_SIZES:
        # M2TS/TTS prefixes each 188-byte packet with a 4-byte arrival stamp,
        # so the sync byte sits at offset 4 within the 192-byte cell.
        base = 4 if size in (192, 208) else 0
        for start in range(0, min(len(data), size * 2)):
            if size in (192, 208) and start < base:
                continue
            if not _ts_packet_ok(data, start):
                continue
            fits = (len(data) - start) // size
            if fits < 1:
                continue
            checked = min(fits, 8)
            if any(not _ts_packet_ok(data, start + n * size) for n in range(checked)):
                continue
            if checked >= 3:
                return size, start
            # Too little data to lock on repetition. Accept only a stream that
            # begins exactly at a packet boundary, which HLS requires of every
            # TS segment, and only once a whole packet is present.
            if start == base:
                return size, start
    return None


def _ts_descriptor_kind(desc: bytes) -> tuple[Kind, str] | None:
    i = 0
    while i + 2 <= len(desc):
        tag, length = desc[i], desc[i + 1]
        body = desc[i + 2 : i + 2 + length]
        if tag == 0x05 and len(body) >= 4:
            hit = TS_REGISTRATIONS.get(bytes(body[:4]))
            if hit:
                return hit
        elif tag in TS_DESCRIPTOR_TAGS:
            hit = TS_DESCRIPTOR_TAGS[tag]
            if hit[0] is not Kind.UNKNOWN:
                return hit
        i += 2 + length
    return None


def _ts_sections(data: bytes, size: int, start: int, pid_wanted: int) -> list[bytes]:
    """Reassemble PSI sections carried on one PID. Good enough for PAT/PMT,
    which packagers keep inside a single packet in practice but are spec'd to
    span several."""
    out: list[bytes] = []
    buf = bytearray()
    want = 0
    for off in range(start, len(data) - size + 1, size):
        pkt = data[off : off + 188] if size in (192, 208) else data[off : off + size]
        if not pkt or pkt[0] != 0x47:
            continue
        pid = ((pkt[1] & 0x1F) << 8) | pkt[2]
        if pid != pid_wanted:
            continue
        payload_start = bool(pkt[1] & 0x40)
        afc = (pkt[3] >> 4) & 0x03
        p = 4
        if afc in (2, 3):
            p += 1 + pkt[4]
        if afc == 2 or p >= len(pkt):
            continue
        if payload_start:
            p += 1 + pkt[p]          # pointer_field
            if p >= len(pkt):
                continue
            buf = bytearray(pkt[p:])
            if len(buf) >= 3:
                want = (((buf[1] & 0x0F) << 8) | buf[2]) + 3
        elif buf:
            buf += pkt[p:]
        else:
            continue
        if want and len(buf) >= want:
            out.append(bytes(buf[:want]))
            buf, want = bytearray(), 0
            if len(out) >= 4:
                break
    return out


def _parse_ts(data: bytes, size: int, start: int, rep: Report) -> None:
    rep.container = "mpeg-ts" if size == 188 else f"mpeg-ts({size}-byte cells)"
    rep.evidence.append(f"0x47 sync lock: {size}-byte packets at offset {start}")

    tracks: dict[int, Track] = {}

    # --- declared: PAT -> PMT -----------------------------------------------
    pmt_pids: list[int] = []
    for sec in _ts_sections(data, size, start, 0x0000):
        if not sec or sec[0] != 0x00:
            continue
        body, end = sec[8:], len(sec) - 4
        for i in range(0, max(0, end - 8), 4):
            if i + 4 > len(body):
                break
            prog = (body[i] << 8) | body[i + 1]
            pid = ((body[i + 2] & 0x1F) << 8) | body[i + 3]
            if prog != 0 and pid not in pmt_pids:
                pmt_pids.append(pid)
    if pmt_pids:
        rep.evidence.append(f"PAT lists PMT pid(s): {', '.join(hex(p) for p in pmt_pids)}")
    if len(pmt_pids) > 1:
        rep.evidence.append(
            f"{len(pmt_pids)} programs in this mux -- the track list below spans "
            "all of them, so 'muxed' here may mean separate programs"
        )

    for pmt_pid in pmt_pids[:64]:
        for sec in _ts_sections(data, size, start, pmt_pid):
            if not sec or sec[0] != 0x02 or len(sec) < 12:
                continue
            prog_info_len = ((sec[10] & 0x0F) << 8) | sec[11]
            i = 12 + prog_info_len
            end = len(sec) - 4
            while i + 5 <= end:
                stype = sec[i]
                epid = ((sec[i + 1] & 0x1F) << 8) | sec[i + 2]
                dlen = ((sec[i + 3] & 0x0F) << 8) | sec[i + 4]
                desc = sec[i + 5 : i + 5 + dlen]
                kind, codec = TS_STREAM_TYPES.get(stype, (Kind.UNKNOWN, f"type-0x{stype:02x}"))
                hit = _ts_descriptor_kind(desc)
                if hit and (kind is Kind.UNKNOWN or stype in (0x06, 0x86, 0x05)):
                    kind, codec = hit
                t = tracks.setdefault(epid, Track(kind=kind, pid=epid))
                t.kind, t.codec, t.declared = kind, codec, True
                i += 5 + dlen
    if tracks:
        rep.evidence.append(f"PMT declares {len(tracks)} elementary stream(s)")

    # --- observed: count packets per PID, read PES stream_id ------------------
    seen: dict[int, int] = {}
    pes_id: dict[int, int] = {}
    scrambled: set[int] = set()
    for off in range(start, len(data) - size + 1, size):
        pkt = data[off : off + 188] if size in (192, 208) else data[off : off + size]
        if not pkt or pkt[0] != 0x47:
            continue
        pid = ((pkt[1] & 0x1F) << 8) | pkt[2]
        if pid in (0x0000, 0x1FFF) or pid in pmt_pids:
            continue
        seen[pid] = seen.get(pid, 0) + 1
        if (pkt[3] >> 6) & 0x03:
            # transport_scrambling_control: 2 = even key, 3 = odd key
            scrambled.add(pid)
            continue
        if pkt[1] & 0x40 and pid not in pes_id:
            afc = (pkt[3] >> 4) & 0x03
            p = 4 + (1 + pkt[4] if afc in (2, 3) else 0)
            if p + 4 <= len(pkt) and pkt[p : p + 3] == b"\x00\x00\x01":
                pes_id[pid] = pkt[p + 3]

    for pid, count in seen.items():
        t = tracks.setdefault(pid, Track(kind=Kind.UNKNOWN, pid=pid))
        t.observed, t.packets = True, count
        sid = pes_id.get(pid)
        if t.kind is Kind.UNKNOWN and sid is not None:
            # ISO/IEC 13818-1 table 2-22 stream_id ranges.
            if 0xE0 <= sid <= 0xEF:
                t.kind, t.codec = Kind.VIDEO, t.codec or "pes-video"
            elif 0xC0 <= sid <= 0xDF:
                t.kind, t.codec = Kind.AUDIO, t.codec or "pes-audio"
            elif sid == 0xBD:
                t.kind, t.codec = Kind.AUDIO, t.codec or "private-stream-1"
            elif sid in (0xBE, 0xBF, 0xF0, 0xF1, 0xFF):
                t.kind, t.codec = Kind.DATA, t.codec or "pes-padding/private"
            t.note = (t.note + f" stream_id 0x{sid:02x}").strip()
        if t.declared and not t.note:
            t.note = f"{count} pkts"

    if scrambled:
        rep.encrypted = "dvb-csa"
        rep.drm.scheme = "DVB-CSA / conditional access"
        rep.drm.note = ("PSI stays in the clear so tracks are still typed from "
                        "the PMT; PES headers are scrambled")
        rep.evidence.append(
            f"transport_scrambling_control set on {len(scrambled)} PID(s) -- "
            "conditional access, so PES stream_id typing is unavailable")

    prefix = (len(data) - start) % size != 0
    for t in tracks.values():
        if t.declared and not t.observed:
            t.note = (t.note + (" declared, not in this prefix" if prefix
                                else " DECLARED BUT ABSENT")).strip()
        if t.observed and not t.declared and t.packets > 2:
            t.note = (t.note + " not in PMT").strip()

    rep.tracks = sorted(tracks.values(), key=lambda t: (t.kind.value, t.pid or 0))

    payload = sum(t.packets for t in tracks.values() if t.kind in (Kind.VIDEO, Kind.AUDIO))
    for t in tracks.values():
        if t.kind in (Kind.VIDEO, Kind.AUDIO) and payload:
            t.note = (t.note + f" {100 * t.packets // payload}% of a/v payload").strip()

    if any("SAMPLE-AES" in t.codec for t in tracks.values()):
        rep.encrypted = "sample-aes"
        rep.drm.scheme = "sample-aes"
        rep.drm.note = "container and PES headers stay in the clear"
        rep.evidence.append("protection: HLS SAMPLE-AES (declared in the PMT)")

    # A TS fragment is a whole number of fixed-size packets. Anything else is a
    # prefix -- which matters a lot, because audio can be interleaved sparsely:
    # a 7 Mbit/s live segment routinely carries 200 KB of video before its first
    # audio packet. Not having seen audio yet is not evidence there is none.
    rep.truncated = (len(data) - start) % size != 0
    if rep.truncated:
        rep.evidence.append(
            f"truncated: {(len(data) - start) % size} bytes past the last whole "
            f"{size}-byte packet, so absent tracks may simply be later in the segment")

    typed = any(t.kind in (Kind.VIDEO, Kind.AUDIO, Kind.TEXT) for t in tracks.values())
    if not typed:
        rep.confidence = "low"
    elif pmt_pids and seen:
        rep.confidence = "high"
    elif seen:
        rep.confidence = "medium"
    else:
        rep.confidence = "low"


# --------------------------------------------------------------------------
# ISO BMFF / fMP4 / CMAF
# --------------------------------------------------------------------------

ISOBMFF_TOP = {
    b"ftyp", b"styp", b"moov", b"moof", b"mdat", b"free", b"skip", b"sidx",
    b"ssix", b"emsg", b"prft", b"mfra", b"meta", b"pdin", b"uuid", b"wide",
}

HANDLER_KINDS = {
    b"vide": Kind.VIDEO,
    b"soun": Kind.AUDIO,
    b"subt": Kind.TEXT,
    b"sbtl": Kind.TEXT,
    b"text": Kind.TEXT,
    b"clcp": Kind.TEXT,
    b"meta": Kind.DATA,
    b"hint": Kind.DATA,
    b"auxv": Kind.VIDEO,
}

# Sample entry 4CCs -> friendly codec name. Encrypted tracks use encv/enca and
# hide the real 4CC under sinf/frma, which _mp4_boxes walks into.
SAMPLE_ENTRIES = {
    b"avc1": "h264", b"avc3": "h264", b"avc2": "h264", b"avc4": "h264",
    b"hvc1": "hevc", b"hev1": "hevc", b"dvh1": "dolby-vision", b"dvhe": "dolby-vision",
    b"vvc1": "vvc", b"vvi1": "vvc",
    b"vp08": "vp8", b"vp09": "vp9", b"av01": "av1",
    b"mp4v": "mpeg4visual", b"jpeg": "jpeg", b"j2ki": "jpeg2000",
    b"mp4a": "aac", b"ac-3": "ac3", b"ec-3": "eac3", b"ac-4": "ac4",
    b"Opus": "opus", b"fLaC": "flac", b"alac": "alac", b"dtsc": "dts",
    b"dtse": "dts", b"dtsh": "dts", b"dtsl": "dts", b"mlpa": "truehd",
    b"samr": "amr", b"sawb": "amr-wb", b"sowt": "pcm", b"ipcm": "pcm",
    b"wvtt": "webvtt", b"stpp": "ttml", b"tx3g": "3gpp-text", b"c608": "cea608",
    b"urim": "timed-metadata", b"mett": "timed-metadata", b"metx": "timed-metadata",
}

_CONTAINER_BOXES = {
    b"moov", b"trak", b"mdia", b"minf", b"stbl", b"moof", b"traf", b"mvex",
    b"edts", b"dinf", b"udta", b"sinf", b"schi", b"mfra", b"stsd",
}


def _mp4_boxes(data: bytes, start: int = 0, end: int | None = None, depth: int = 0):
    """Yield (type, payload_offset, payload_end, depth), descending into
    containers. stsd needs its 8-byte version/entry_count header skipped, and
    sample entries need their own fixed header skipped, before children appear."""
    end = len(data) if end is None else end
    i = start
    while i + 8 <= end:
        size = struct.unpack_from(">I", data, i)[0]
        btype = data[i + 4 : i + 8]
        hdr = 8
        if size == 1:
            if i + 16 > end:
                return
            size = struct.unpack_from(">Q", data, i + 8)[0]
            hdr = 16
        elif size == 0:
            size = end - i
        if size < hdr or i + size > end:
            # Truncated tail -- normal when sniffing only the head of a file.
            yield btype, i + hdr, end, depth
            return
        body, body_end = i + hdr, i + size
        yield btype, body, body_end, depth
        if btype in _CONTAINER_BOXES:
            skip = 8 if btype == b"stsd" else 0
            yield from _mp4_boxes(data, body + skip, body_end, depth + 1)
        elif btype in SAMPLE_ENTRIES or btype in (b"encv", b"enca", b"encs", b"enct"):
            # VisualSampleEntry = 78 bytes, AudioSampleEntry = 28, others 8.
            head = 78 if btype in (b"encv",) or SAMPLE_ENTRIES.get(btype, "") in (
                "h264", "hevc", "vvc", "vp8", "vp9", "av1", "mpeg4visual",
                "dolby-vision", "jpeg", "jpeg2000",
            ) else 28 if btype in (b"enca",) or btype in (
                b"mp4a", b"ac-3", b"ec-3", b"ac-4", b"Opus", b"fLaC", b"alac",
                b"dtsc", b"dtse", b"dtsh", b"dtsl", b"mlpa", b"samr", b"sawb",
                b"sowt", b"ipcm",
            ) else 8
            if body + head < body_end:
                yield from _mp4_boxes(data, body + head, body_end, depth + 1)
        i += size


# H.264 and HEVC cannot be told apart from one NAL header byte -- an SEI unit
# reads as plausible in both. Parameter-set units are unambiguous though, so
# collect several headers and look for one.
#   H.264 (1 byte):  forbidden(1) nal_ref_idc(2) type(5)   SPS=7 PPS=8
#   HEVC  (2 bytes): forbidden(1) type(6) layer(6) tid+1(3) VPS=32 SPS=33 PPS=34
H264_PARAM_SETS = {7, 8}
HEVC_PARAM_SETS = {32, 33, 34}


def _guess_nal_codec(headers: list[int]) -> str:
    for b in headers:
        if b & 0x80:
            continue
        if (b >> 1) & 0x3F in HEVC_PARAM_SETS and (b & 0x01) == 0:
            return "hevc"
    for b in headers:
        if b & 0x80:
            continue
        if b & 0x1F in H264_PARAM_SETS:
            return "h264"
    return "h264/hevc"


def _nal_chain(buf: bytes, probes: int = 3, headers: list[int] | None = None,
               max_len: int | None = None) -> bool:
    """True if buf looks like length-prefixed NAL units (AVC/HEVC in mdat).

    Appends each NAL header byte it validates to `headers`, if given. Pass
    `max_len` -- the size of the payload actually containing these bytes -- to
    bound the length field: a NAL cannot be longer than its own mdat, whereas a
    random u32 averages two billion. Without that bound a single chance hit on
    high-entropy data reads as video.
    """
    i, ok = 0, 0
    while ok < probes and i + 5 <= len(buf):
        n = struct.unpack_from(">I", buf, i)[0]
        if n == 0 or n > 16 << 20 or (max_len is not None and n > max_len):
            return False
        hdr = buf[i + 4]
        if hdr & 0x80:                       # forbidden_zero_bit must be 0
            return False
        avc_type, hevc_type = hdr & 0x1F, (hdr >> 1) & 0x3F
        if not (1 <= avc_type <= 23 or hevc_type <= 47):
            return False
        if headers is not None:
            headers.append(hdr)
        ok += 1
        if i + 4 + n > len(buf):
            # Last unit runs past our window. Credible only if it is large
            # enough to be real video payload rather than a coincidence -- an
            # 8-byte WebVTT box header otherwise reads as a valid NAL.
            return ok >= 2 or n >= 256
        i += 4 + n
    return ok >= probes or (ok >= 2 and i + 5 > len(buf))


def _adts_chain(buf: bytes, probes: int = 3) -> bool:
    i, ok = 0, 0
    while ok < probes and i + 7 <= len(buf):
        if buf[i] != 0xFF or (buf[i + 1] & 0xF0) != 0xF0 or (buf[i + 1] & 0x06):
            return False
        n = ((buf[i + 3] & 0x03) << 11) | (buf[i + 4] << 3) | (buf[i + 5] >> 5)
        if n < 7:
            return False
        ok += 1
        if i + n > len(buf):
            return ok >= 1
        i += n
    return ok >= probes


# --- moof sample-table analysis -------------------------------------------
#
# When a media fragment arrives without its init segment -- the normal case for
# a CMAF/DASH .m4s or .dash chunk -- the only thing that distinguishes an audio
# track from a video track is the *shape of its sample table*:
#
#            video track                      audio track
#   count    one sample per frame (~24-60/s)  one per codec frame (~47/s AAC)
#   sizes    wildly uneven: an IDR is 10-50x  near-constant: a CBR-ish frame
#            the mean, P/B frames are tiny    varies maybe +/-30%
#   timing   composition offsets when B-      no reordering, ever
#            frames are present
#   flags    sync/non-sync marked, because    every sample is a sync sample, so
#            only keyframes are seekable      packagers omit the flags entirely
#   dur      timescale-dependent              1024/2048 (AAC), 1536 (AC-3),
#                                             960/480 (Opus), 1152 (MP3)
#
# Size *variance* is the strongest single signal and is bitrate-independent:
# a 224x100 video clip can have a smaller mean sample than its own AAC track,
# but never a smaller coefficient of variation.

# Codec frame lengths, in the track's own timescale. Uniform sample durations
# matching one of these are a *positive* signal for audio.
#
# They are not unambiguous, and the ambiguity is irreducible from a moof alone:
# a sample duration is timescale/fps, the timescale lives in the init segment's
# mdhd, and without it duration 1024 means either "an AAC frame" or "24 fps at
# timescale 24576". 33 of these values are reachable by real video at standard
# frame rates -- 512@12800 and 480@12000 are both 25 fps, 1024@24576 is 24 fps.
# Fetch the init segment when the answer has to be certain.
AUDIO_FRAME_DURATIONS = {1024, 2048, 1536, 1152, 960, 576, 512, 480, 320, 240}

TRUN_DATA_OFFSET = 0x000001
TRUN_FIRST_SAMPLE_FLAGS = 0x000004
TRUN_SAMPLE_DURATION = 0x000100
TRUN_SAMPLE_SIZE = 0x000200
TRUN_SAMPLE_FLAGS = 0x000400
TRUN_SAMPLE_CTS = 0x000800


@dataclass
class TrafInfo:
    track_id: int = 0
    sample_count: int = 0
    sizes: list[int] = field(default_factory=list)
    durations: list[int] = field(default_factory=list)
    ctts: list[int] = field(default_factory=list)
    default_duration: int = 0
    default_size: int = 0
    trun_flags: int = 0
    has_subsample_encryption: bool = False


def _scan_trafs(data: bytes) -> list[TrafInfo]:
    """Parse every traf in every moof into a TrafInfo. Handles the tfhd/trun
    flag-driven optional fields and the tfhd defaults that trun inherits."""
    # A segment may hold several moofs (one per fragment). Merge their trafs by
    # track_id so one track yields one sample table, not N partial ones.
    merged: dict[int, TrafInfo] = {}
    order: list[int] = []
    cur: TrafInfo | None = None
    pending: list[TrafInfo] = []
    for btype, body, body_end, _ in _mp4_boxes(data):
        if btype == b"traf":
            cur = TrafInfo()
            pending.append(cur)
        elif cur is None:
            continue
        elif btype == b"tfhd" and body + 8 <= body_end:
            flags = struct.unpack_from(">I", data, body)[0] & 0xFFFFFF
            cur.track_id = struct.unpack_from(">I", data, body + 4)[0]
            i = body + 8
            if flags & 0x000001:            # base_data_offset (64-bit)
                i += 8
            if flags & 0x000002:            # sample_description_index
                i += 4
            if flags & 0x000008 and i + 4 <= body_end:
                cur.default_duration = struct.unpack_from(">I", data, i)[0]
                i += 4
            if flags & 0x000010 and i + 4 <= body_end:
                cur.default_size = struct.unpack_from(">I", data, i)[0]
                i += 4
        elif btype == b"trun" and body + 8 <= body_end:
            ver = data[body]
            flags = struct.unpack_from(">I", data, body)[0] & 0xFFFFFF
            count = struct.unpack_from(">I", data, body + 4)[0]
            cur.trun_flags |= flags
            cur.sample_count += count
            i = body + 8
            if flags & TRUN_DATA_OFFSET:
                i += 4
            if flags & TRUN_FIRST_SAMPLE_FLAGS:
                i += 4
            for _n in range(count):
                if i > body_end:
                    break
                if flags & TRUN_SAMPLE_DURATION:
                    cur.durations.append(struct.unpack_from(">I", data, i)[0])
                    i += 4
                if flags & TRUN_SAMPLE_SIZE:
                    cur.sizes.append(struct.unpack_from(">I", data, i)[0])
                    i += 4
                if flags & TRUN_SAMPLE_FLAGS:
                    i += 4
                if flags & TRUN_SAMPLE_CTS:
                    raw = struct.unpack_from(">I", data, i)[0]
                    # version 1 makes composition offsets signed
                    cur.ctts.append(struct.unpack_from(">i", data, i)[0] if ver else raw)
                    i += 4
        elif btype in (b"senc", b"saiz"):
            cur.has_subsample_encryption = True

    for info in pending:
        tid = info.track_id
        if tid not in merged:
            merged[tid] = info
            order.append(tid)
            continue
        acc = merged[tid]
        acc.sample_count += info.sample_count
        acc.sizes += info.sizes
        acc.durations += info.durations
        acc.ctts += info.ctts
        acc.trun_flags |= info.trun_flags
        acc.default_duration = acc.default_duration or info.default_duration
        if info.default_size and not info.sizes:
            # this fragment's samples are all tfhd-default sized
            acc.sizes += [info.default_size] * max(1, info.sample_count)
        acc.default_size = acc.default_size or info.default_size
        acc.has_subsample_encryption |= info.has_subsample_encryption
    return [merged[t] for t in order]


def _classify_traf(info: TrafInfo) -> tuple[Kind, str, str, float]:
    """Score one traf as video or audio from its sample table alone.

    The scoring is deliberately asymmetric. Audio has a positive structural
    tell -- a uniform sample duration equal to a codec frame length -- while
    every video tell (B-frames, non-sync samples, a keyframe size peak) is a
    marker of *complexity* that all-intra video legitimately lacks. So video
    evidence may be concluded from its own presence, but audio is never
    concluded from the mere absence of video evidence.

    Returns (kind, codec-guess, reasoning, signed score: >0 video, <0 audio)."""
    sizes = info.sizes or ([info.default_size] * info.sample_count if info.default_size else [])
    score = 0.0                                  # >0 video, <0 audio
    why: list[str] = []

    if info.trun_flags & TRUN_SAMPLE_CTS and any(c for c in info.ctts):
        score += 3.0
        why.append("composition offsets (B-frames)")
    if info.trun_flags & TRUN_SAMPLE_FLAGS:
        score += 1.0
        why.append("per-sample flags")
    if info.trun_flags & TRUN_FIRST_SAMPLE_FLAGS:
        score += 1.0
        why.append("first_sample_flags (sync sample marked)")

    # Decide this first: a keyframe peak is positive video evidence, and it
    # outranks a duration that merely coincides with a codec frame length.
    sizes_early = info.sizes or (
        [info.default_size] * info.sample_count if info.default_size else [])
    decisive_peak = False
    if len(sizes_early) >= 4:
        mean_early = sum(sizes_early) / len(sizes_early)
        decisive_peak = mean_early > 0 and max(sizes_early) / mean_early >= 3.0

    durs = info.durations or ([info.default_duration] if info.default_duration else [])
    audio_frame_timing = False
    if durs:
        uniform = len(set(durs)) == 1
        d = durs[0]
        if uniform and d in AUDIO_FRAME_DURATIONS and not decisive_peak:
            audio_frame_timing = True
            score -= 2.5
            why.append(f"uniform sample duration {d} = a codec frame size")
        elif uniform and d in AUDIO_FRAME_DURATIONS:
            why.append(f"duration {d} matches a codec frame size, but a keyframe "
                       "peak is present -- treating the match as coincidence")
        elif uniform:
            score -= 0.5
            why.append(f"uniform sample duration {d}")
        else:
            score += 0.5
            why.append("variable sample durations")

    if len(sizes) >= 4:
        mean = sum(sizes) / len(sizes)
        if mean > 0:
            var = sum((x - mean) ** 2 for x in sizes) / len(sizes)
            cv = (var ** 0.5) / mean
            peak = max(sizes) / mean
            why.append(f"size cv {cv:.2f}, max/mean {peak:.1f}")
            # A keyframe is many times the mean of the frames around it. Nothing
            # in an audio track produces that, at any bitrate. The converse is
            # weaker: low peak means "no keyframe here", which is audio only if
            # the frames are also uniform -- an all-intra video track is every
            # keyframe and so has no peak either.
            enough = len(sizes) >= 8
            if peak >= 3.0:
                score += 3.0
            elif not enough:
                why.append(f"only {len(sizes)} samples -- too few to infer audio")
            elif cv <= 0.35 and peak <= 2.0:
                # Flat frame sizes with no keyframe peak. This is what audio
                # looks like -- and also exactly what all-intra CBR video looks
                # like, because every frame is a keyframe and the rate control
                # holds them the same size. Absence of video tells is NOT
                # evidence of audio, so this only counts alongside a positive
                # audio signal. Without one the fragment stays inconclusive and
                # the payload tier, or the init segment, decides.
                if audio_frame_timing:
                    score -= 3.0
                else:
                    why.append("flat sizes but no audio frame timing -- "
                               "consistent with all-intra video, not concluded")
            elif cv <= 0.55 and peak <= 1.8 and audio_frame_timing:
                score -= 2.0                     # VBR audio: uneven but no peak
            elif cv >= 0.60:
                score += 0.5                     # uneven with no peak: all-intra
            if len(set(sizes)) == 1 and audio_frame_timing:
                score -= 1.5
                why.append("constant sample size (CBR frames or PCM)")

    reason = "sample table: " + ", ".join(why) if why else "sample table: no usable signal"
    if score >= 2.0:
        return Kind.VIDEO, "video (codec unknown)", reason, score
    if score <= -2.0:
        return Kind.AUDIO, "audio (codec unknown)", reason, score
    return Kind.UNKNOWN, "", reason + " -- inconclusive, fetch the init segment", score


def _mp4_truncated(data: bytes) -> bool:
    """True if the last top-level box declares a size that runs past the buffer."""
    i = 0
    while i + 8 <= len(data):
        size = struct.unpack_from(">I", data, i)[0]
        hdr = 8
        if size == 1:
            if i + 16 > len(data):
                return True
            size = struct.unpack_from(">Q", data, i + 8)[0]
            hdr = 16
        elif size == 0:
            return False                     # "to end of file" -- complete
        if size < hdr:
            return False
        if i + size > len(data):
            return True
        i += size
    return i != len(data)


def _parse_mp4(data: bytes, rep: Report) -> None:
    rep.container = "isobmff"
    rep.truncated = _mp4_truncated(data)
    tracks: list[Track] = []
    cur: Track | None = None
    saw_moov = saw_moof = saw_mdat = saw_sidx = False
    mdat_range: tuple[int, int] | None = None
    frma: bytes | None = None
    scheme = ""

    for btype, body, body_end, _ in _mp4_boxes(data):
        if btype in (b"ftyp", b"styp"):
            major = data[body : body + 4].decode("latin-1", "replace").strip()
            compat = [
                data[j : j + 4].decode("latin-1", "replace").strip()
                for j in range(body + 8, min(body_end, body + 8 + 64), 4)
            ]
            rep.brands = [b for b in [major, *compat] if b]
            rep.evidence.append(f"{btype.decode()} brands: {' '.join(rep.brands)}")
        elif btype == b"moov":
            saw_moov = True
        elif btype == b"moof":
            saw_moof = True
        elif btype == b"sidx":
            saw_sidx = True
        elif btype == b"mdat":
            saw_mdat = True
            if mdat_range is None:
                mdat_range = (body, body_end)
        elif btype == b"trak":
            cur = Track(kind=Kind.UNKNOWN, declared=True)
            tracks.append(cur)
        elif btype == b"tkhd" and cur is not None:
            ver = data[body]
            off = body + (20 if ver == 1 else 12)
            if off + 4 <= body_end:
                cur.track_id = struct.unpack_from(">I", data, off)[0]
        elif btype == b"hdlr" and cur is not None:
            h = data[body + 8 : body + 12]
            cur.kind = HANDLER_KINDS.get(h, Kind.UNKNOWN)
            if cur.kind is Kind.UNKNOWN:
                cur.note = f"handler {h.decode('latin-1', 'replace')}"
        elif btype in SAMPLE_ENTRIES and cur is not None and not cur.codec:
            cur.codec = SAMPLE_ENTRIES[btype]
        elif btype in (b"encv", b"enca") and cur is not None:
            cur.codec = cur.codec or "encrypted"
            scheme = scheme or "cenc"
        elif btype == b"frma":
            frma = data[body : body + 4]
            if cur is not None:
                cur.codec = SAMPLE_ENTRIES.get(frma, frma.decode("latin-1", "replace"))
                cur.note = (cur.note + " encrypted").strip()
        elif btype == b"schm":
            scheme = data[body + 4 : body + 8].decode("latin-1", "replace")
            rep.drm.scheme = scheme
        elif btype == b"tenc" and body + 24 <= body_end:
            # FullBox: version/flags, reserved, [pattern if v1], isProtected,
            # per-sample IV size, default_KID[16]
            ver = data[body]
            if ver >= 1:
                packed = data[body + 5]
                rep.drm.pattern = f"{packed >> 4}:{packed & 0x0F}"
            rep.drm.iv_size = data[body + 7]
            kid = data[body + 8 : body + 24].hex()
            if kid != "00" * 16 and kid not in rep.drm.key_ids:
                rep.drm.key_ids.append(_pretty_uuid(kid))
        elif btype == b"pssh" and body + 20 <= body_end:
            sysid = _uuid_hex(data, body + 4)
            name = DRM_SYSTEMS.get(sysid, f"unknown system {_pretty_uuid(sysid)}")
            if name not in rep.drm.systems:
                rep.drm.systems.append(name)
            # pssh v1 carries the KID list in the clear, ahead of the blob
            if data[body] >= 1 and body + 24 <= body_end:
                count = struct.unpack_from(">I", data, body + 20)[0]
                for k in range(min(count, 16)):
                    off = body + 24 + k * 16
                    if off + 16 > body_end:
                        break
                    kid = _pretty_uuid(_uuid_hex(data, off))
                    if kid not in rep.drm.key_ids:
                        rep.drm.key_ids.append(kid)
        elif btype == b"uuid" and body + 16 <= body_end:
            hit = PIFF_UUIDS.get(_uuid_hex(data, body))
            if hit and hit not in rep.evidence:
                rep.evidence.append(f"PIFF/Smooth extension box: {hit}")
                if "encryption" in hit and not rep.drm.scheme:
                    rep.drm.scheme = "piff"
        elif btype in (b"senc", b"saiz", b"saio"):
            rep.drm.per_sample = True
            scheme = scheme or "cenc"
        elif btype == b"sinf":
            scheme = scheme or "cenc"

    rep.is_init = saw_moov and not saw_mdat
    if scheme:
        rep.encrypted = scheme
        rep.drm.scheme = rep.drm.scheme or scheme
        rep.evidence.append("protection: " + (rep.drm.describe() or scheme))

    if tracks:
        rep.evidence.append(f"moov declares {len(tracks)} trak(s)")
        for t in tracks:
            t.observed = bool(saw_mdat)
    else:
        # Media segment with no init segment. moof/traf says how many tracks are
        # interleaved but never what kind they are -- hdlr lives in the moov.
        # Two independent recoveries, in order of reliability:
        #   1. the sample table in each trun (timing + size distribution)
        #   2. codec syncwords in the mdat payload
        # (1) is the load-bearing one: raw AAC in a CMAF mdat has no syncword at
        # all, so payload sniffing alone silently fails on every audio segment.
        trafs = _scan_trafs(data)
        nmoof = sum(1 for bt, _b, _e, _d in _mp4_boxes(data) if bt == b"moof")
        rep.evidence.append(
            f"{nmoof} moof(s) covering {len(trafs)} track(s): track_id "
            + ", ".join(str(t.track_id) for t in trafs)
        )
        for info in trafs:
            t = Track(kind=Kind.UNKNOWN, track_id=info.track_id, observed=True)
            kind, codec, why, _score = _classify_traf(info)
            t.kind, t.codec = kind, codec
            t.packets = len(info.sizes) or info.sample_count
            t.note = why
            tracks.append(t)

        if mdat_range and not rep.encrypted:
            head = data[mdat_range[0] : min(mdat_range[1], mdat_range[0] + 8192)]
            guess: tuple[Kind, str] | None = None
            nal_headers: list[int] = []
            # The mdat box header states its true size even when our Range
            # request cut the payload short. Bounding by the bytes we happen to
            # hold would reject a legitimate NAL in any segment larger than the
            # prefix we fetched -- which is the normal case for this tool.
            mdat_len = mdat_range[1] - mdat_range[0]
            hdr_at = mdat_range[0] - 8
            if hdr_at >= 0:
                declared = struct.unpack_from(">I", data, hdr_at)[0]
                if declared == 1 and hdr_at - 8 >= 0:
                    declared = struct.unpack_from(">Q", data, mdat_range[0] - 16)[0]
                    mdat_len = max(mdat_len, declared - 16)
                elif declared > 8:
                    mdat_len = max(mdat_len, declared - 8)
            if _vtt_sample_chain(head):
                guess = (Kind.TEXT, "webvtt")
            elif _is_xml_text(head):
                guess = (Kind.TEXT, "ttml")
            elif _nal_chain(head, probes=12, headers=nal_headers, max_len=mdat_len):
                guess = (Kind.VIDEO, _guess_nal_codec(nal_headers))
            elif _adts_chain(head):
                guess = (Kind.AUDIO, "aac-adts")
            elif head[:2] == b"\x0b\x77":
                guess = (Kind.AUDIO, "ac3/eac3")
            elif head[:4] == b"OggS":
                guess = (Kind.AUDIO, "ogg")
            elif head[:6] == b"WEBVTT":
                guess = (Kind.TEXT, "webvtt")
            if guess:
                rep.evidence.append(f"mdat payload syncword: {guess[1]}")
                if len(tracks) == 1:
                    was = tracks[0].kind
                    if was is Kind.UNKNOWN:
                        tracks[0].kind, tracks[0].codec = guess
                        tracks[0].note = "kind from mdat syncword"
                    elif was is guess[0]:
                        tracks[0].codec = guess[1]
                        tracks[0].note += "; mdat syncword agrees"
                    else:
                        # Direct observation of the elementary stream beats a
                        # statistical read of the sample table. All-intra video
                        # is the case that lands here: its sample table looks
                        # like VBR audio, its payload is unmistakably NALs.
                        tracks[0].kind, tracks[0].codec = guess
                        tracks[0].note = (
                            f"mdat syncword says {guess[0].value}, overriding "
                            f"sample-table guess of {was.value} ({tracks[0].note})"
                        )
                        rep.evidence.append(
                            f"sample table suggested {was.value}; mdat payload "
                            f"overrides -- trusting the observed elementary stream"
                        )
            elif len(tracks) == 1 and tracks[0].kind is not Kind.UNKNOWN:
                rep.evidence.append(
                    "no mdat syncword (expected: CMAF stores raw access units)"
                )

    rep.tracks = tracks
    if saw_moov:
        rep.confidence = "high"
    elif saw_moof and any(t.kind in (Kind.VIDEO, Kind.AUDIO, Kind.TEXT) for t in tracks):
        rep.confidence = "medium"
    elif saw_moof:
        rep.confidence = "low"
    if saw_sidx and not saw_moov and not saw_moof:
        rep.evidence.append("sidx-only: this is an index/segment-index file")



# --------------------------------------------------------------------------
# DRM / encryption
# --------------------------------------------------------------------------
#
# Common Encryption (ISO/IEC 23001-7) is deliberately designed so that a player
# can parse the container without holding any key: boxes, the sample table, and
# the track handlers all stay in the clear, and only the media samples are
# encrypted. That means a DRM-protected fragment classifies exactly as well as
# a clear one -- and on top of that, the protection metadata itself is readable,
# so the DRM system and key id can be reported too.
#
# The one exception is HLS EXT-X-KEY:METHOD=AES-128, which encrypts the whole
# segment as an opaque blob. Nothing is recoverable there without the key, and
# guessing would be worse than saying so.

DRM_SYSTEMS = {
    "edef8ba979d64acea3c827dcd51d21ed": "Widevine",
    "9a04f07998404286ab92e65be0885f95": "PlayReady",
    "94ce86fb07ff4f43adb893d2fa968ca2": "FairPlay",
    "1077efecc0b24d02ace33c1e52e2fb4b": "ClearKey (W3C common)",
    "5e629af538da4063897797ffbd9902d4": "Marlin",
    "f239e769efa348509c16a903c6932efb": "Adobe PrimeTime",
    "3d5e6d359b9a41e8b843dd3c6e72c42c": "WisePlay (Huawei)",
    "adb41c242dbf4a6d958b4457c0d27b95": "Nagra",
    "80a6be7e14484c379e70d5aebe04c8d2": "Irdeto",
    "9a27dd82fde247258cbc4234aa06ec09": "Verimatrix VCAS",
    "a68129d3575b4f1a9cba3223846cf7c3": "Viaccess-Orca",
    "6dd8b3c345f44a688f2f5bfd5f5a9781": "SecureMedia",
    "793b79569f944946a94223e7ef7e44b4": "Vualto",
}

# 23001-7 protection schemes: full-sample vs pattern, CTR vs CBC.
CENC_SCHEMES = {
    "cenc": "AES-CTR, full sample",
    "cbc1": "AES-CBC, full sample",
    "cens": "AES-CTR, pattern",
    "cbcs": "AES-CBC, pattern (FairPlay / HLS fMP4)",
    "piff": "PIFF legacy (Smooth Streaming)",
}

# PIFF/Smooth Streaming extension boxes, carried as uuid boxes.
PIFF_UUIDS = {
    "a2394f525a9b4f14a2446c427c648df4": "PIFF sample encryption",
    "8974dbce7be74c5184f97148f9882554": "PIFF track encryption",
    "d08a4f1810f34a82b6c832d8aba183d3": "PIFF protection header",
    "6d1d9b0542d544e680e2141daff757b2": "Smooth tfxd (fragment timing)",
    "d4807ef2ca3946958e5426cb9e46a79f": "Smooth tfrf (next fragment)",
}


@dataclass
class Drm:
    """Everything recoverable about protection without holding a key."""
    scheme: str = ""                                  # cenc | cbc1 | cens | cbcs
    systems: list[str] = field(default_factory=list)  # Widevine, PlayReady, ...
    key_ids: list[str] = field(default_factory=list)  # default_KID / ContentEncKeyID
    iv_size: int = 0
    pattern: str = ""                                 # e.g. "1:9" for cbcs
    per_sample: bool = False                          # senc/saiz present
    note: str = ""

    def __bool__(self) -> bool:
        return bool(self.scheme or self.systems or self.key_ids or self.per_sample)

    def describe(self) -> str:
        bits = []
        if self.scheme:
            detail = CENC_SCHEMES.get(self.scheme)
            bits.append(f"{self.scheme} ({detail})" if detail else self.scheme)
        if self.pattern:
            bits.append(f"pattern {self.pattern}")
        if self.systems:
            bits.append("systems: " + ", ".join(self.systems))
        elif self.scheme in CENC_SCHEMES:
            # pssh may legitimately live only in the MPD/playlist, so its
            # absence is not a parse failure -- say where to look instead.
            bits.append("no in-band pssh (DRM system declared in the manifest)")
        if self.key_ids:
            bits.append("KID " + ", ".join(self.key_ids))
        if self.iv_size:
            bits.append(f"IV {self.iv_size}B")
        if self.note:
            bits.append(self.note)
        return "; ".join(bits)


def _uuid_hex(data: bytes, off: int) -> str:
    return data[off : off + 16].hex()


def _pretty_uuid(hexid: str) -> str:
    return (f"{hexid[:8]}-{hexid[8:12]}-{hexid[12:16]}-{hexid[16:20]}-{hexid[20:]}")


# --------------------------------------------------------------------------
# HLS packed audio (bare elementary streams)
# --------------------------------------------------------------------------

def _id3_len(data: bytes) -> int:
    """Total size of an ID3v2 tag at offset 0, including the 10-byte header."""
    if data[:3] != b"ID3" or len(data) < 10:
        return 0
    b = data[6:10]
    if any(x & 0x80 for x in b):
        return 0
    return 10 + ((b[0] << 21) | (b[1] << 14) | (b[2] << 7) | b[3])


MPEG_AUDIO_VERSIONS = {0: "mpeg2.5", 2: "mpeg2", 3: "mpeg1"}


def _parse_packed_audio(data: bytes, rep: Report, id3: int) -> bool:
    body = data[id3:]
    if id3:
        rep.evidence.append(f"ID3v2 tag of {id3} bytes at offset 0 (HLS packed-audio marker)")
    if _adts_chain(body):
        rep.container = "adts"
        prof = ((body[2] >> 6) & 0x03) + 1
        rep.tracks = [Track(kind=Kind.AUDIO, codec=f"aac-adts(profile {prof})",
                            declared=True, observed=True)]
        rep.evidence.append("ADTS syncword chain validated across 3 frames")
        rep.confidence = "high"
        return True
    if body[:2] == b"\x0b\x77":
        rep.container = "ac3"
        bsid = body[5] >> 3 if len(body) > 5 else 8
        rep.tracks = [Track(kind=Kind.AUDIO, codec="eac3" if bsid > 10 else "ac3",
                            declared=True, observed=True)]
        rep.evidence.append(f"AC-3 syncword 0x0B77, bsid {bsid}")
        rep.confidence = "high"
        return True
    chain = _mpeg_audio_chain(body, 0)
    if chain:
        rep.container = "mpeg-audio"
        rep.tracks = [Track(kind=Kind.AUDIO, codec=chain[1],
                            declared=True, observed=True)]
        rep.evidence.append(
            f"MPEG audio frame chain validated across {chain[0]} frames: {chain[1]}")
        rep.confidence = "high"
        return True
    if body[:4] == b"fLaC":
        rep.container, rep.confidence = "flac", "high"
        rep.tracks = [Track(kind=Kind.AUDIO, codec="flac", declared=True, observed=True)]
        return True
    if body[:4] == b"OggS":
        rep.container, rep.confidence = "ogg", "medium"
        rep.tracks = [Track(kind=Kind.AUDIO, codec="vorbis/opus", declared=True, observed=True)]
        return True
    if id3:
        # ID3 present but nothing chain-validated at offset 0. Before settling
        # for "packed audio of unknown codec", scan: the ID3 tag length can be
        # miscomputed by the packager, or padding can sit between the tag and
        # the first frame.
        hit = _resync(body)
        if hit:
            offset, kind, codec = hit
            rep.container = "packed-audio"
            rep.tracks = [Track(kind=kind, codec=codec, declared=True, observed=True,
                                note=f"first frame {offset} bytes after the ID3 tag")]
            rep.evidence.append(f"validated {codec} after resync past the ID3 tag")
            rep.confidence = "high"
            return True
        rep.container = "packed-audio"
        rep.tracks = [Track(kind=Kind.AUDIO, codec="unknown", declared=True, observed=True,
                            note="ID3-prefixed but no frame chain validated")]
        rep.confidence = "low"
        return True
    return False



# --------------------------------------------------------------------------
# MPEG audio frame parsing and resync
# --------------------------------------------------------------------------
#
# A continuous Icecast/SHOUTcast stream has no container, no header and no
# segment boundary: a read joins wherever the encoder happens to be, which is
# almost never a frame boundary. Checking offset 0 for a syncword is therefore
# wrong for exactly the streams that have nothing else to identify them.
#
# Scanning for a syncword is not enough either -- 0xFFEx occurs about once every
# 4 KB in arbitrary data. The frame length has to be computed and the *next*
# syncword confirmed at the predicted offset, several times over.

# ISO/IEC 11172-3 and 13818-3 bitrate tables, in kbit/s.
_BR_V1 = {
    1: (0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448),
    2: (0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384),
    3: (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320),
}
_BR_V2 = {
    1: (0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256),
    2: (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
}
_BR_V2[3] = _BR_V2[2]
_SR = {3: (44100, 48000, 32000), 2: (22050, 24000, 16000), 0: (11025, 12000, 8000)}


def _mpeg_audio_frame(hdr: bytes) -> tuple[int, str] | None:
    """Length and description of the MPEG audio frame starting at hdr[0:4]."""
    if len(hdr) < 4 or hdr[0] != 0xFF or (hdr[1] & 0xE0) != 0xE0:
        return None
    ver = (hdr[1] >> 3) & 0x03            # 3 = MPEG1, 2 = MPEG2, 0 = MPEG2.5
    layer_bits = (hdr[1] >> 1) & 0x03
    if ver == 1 or layer_bits == 0:       # reserved
        return None
    layer = 4 - layer_bits                # 1, 2 or 3
    br_idx = (hdr[2] >> 4) & 0x0F
    sr_idx = (hdr[2] >> 2) & 0x03
    if br_idx in (0, 15) or sr_idx == 3:
        return None
    table = _BR_V1 if ver == 3 else _BR_V2
    bitrate = table[layer][br_idx] * 1000
    rate = _SR[ver][sr_idx]
    pad = (hdr[2] >> 1) & 0x01
    if layer == 1:
        length = (12 * bitrate // rate + pad) * 4
    else:
        # Layer III at MPEG2/2.5 uses 576 samples per frame, not 1152.
        per_frame = 72 if (layer == 3 and ver != 3) else 144
        length = per_frame * bitrate // rate + pad
    if length < 8:
        return None
    name = {3: "mpeg1", 2: "mpeg2", 0: "mpeg2.5"}[ver]
    return length, f"{name}-layer{layer} {bitrate // 1000}kbps {rate}Hz"


def _mpeg_audio_chain(buf: bytes, off: int = 0, probes: int = 4) -> tuple[int, str] | None:
    """Follow MPEG audio frame lengths from `off`. Returns (frames, description)
    only if each frame's length lands exactly on the next syncword."""
    i, ok, desc = off, 0, ""
    while ok < probes and i + 4 <= len(buf):
        hit = _mpeg_audio_frame(buf[i : i + 4])
        if not hit:
            return None
        length, desc = hit if not desc else (hit[0], desc)
        ok += 1
        if i + length + 4 > len(buf):
            return (ok, desc) if ok >= 2 else None
        i += length
    return (ok, desc) if ok >= probes else None


def _adts_frame_len(buf: bytes, i: int) -> int | None:
    if i + 7 > len(buf) or buf[i] != 0xFF or (buf[i + 1] & 0xF0) != 0xF0:
        return None
    if buf[i + 1] & 0x06:                 # layer must be 00 for ADTS
        return None
    if (buf[i + 2] >> 2) & 0x0F > 12:     # sampling_frequency_index 13-15 invalid
        return None
    n = ((buf[i + 3] & 0x03) << 11) | (buf[i + 4] << 3) | (buf[i + 5] >> 5)
    return n if 7 <= n <= 8192 else None


def _adts_key(buf: bytes, i: int) -> tuple:
    """Stream identity: profile, sample rate index, channel config. Constant for
    the life of a real stream, and the property random data cannot fake."""
    return (buf[i + 2] >> 6, (buf[i + 2] >> 2) & 0x0F,
            ((buf[i + 2] & 0x01) << 2) | (buf[i + 3] >> 6))


def _mpeg_key(buf: bytes, i: int) -> tuple:
    """MPEG audio version, layer and sample rate index -- likewise constant."""
    return ((buf[i + 1] >> 3) & 0x03, (buf[i + 1] >> 1) & 0x03, (buf[i + 2] >> 2) & 0x03)


def _frame_coverage(data: bytes, frame_fn, key_fn=None, sync: bytes = b"\xff",
                    limit: int = 65536) -> tuple[float, int, int]:
    """Fraction of `data` covered by validated frames, resyncing across gaps.

    Demanding N consecutive frames from a fixed offset is brittle against three
    things that all occur in real streams: joining mid-frame, a corrupt frame,
    and SHOUTcast ICY metadata, which is injected *into* the audio every
    `icy-metaint` bytes -- often only ~1 KB, leaving room for just two frames
    between blocks. Coverage handles all three uniformly, and is much harder to
    trigger on random data than a short consecutive run is.

    Coverage is accumulated per stream identity (sample rate, channel layout)
    and the best identity wins. That distinction is what separates the two
    things that both look like "broken chains":

      ICY-interleaved audio -- many runs, every one the same identity, because
      it is one stream with metadata punched into it. Sums to ~99%.
      random bytes -- many runs with scattered identities, because nothing ties
      them together. No single identity accumulates much.

    Summing everything indiscriminately lets noise creep toward the threshold;
    taking only the longest single run throws away real streams. Grouping by
    identity does neither.

    Returns (coverage, frame_count, first_frame_offset) for the best identity.
    """
    n = min(len(data), limit)
    by_key: dict[tuple, list] = {}          # key -> [covered, frames, first]
    i, guard = 0, 0
    while i < n and guard < 8192:
        guard += 1
        if frame_fn(data, i) is None:
            nxt = data.find(sync, i + 1, n)
            if nxt < 0:
                break
            i = nxt
            continue
        run_key = None
        run_cov = run_frames = 0
        run_start = i
        while i < n:
            length = frame_fn(data, i)
            if length is None or i + length > n:
                break
            if key_fn is not None:
                key = key_fn(data, i)
                if run_key is None:
                    run_key = key
                elif key != run_key:
                    # Sample rate or channel layout changed mid-run: real
                    # streams do not do this, random bytes constantly do.
                    break
            run_cov += length
            run_frames += 1
            i += length
        if run_frames:
            slot = by_key.setdefault(run_key, [0, 0, run_start])
            slot[0] += run_cov
            slot[1] += run_frames
        i += 1
    if not by_key or not n:
        return 0.0, 0, -1
    covered, frames, first = max(by_key.values(), key=lambda v: v[0])
    return covered / n, frames, first


def _mpeg_frame_len(buf: bytes, i: int) -> int | None:
    hit = _mpeg_audio_frame(buf[i : i + 4])
    return hit[0] if hit else None


# ATSC A/52 table 5.18: AC-3 frame size in 16-bit words, by frmsizecod and
# fscod (48 kHz, 44.1 kHz, 32 kHz). E-AC-3 instead carries an explicit frmsiz
# field, so the two are computed differently.
_AC3_SIZES = (
    (64, 69, 96), (64, 70, 96), (80, 87, 120), (80, 88, 120), (96, 104, 144),
    (96, 105, 144), (112, 121, 168), (112, 122, 168), (128, 139, 192),
    (128, 140, 192), (160, 174, 240), (160, 175, 240), (192, 208, 288),
    (192, 209, 288), (224, 243, 336), (224, 244, 336), (256, 278, 384),
    (256, 279, 384), (320, 348, 480), (320, 349, 480), (384, 417, 576),
    (384, 418, 576), (448, 487, 672), (448, 488, 672), (512, 557, 768),
    (512, 558, 768), (640, 696, 960), (640, 697, 960), (768, 835, 1152),
    (768, 836, 1152), (896, 975, 1344), (896, 976, 1344), (1024, 1114, 1536),
    (1024, 1115, 1536), (1152, 1253, 1728), (1152, 1254, 1728),
    (1280, 1393, 1920), (1280, 1394, 1920),
)


def _ac3_frame_len(buf: bytes, i: int) -> int | None:
    """Length of the AC-3 or E-AC-3 frame at buf[i]. Checking only for the 0x0B77
    syncword is not enough -- two arbitrary bytes match it every 64 KB, and that
    false positive classified an obfuscated video stream as AC-3 audio."""
    if i + 6 > len(buf) or buf[i] != 0x0B or buf[i + 1] != 0x77:
        return None
    bsid = buf[i + 5] >> 3
    if bsid <= 10:                                  # AC-3
        fscod, frmsizecod = buf[i + 4] >> 6, buf[i + 4] & 0x3F
        if fscod > 2 or frmsizecod > 37:
            return None
        return _AC3_SIZES[frmsizecod][fscod] * 2
    if bsid <= 16:                                  # E-AC-3
        frmsiz = ((buf[i + 2] & 0x07) << 8) | buf[i + 3]
        return (frmsiz + 1) * 2
    return None


def _ac3_key(buf: bytes, i: int) -> tuple:
    return (buf[i + 5] >> 3, buf[i + 4] >> 6)       # bsid, fscod


def _resync(data: bytes, scan: int = 16384) -> tuple[int, Kind, str] | None:
    """Find the first offset carrying a validated elementary-stream frame chain.

    Returns (offset, kind, codec) or None. Used for continuous streams joined
    mid-frame, where offset 0 is meaningless.
    """
    # Coverage first: it survives interleaved metadata and corrupt frames,
    # which a consecutive-run check does not.
    for frame_fn, key_fn, sync, codec in (
            (_adts_frame_len, _adts_key, b"\xff", "aac-adts"),
            (_mpeg_frame_len, _mpeg_key, b"\xff", "mpeg-audio"),
            (_ac3_frame_len, _ac3_key, b"\x0b", "ac3/eac3")):
        cover, frames, first = _frame_coverage(data, frame_fn, key_fn, sync)
        # Threshold sits in the measured gap: elementary-stream audio that
        # reaches this path covers 97-100%, while the worst non-audio case in
        # the corpus (an Annex-B H.264 stream read as MPEG audio) reaches 40%.
        if cover >= 0.75 and frames >= 8:
            desc = codec
            if codec == "mpeg-audio" and first >= 0:
                hit = _mpeg_audio_frame(data[first : first + 4])
                if hit:
                    desc = hit[1]
            return first, Kind.AUDIO, f"{desc} ({cover * 100:.0f}% frame coverage)"

    limit = min(len(data), scan)
    for i in range(limit):
        b = data[i]
        if b == 0xFF:
            nxt = data[i + 1] if i + 1 < len(data) else 0
            if (nxt & 0xF0) == 0xF0 and not (nxt & 0x06):
                if _adts_chain(data[i:], probes=4):
                    return i, Kind.AUDIO, "aac-adts"
            if (nxt & 0xE0) == 0xE0:
                hit = _mpeg_audio_chain(data, i)
                if hit:
                    return i, Kind.AUDIO, hit[1]
        elif b == 0x4F and data[i : i + 4] == b"OggS":
            return i, Kind.AUDIO, "ogg"
    return None


# --------------------------------------------------------------------------
# WebM / Matroska
# --------------------------------------------------------------------------

VTT_SAMPLE_BOXES = {b"vttc", b"vtte", b"vttx", b"vttC", b"payl", b"sttg", b"iden"}


def _is_xml_text(buf: bytes) -> bool:
    """True only for something that is actually XML/TTML.

    A bare "<" occurs in one byte out of 256, which is frequent enough that
    high-entropy payloads were being reported as subtitles. Require a real
    marker and ASCII-clean leading bytes.
    """
    head = buf[:256].lstrip()[:128]
    if not head.startswith(b"<"):
        return False
    if not all(32 <= c < 127 or c in (9, 10, 13) for c in head[:16]):
        return False
    return any(m in buf[:1024] for m in (b"<?xml", b"<tt", b"<TT", b"xmlns", b"<div", b"<p "))


def _vtt_sample_chain(buf: bytes) -> bool:
    """True if buf is a run of ISO/IEC 14496-30 WebVTT sample boxes."""
    i, ok = 0, 0
    while i + 8 <= len(buf) and ok < 4:
        size = struct.unpack_from(">I", buf, i)[0]
        if buf[i + 4 : i + 8] not in VTT_SAMPLE_BOXES or size < 8:
            return ok >= 1
        ok += 1
        i += size
    return ok >= 1


WEBM_TRACK_TYPES = {1: Kind.VIDEO, 2: Kind.AUDIO, 3: Kind.VIDEO, 0x10: Kind.TEXT,
                    0x11: Kind.TEXT, 0x12: Kind.DATA, 0x20: Kind.DATA}


def _ebml_num(data: bytes, i: int, keep_marker: bool) -> tuple[int, int]:
    if i >= len(data):
        return 0, 0
    first = data[i]
    if first == 0:
        return 0, 0
    length = 8 - first.bit_length() + 1
    if i + length > len(data):
        return 0, 0
    val = int.from_bytes(data[i : i + length], "big")
    if not keep_marker:
        val &= (1 << (7 * length)) - 1
    return val, length


def _parse_webm(data: bytes, rep: Report) -> None:
    rep.container = "webm/matroska"
    rep.evidence.append("EBML magic 1A45DFA3")
    tracks: list[Track] = []
    # Scan the element tree, descending only into the master elements on the
    # path to TrackEntry. Clusters are skipped wholesale.
    # Segment, Tracks, TrackEntry, then the ContentEncodings subtree that
    # carries WebM's equivalent of a sinf box.
    masters = {0x18538067, 0x1654AE6B, 0xAE, 0x6D80, 0x6240, 0x5035, 0x47E7}
    stack: list[tuple[int, int]] = [(0, len(data))]
    entry: Track | None = None
    i = 0
    guard = 0
    while i < len(data) and guard < 200_000:
        guard += 1
        eid, n1 = _ebml_num(data, i, True)
        if not n1:
            break
        size, n2 = _ebml_num(data, i + n1, False)
        if not n2:
            break
        body = i + n1 + n2
        if eid in masters:
            if eid == 0xAE:
                entry = Track(kind=Kind.UNKNOWN, declared=True, observed=True)
                tracks.append(entry)
            i = body
            continue
        if entry is not None and body + size <= len(data):
            if eid == 0x83 and size:                     # TrackType
                entry.kind = WEBM_TRACK_TYPES.get(data[body], Kind.UNKNOWN)
            elif eid == 0xD7 and size:                   # TrackNumber
                entry.track_id = int.from_bytes(data[body : body + size], "big")
            elif eid == 0x86:                            # CodecID
                entry.codec = data[body : body + size].decode("latin-1", "replace").strip("\x00")
            elif eid == 0x47E1 and size:                 # ContentEncAlgo
                algo = {0: "unencrypted", 1: "DES", 2: "3DES", 3: "Twofish",
                        4: "Blowfish", 5: "AES"}.get(data[body], f"algo {data[body]}")
                rep.drm.scheme = rep.drm.scheme or f"WebM ContentEncryption / {algo}"
                entry.note = (entry.note + f" encrypted ({algo})").strip()
            elif eid == 0x47E2 and size:                 # ContentEncKeyID
                kid = data[body : body + size].hex()
                if kid and kid not in rep.drm.key_ids:
                    rep.drm.key_ids.append(kid)
            elif eid == 0x47E8 and size:                 # AESSettingsCipherMode
                rep.drm.note = {1: "AES-CTR", 2: "AES-CBC"}.get(data[body], "")
        if eid == 0x1F43B675:                            # Cluster -- media payload
            for t in tracks:
                t.observed = True
            break
        i = body + size
        if size == 0 or i <= body - 1:
            break
    rep.tracks = tracks
    rep.is_init = bool(tracks) and b"\x1f\x43\xb6\x75" not in data[:65536]
    rep.confidence = "high" if tracks else "low"
    if rep.drm:
        rep.encrypted = rep.encrypted or "webm-encrypted"
        rep.evidence.append("protection: " + rep.drm.describe())



# --------------------------------------------------------------------------
# FLV (RTMP ingest / recording)
# --------------------------------------------------------------------------

FLV_TAG_KINDS = {8: (Kind.AUDIO, "flv-audio"), 9: (Kind.VIDEO, "flv-video"),
                 18: (Kind.DATA, "flv-script")}
FLV_AUDIO_CODECS = {0: "pcm", 1: "adpcm", 2: "mp3", 4: "nellymoser16", 5: "nellymoser8",
                    6: "nellymoser", 7: "g711a", 8: "g711u", 10: "aac", 11: "speex",
                    14: "mp3-8k", 15: "device-specific"}
FLV_VIDEO_CODECS = {2: "h263", 3: "screen", 4: "vp6", 5: "vp6-alpha", 6: "screen2",
                    7: "h264", 12: "hevc", 13: "av1"}


def _parse_flv(data: bytes, rep: Report) -> None:
    """FLV declares a/v presence in its 9-byte header, then repeats it in the
    tag stream. Both are checked: a muxer that sets the header flags but writes
    only one kind of tag is common in RTMP ingest."""
    rep.container = "flv"
    flags = data[4]
    declared = {Kind.VIDEO: bool(flags & 0x01), Kind.AUDIO: bool(flags & 0x04)}
    rep.evidence.append(
        f"FLV header flags 0x{flags:02x}: declares"
        f"{' video' if declared[Kind.VIDEO] else ''}"
        f"{' audio' if declared[Kind.AUDIO] else ''}" or "nothing")

    tracks: dict[Kind, Track] = {}
    for kind, present in declared.items():
        if present:
            tracks[kind] = Track(kind=kind, declared=True)

    offset = struct.unpack_from(">I", data, 5)[0] if len(data) >= 9 else 9
    i = max(offset, 9) + 4                     # skip PreviousTagSize0
    seen = 0
    while i + 11 <= len(data) and seen < 400:
        tag = data[i] & 0x1F
        size = int.from_bytes(data[i + 1 : i + 4], "big")
        payload = i + 11
        hit = FLV_TAG_KINDS.get(tag)
        if hit and payload < len(data):
            kind, codec = hit
            t = tracks.setdefault(kind, Track(kind=kind))
            t.observed, t.packets = True, t.packets + 1
            if not t.codec or t.codec.startswith("flv-"):
                b = data[payload]
                if kind is Kind.AUDIO:
                    t.codec = FLV_AUDIO_CODECS.get(b >> 4, f"audio-{b >> 4}")
                elif kind is Kind.VIDEO:
                    t.codec = FLV_VIDEO_CODECS.get(b & 0x0F, f"video-{b & 0x0F}")
                else:
                    t.codec = codec
        if size == 0 or size > len(data):
            break
        i = payload + size + 4                 # payload + PreviousTagSize
        seen += 1

    for kind, t in tracks.items():
        if t.declared and not t.observed:
            t.note = "DECLARED IN HEADER BUT NO TAGS SEEN"
        elif t.observed and not t.declared:
            t.note = "tags present but not declared in the header"
        if t.packets:
            t.note = (t.note + f" {t.packets} tags").strip()
    rep.tracks = list(tracks.values())
    rep.confidence = "high" if any(t.observed for t in tracks.values()) else "medium"


# --------------------------------------------------------------------------
# Headless payloads: LL-HLS partial segments and raw elementary streams
# --------------------------------------------------------------------------

def _parse_headless(data: bytes, rep: Report) -> bool:
    """Classify a fragment that begins with media data and no container header.

    This is the shape of an LL-HLS partial segment (EXT-X-PART) that is not the
    first part of its parent segment: the moof went out in an earlier part, so
    this one opens in the middle of an mdat. Byte-range requests into a single
    packed file land here too, as do raw elementary streams.
    """
    head = data[:8192]

    if head[:3] == b"\x00\x00\x01" or head[:4] == b"\x00\x00\x00\x01":
        headers = []
        i = 0
        while i + 4 < len(head) and len(headers) < 24:
            if head[i : i + 3] == b"\x00\x00\x01":
                hdr = head[i + 3]
                # forbidden_zero_bit clear, and a type valid under either codec
                if not hdr & 0x80 and (1 <= hdr & 0x1F <= 23 or (hdr >> 1) & 0x3F <= 47):
                    headers.append(hdr)
                i += 4
            else:
                i += 1
        if len(headers) >= 3:
            codec = _guess_nal_codec(headers)
            rep.container = "annex-b elementary stream"
            rep.tracks = [Track(kind=Kind.VIDEO, codec=codec, observed=True,
                                note=f"{len(headers)} NAL units scanned")]
            rep.evidence.append(
                f"{len(headers)} Annex-B start codes with valid NAL headers")
            rep.confidence = "high" if codec != "h264/hevc" else "medium"
            return True

    # Audio frame coverage before the NAL heuristic: 99% of the buffer walking
    # cleanly as MPEG frames is far stronger evidence than four bytes that
    # happen to read as a NAL length. A live MP3 stream was classified as video
    # when this ran in the other order.
    hit = _resync(data)
    if hit:
        offset, kind, codec = hit
        rep.container = "elementary stream (resynced)"
        rep.tracks = [Track(kind=kind, codec=codec, observed=True,
                            note=f"frame chain starts at offset {offset}")]
        rep.evidence.append(
            f"no container header; validated {codec} at offset {offset} -- "
            "continuous stream joined mid-frame")
        rep.confidence = "high" if offset < 4096 else "medium"
        return True

    headers = []
    if _nal_chain(head, probes=12, headers=headers, max_len=len(data)):
        rep.container = "headless media payload"
        rep.tracks = [Track(kind=Kind.VIDEO, codec=_guess_nal_codec(headers), observed=True,
                            note="length-prefixed NAL run with no container header")]
        rep.evidence.append(
            "length-prefixed NAL chain at offset 0 -- looks like an LL-HLS "
            "partial segment continuing a previous part's mdat")
        rep.confidence = "medium"
        return True

    if _adts_chain(head):
        rep.container = "headless media payload"
        rep.tracks = [Track(kind=Kind.AUDIO, codec="aac-adts", observed=True,
                            note="ADTS frame run with no container header")]
        rep.evidence.append("ADTS frame chain at offset 0, no container header")
        rep.confidence = "medium"
        return True

    return False


# --------------------------------------------------------------------------
# Top-level dispatch
# --------------------------------------------------------------------------

def _entropy(buf: bytes) -> float:
    if not buf:
        return 0.0
    import math
    counts = [0] * 256
    for b in buf:
        counts[b] += 1
    n = len(buf)
    return -sum((c / n) * math.log2(c / n) for c in counts if c)


def sniff(data: bytes) -> Report:
    """Classify a media fragment from its bytes. Feed it the whole fragment, or
    at least the first ~256 KB; TS needs enough packets to see the PMT and a
    representative packet mix."""
    rep = Report()
    if not data:
        rep.verdict = "empty"
        return rep

    # Some origins serve playlists and even segments gzipped, and not every
    # client decompresses transparently. Compressed bytes look like noise, and
    # noise is exactly what produces spurious container hits, so unwrap first.
    if data[:3] == b"\x1f\x8b\x08":
        try:
            import gzip
            inner = gzip.decompress(data)
        except (OSError, EOFError, zlib.error):
            # A truncated gzip stream still decompresses up to the cut point.
            try:
                inner = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(data)
            except zlib.error:
                inner = b""
        if inner:
            rep = sniff(inner)
            rep.evidence.insert(0, f"gzip-compressed ({len(data)} -> {len(inner)} bytes)")
            return rep
        rep.container = "gzip"
        rep.verdict = "gzip stream that could not be decompressed"
        return rep

    head = data[:16]

    if data[:7] == b"#EXTM3U" or data.lstrip()[:7] == b"#EXTM3U":
        rep.container, rep.is_playlist, rep.confidence = "m3u8-playlist", True, "high"
        rep.verdict = "playlist (not a media fragment)"
        rep.evidence.append("#EXTM3U at offset 0")
        return rep
    if b"<MPD" in data[:4096]:
        rep.container, rep.is_playlist, rep.confidence = "dash-mpd", True, "high"
        rep.verdict = "manifest (not a media fragment)"
        return rep

    # Container detection first. A CMAF fragment carrying TTML or WebVTT is an
    # ISOBMFF file with a text track -- not a raw subtitle file -- and answering
    # "ttml" for it loses the track structure. Only fall through to the text
    # heuristics once no container magic matched.
    if len(data) >= 8 and data[4:8] in ISOBMFF_TOP:
        _parse_mp4(data, rep)
        return _verdict(rep)

    layout = _ts_layout(data)
    if layout:
        _parse_ts(data, layout[0], layout[1], rep)
        return _verdict(rep)

    if data[:4] == b"\x1a\x45\xdf\xa3":
        _parse_webm(data, rep)
        return _verdict(rep)

    if data[:6] == b"WEBVTT" or (_id3_len(data) and data[_id3_len(data):][:6] == b"WEBVTT"):
        rep.container, rep.confidence = "webvtt", "high"
        rep.tracks = [Track(kind=Kind.TEXT, codec="webvtt", declared=True, observed=True)]
        rep.evidence.append("WEBVTT signature")
        return _verdict(rep)
    if b"<tt" in data[:2048] and b"ttml" in data[:4096].lower():
        rep.container, rep.confidence = "ttml", "high"
        rep.tracks = [Track(kind=Kind.TEXT, codec="ttml/imsc", declared=True, observed=True)]
        return _verdict(rep)

    if data[:3] == b"FLV" and len(data) >= 9:
        _parse_flv(data, rep)
        return _verdict(rep)

    id3 = _id3_len(data)
    if _parse_packed_audio(data, rep, id3):
        return _verdict(rep)

    # No container header at all. Before calling it opaque, check whether the
    # bytes are simply media that started mid-stream.
    if _parse_headless(data, rep):
        return _verdict(rep)

    # Nothing matched. Whole-segment AES-128 (HLS EXT-X-KEY METHOD=AES-128) turns
    # a fragment into opaque high-entropy bytes whose length is a multiple of 16.
    ent = _entropy(data[:65536])
    if len(data) % 16 == 0 and ent > 7.5:
        # High entropy is consistent with AES-128 whole-segment encryption, but
        # it is equally consistent with any compressed payload -- 320 kbps MP3
        # measures ~7.99. This is the most likely explanation for a fragment
        # pulled from an HLS playlist, and it is still only an inference, so it
        # is reported as one. Confirm it with EXT-X-KEY in the playlist.
        rep.container, rep.encrypted, rep.confidence = "opaque", "aes-128-full", "low"
        rep.drm.scheme = "aes-128-full (inferred)"
        rep.drm.note = "inferred from entropy alone; confirm with EXT-X-KEY"
        rep.verdict = ("opaque high-entropy bytes -- most likely AES-128 "
                       "whole-segment encryption; no structure to classify")
        rep.evidence.append(
            f"entropy {ent:.2f} bits/byte, length {len(data)} is a multiple of 16, "
            "and no container or frame chain was found")
        return rep

    # A dead or geofenced stream URL usually redirects to an HTML error page.
    # Saying so is far more actionable than "unknown".
    probe = data[:1024].lstrip()[:512].lower()
    if probe.startswith((b"<!doctype html", b"<html", b"<?xml")) or b"<head" in probe:
        rep.container, rep.confidence = "html", "high"
        rep.verdict = "HTML page, not media (dead link, redirect, or error page)"
        return rep
    if probe[:1] in (b"{", b"[") and b'"' in probe:
        rep.container, rep.confidence = "json", "high"
        rep.verdict = "JSON, not media (likely an API error response)"
        return rep

    # Plain-text error bodies are common: one origin in the sample answers with
    # a literal shrug emoticon. Naming it beats reporting "unknown".
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = ""
    if text and len(data) < 1 << 16:
        printable = sum(1 for c in text if c.isprintable() or c in "\r\n\t")
        if printable / len(text) > 0.95:
            rep.container, rep.confidence = "text", "high"
            snippet = " ".join(text.split())[:48]
            rep.verdict = f"plain text, not media (likely an error body: {snippet!r})"
            return rep

    rep.evidence.append(f"no signature matched; entropy {ent:.2f} bits/byte")
    rep.verdict = "unknown"
    return rep


def _verdict(rep: Report) -> Report:
    obs = rep.kinds(observed_only=True)
    dec = rep.kinds()
    # On a complete fragment, what the bytes actually carry beats what the index
    # claims. On a prefix we cannot make that argument: a track we have not seen
    # yet may simply appear later, so declaration and observation are unioned.
    kinds = (obs | dec) if rep.truncated else (obs or dec)

    av = {k for k in kinds if k in (Kind.VIDEO, Kind.AUDIO)}
    if av == {Kind.VIDEO, Kind.AUDIO}:
        rep.verdict = "muxed (audio+video)"
    elif av == {Kind.VIDEO}:
        rep.verdict = "video-only"
    elif av == {Kind.AUDIO}:
        rep.verdict = "audio-only"
    elif Kind.TEXT in kinds:
        rep.verdict = "subtitles/text-only"
    elif any(t.kind is Kind.DATA for t in rep.tracks):
        rep.verdict = "metadata-only"
    elif any(t.kind is Kind.UNKNOWN for t in rep.tracks):
        # There is a track here; we simply cannot say what it carries. That is
        # a different statement from "no media", and conflating the two hides
        # the one case where this tool genuinely cannot answer: an unreadable
        # payload whose sample table has no usable tells.
        n = sum(1 for t in rep.tracks if t.kind is Kind.UNKNOWN)
        rep.verdict = (f"{n} track(s) present, kind undetermined "
                       "-- fetch the init segment to resolve")
    else:
        rep.verdict = "no identifiable media tracks"

    if rep.is_init:
        rep.verdict += " [init segment]"
    if rep.encrypted:
        rep.verdict += f" [{rep.encrypted}]"
    if obs and dec and obs != dec:
        missing = ", ".join(sorted(k.value for k in dec - obs))
        extra = ", ".join(sorted(k.value for k in obs - dec))
        detail = []
        if missing:
            detail.append(
                f"declares {missing}, not seen in this prefix" if rep.truncated
                else f"declares {missing} but carries none")
        if extra:
            detail.append(f"carries undeclared {extra}")
        rep.verdict += "  <- " + "; ".join(detail)
        if not rep.truncated:
            rep.confidence = "medium" if rep.confidence == "high" else rep.confidence
    return rep


def format_report(name: str, rep: Report) -> str:
    lines = [
        f"{name}",
        f"  container : {rep.container}"
        + (f"   brands: {' '.join(rep.brands)}" if rep.brands else ""),
        f"  verdict   : {rep.verdict}   (confidence: {rep.confidence})",
    ]
    if rep.tracks:
        lines.append("  tracks    :")
        lines += [f"      {t}" for t in rep.tracks]
    if rep.evidence:
        lines.append("  evidence  :")
        lines += [f"      - {e}" for e in rep.evidence]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Resolving a stream URL down to fragments
# --------------------------------------------------------------------------
#
# sniff() above is pure: bytes in, Report out, no I/O, no imports beyond the
# stdlib basics. This section is a thin layer on top for the form the question
# usually arrives in -- you have a stream URL, not a fragment on disk, and what
# you want to know is which of its renditions carry what.
#
# It walks a master playlist or MPD down to one fragment per rendition and
# classifies each, then shows what the manifest *claimed* beside what the bytes
# actually carry. Those two disagree often enough in the wild that showing both
# is the point rather than a debugging aid.

_UA = "mediasniff/1.0 (+https://github.com/FerventOrange/mediasniff)"

_HLS_MEDIA_KINDS = {"AUDIO": "audio", "SUBTITLES": "subtitles",
                    "CLOSED-CAPTIONS": "closed captions"}

# Used only to turn a CODECS attribute into a comparable claim. Deliberately
# not exhaustive -- it needs to answer "video, audio, or both", nothing finer.
_CODECS_VIDEO = ("avc1", "avc2", "avc3", "avc4", "hvc1", "hev1", "dvh1", "dvhe",
                 "vp08", "vp09", "vp8", "vp9", "av01", "mp4v", "vvc1", "vvi1")
_CODECS_AUDIO = ("mp4a", "ac-3", "ec-3", "ac-4", "opus", "flac", "alac", "dts",
                 "mp3", "mha1", "mhm1")


def _claim_from_codecs(codecs: str) -> str:
    """What a CODECS attribute claims the fragment carries."""
    low = codecs.lower()
    has_v = any(c in low for c in _CODECS_VIDEO)
    has_a = any(c in low for c in _CODECS_AUDIO)
    if has_v and has_a:
        return "muxed"
    if has_v:
        return "video"
    if has_a:
        return "audio"
    return codecs or "?"


@dataclass
class Rendition:
    """One selectable stream within a manifest."""
    label: str
    url: str
    declared: str = ""          # what the manifest says it carries
    note: str = ""

    def __str__(self) -> str:
        return self.label


def _http_get(url: str, nbytes: int | None = None, timeout: float = 15.0,
              insecure: bool = False) -> tuple[bytes, str]:
    """GET a URL, optionally only its first `nbytes`. Returns (body, final_url).

    Handles the two things that otherwise produce mystery failures: origins that
    gzip a playlist without the client decoding it, and origins with broken TLS.
    """
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "*/*"})
    if nbytes:
        req.add_header("Range", f"bytes=0-{nbytes - 1}")
    ctx = None
    if insecure:
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as fh:
        body = fh.read(nbytes or (8 << 20))
        if fh.headers.get("Content-Encoding", "").lower() == "gzip":
            try:
                body = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(body)
            except zlib.error:
                pass
        return body, fh.geturl()


def _attr(line: str, name: str) -> str:
    import re as _re
    hit = _re.search(name + r'="([^"]*)"', line) or _re.search(name + r"=([^,\s]+)", line)
    return hit.group(1) if hit else ""


def hls_master_renditions(text: str, base: str) -> list[Rendition]:
    """Every selectable rendition in an HLS master playlist.

    An EXT-X-MEDIA entry *without* a URI is skipped deliberately: per RFC 8216
    that rendition is already present inside the variant streams, so it has no
    fragments of its own to fetch.
    """
    import urllib.parse

    out: list[Rendition] = []
    lines = text.splitlines()
    external_audio_groups = {
        _attr(ln, "GROUP-ID") for ln in lines
        if ln.startswith("#EXT-X-MEDIA:") and "TYPE=AUDIO" in ln and _attr(ln, "URI")
    }
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("#EXT-X-MEDIA:"):
            uri = _attr(line, "URI")
            kind = _HLS_MEDIA_KINDS.get(_attr(line, "TYPE"), "")
            if not uri or not kind:
                continue
            name = _attr(line, "NAME") or _attr(line, "LANGUAGE") or "?"
            lang = _attr(line, "LANGUAGE")
            label = f"{kind}: {name}" + (f" ({lang})" if lang and lang != name else "")
            channels = _attr(line, "CHANNELS")
            out.append(Rendition(label, urllib.parse.urljoin(base, uri), kind,
                                 f"{channels}ch" if channels else ""))
        elif line.startswith("#EXT-X-I-FRAME-STREAM-INF"):
            uri = _attr(line, "URI")
            if uri:
                res = _attr(line, "RESOLUTION")
                out.append(Rendition(f"trickplay {res}".strip(),
                                     urllib.parse.urljoin(base, uri),
                                     "video", "I-frame only"))
        elif line.startswith("#EXT-X-STREAM-INF") and i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            if not nxt or nxt.startswith("#"):
                continue
            res = _attr(line, "RESOLUTION")
            codecs = _attr(line, "CODECS")
            claim = _claim_from_codecs(codecs)
            # An EXT-X-MEDIA group without a URI means that rendition already
            # lives inside this variant, so AUDIO= alone does not move it out.
            if claim == "muxed" and _attr(line, "AUDIO") in external_audio_groups:
                claim = "video"
            out.append(Rendition(f"variant {res or 'audio-only'}",
                                 urllib.parse.urljoin(base, nxt), claim, codecs))
    return out


def hls_media_target(text: str, base: str, prefer_media: bool = False) -> tuple[str, str]:
    """Pick one fragment to fetch from an HLS media playlist.

    The EXT-X-MAP init segment is preferred when present: it carries the moov,
    so the track kinds are read from a field rather than inferred. Pass
    prefer_media=True to target a media fragment instead, which exercises the
    harder init-less path.
    """
    import urllib.parse

    init = ""
    segments: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#EXT-X-MAP:"):
            uri = _attr(line, "URI")
            if uri:
                init = urllib.parse.urljoin(base, uri)
        elif line and not line.startswith("#"):
            segments.append(urllib.parse.urljoin(base, line))
    if init and not prefer_media:
        return init, "init segment"
    if segments:
        # The last listed segment is the freshest on a live playlist; the first
        # is the one most likely to roll off mid-fetch.
        return segments[-1], "media segment"
    return (init, "init segment") if init else ("", "")


def mpd_renditions(text: str, base: str, prefer_media: bool = False) -> list[Rendition]:
    """Every AdaptationSet in an MPD, resolved to one fragment each.

    DASH states contentType/mimeType outright, so the manifest's own claim is
    unambiguous here in a way HLS CODECS never is.
    """
    import re as _re
    import urllib.parse
    import xml.etree.ElementTree as ET

    ns = "{urn:mpeg:dash:schema:mpd:2011}"
    root = ET.fromstring(text)
    base_el = root.find(ns + "BaseURL")
    if base_el is not None and base_el.text:
        base = urllib.parse.urljoin(base, base_el.text.strip())

    out: list[Rendition] = []
    for period in root.iter(ns + "Period"):
        for aset in period.iter(ns + "AdaptationSet"):
            reps = aset.findall(ns + "Representation")
            if not reps:
                continue
            rep = reps[0]
            mime = aset.get("mimeType") or rep.get("mimeType") or ""
            ctype = aset.get("contentType") or mime.split("/")[0]
            codecs = aset.get("codecs") or rep.get("codecs") or ""
            lang = aset.get("lang") or ""
            # An ElementTree Element with no children is falsy, so `a or b`
            # silently discards a valid childless SegmentTemplate. Always
            # compare against None.
            tmpl = aset.find(ns + "SegmentTemplate")
            if tmpl is None:
                tmpl = rep.find(ns + "SegmentTemplate")
            if tmpl is None:
                continue
            tpl = tmpl.get("media") if prefer_media else tmpl.get("initialization")
            if not tpl:
                tpl = tmpl.get("initialization") or tmpl.get("media")
            if not tpl:
                continue
            url = tpl.replace("$RepresentationID$", rep.get("id") or "")
            url = url.replace("$Bandwidth$", rep.get("bandwidth") or "")
            # Always attempt substitution rather than pattern-matching for
            # specific spellings first: $Number%4d$ contains neither "$Number$"
            # nor "%0", and a narrower guard silently dropped the rendition.
            if "$" in url:
                start = tmpl.get("startNumber") or "1"
                first_t = "0"
                timeline = tmpl.find(ns + "SegmentTimeline")
                if timeline is not None:
                    seg = timeline.find(ns + "S")
                    if seg is not None:
                        first_t = seg.get("t") or "0"

                def _sub(match):
                    var, fmt = match.group(1), match.group(2)
                    val = start if var == "Number" else first_t
                    # Keep the zero-pad flag: $Number%04d$ must render "0001",
                    # not "   1". A bare $Number%4d$ is legal printf but means
                    # space padding, which is never what a URL wants, so it is
                    # normalised to zero padding too.
                    if not fmt:
                        return val
                    return ("%" + (fmt if fmt.startswith("0") else "0" + fmt)) % int(val)

                url = _re.sub(r"\$(Number|Time)(?:%(0?\d+d))?\$", _sub, url)
            if "$" in url:
                continue
            label = f"{ctype or 'stream'}" + (f" ({lang})" if lang else "")
            claim = {"video": "video", "audio": "audio",
                     "text": "subtitles"}.get(ctype, "")
            if not claim:
                # DASH carries subtitle tracks as application/mp4; the codec is
                # what distinguishes them from any other application payload.
                low = codecs.lower()
                if any(c in low for c in ("wvtt", "stpp", "ttml", "tx3g")):
                    claim = "subtitles"
                else:
                    claim = _claim_from_codecs(codecs) if codecs else (ctype or "?")
            out.append(Rendition(label, urllib.parse.urljoin(base, url), claim, codecs))
    return out


def probe_url(url: str, *, nbytes: int = 65536, timeout: float = 15.0,
              prefer_media: bool = False, insecure: bool = False,
              limit: int | None = None) -> list[tuple[Rendition, Report | None, str]]:
    """Classify every rendition reachable from a stream URL.

    Returns a list of (rendition, report, error). `report` is None when that
    rendition could not be fetched, in which case `error` says why. A URL that
    is already a fragment comes back as a single entry.
    """
    # Manifests are fetched whole. Applying the per-fragment byte limit to them
    # truncates a playlist mid-URL, which yields a malformed segment URL and an
    # HTTP 400 that looks like a dead stream rather than our own doing.
    body, final = _http_get(url, None, timeout, insecure)
    top = sniff(body)

    if not top.is_playlist:
        return [(Rendition("(direct fragment)", final, ""), top, "")]

    text = body.decode("utf-8", "replace")
    if top.container == "dash-mpd":
        rends = mpd_renditions(text, final, prefer_media)
    else:
        rends = hls_master_renditions(text, final)
        if not rends:
            # A media playlist handed to us directly rather than a master.
            target, note = hls_media_target(text, final, prefer_media)
            rends = [Rendition("(media playlist)", target, "", note)] if target else []

    if limit:
        rends = rends[:limit]

    results: list[tuple[Rendition, Report | None, str]] = []
    for rend in rends:
        try:
            # Same again: the rendition URL may itself be a media playlist, so
            # fetch it whole and only cap the fragment it points at.
            blob, where = _http_get(rend.url, None, timeout, insecure)
            child = sniff(blob)
            if child.is_playlist:
                target, note = hls_media_target(
                    blob.decode("utf-8", "replace"), where, prefer_media)
                if not target:
                    results.append((rend, None, "playlist with no fetchable fragment"))
                    continue
                rend.note = rend.note or note
                blob, where = _http_get(target, nbytes, timeout, insecure)
                child = sniff(blob)
            rend.url = where
            results.append((rend, child, ""))
        except Exception as exc:                            # noqa: BLE001
            results.append((rend, None, f"{type(exc).__name__}: {exc}"))
    return results


def _short_verdict(rep: Report) -> str:
    return rep.verdict.split("<-")[0].strip()


def _normalized_kind(rep: Report) -> str:
    """Reduce a verdict to the manifest's own vocabulary, for comparison."""
    v = _short_verdict(rep).split("[")[0].strip()
    for prefix, label in (("muxed", "muxed"), ("video-only", "video"),
                          ("audio-only", "audio"), ("subtitles", "subtitles")):
        if v.startswith(prefix):
            return label
    return ""


def format_probe(url: str, results: list[tuple[Rendition, Report | None, str]]) -> str:
    """Render probe_url output as a table, manifest claim beside measured reality."""
    lines = [url, f"  {len(results)} rendition(s)", ""]
    wl = max([len(r.label) for r, _, _ in results] + [9])
    wd = max([len(r.declared) for r, _, _ in results] + [8])
    lines.append(f"  {'rendition'.ljust(wl)}  {'declared'.ljust(wd)}  carries")
    lines.append(f"  {'-' * wl}  {'-' * wd}  {'-' * 34}")
    mismatches = []
    for rend, rep, err in results:
        if rep is None:
            carries = f"[unreachable: {err}]"
        else:
            carries = _short_verdict(rep)
            if rep.confidence != "high":
                carries += f"  ({rep.confidence} confidence)"
            claimed = rend.declared
            got = _normalized_kind(rep)
            if claimed in ("video", "audio", "muxed", "subtitles") and got and claimed != got:
                carries += "   <-- manifest said " + claimed
                mismatches.append((rend, "manifest",
                                   f"claims {claimed}, bytes carry {got}"))
            if "<-" in rep.verdict:
                mismatches.append((rend, "container",
                                   rep.verdict.split("<-", 1)[1].strip()))
        lines.append(f"  {rend.label.ljust(wl)}  {rend.declared.ljust(wd)}  {carries}")
    if mismatches:
        lines.append("")
        lines.append("  disagreements:")
        for rend, source, detail in mismatches:
            where = "manifest vs bytes" if source == "manifest" else "container index vs bytes"
            lines.append(f"    {rend.label} [{where}]: {detail}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        prog="mediasniff",
        description="Classify media fragments by their content, not their filename.",
        epilog="Give it local fragments, or a stream URL (HLS master, DASH MPD, "
               "media playlist, or a fragment) to have every rendition resolved "
               "and classified.")
    ap.add_argument("target", nargs="+", help="file path or http(s) URL")
    ap.add_argument("-n", "--bytes", type=int, default=65536, metavar="N",
                    help="bytes to read per fragment (default 65536; 4096 is "
                         "enough for every format here)")
    ap.add_argument("--media", action="store_true",
                    help="target media fragments rather than init segments, "
                         "exercising the harder init-less path")
    ap.add_argument("--timeout", type=float, default=15.0, metavar="S")
    ap.add_argument("--limit", type=int, default=None, metavar="N",
                    help="classify at most N renditions per URL")
    ap.add_argument("--insecure", action="store_true",
                    help="skip TLS verification (many stream origins have "
                         "broken certificates)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="full per-track detail and evidence for each rendition")
    args = ap.parse_args()

    failed = False
    for target in args.target:
        if target.startswith(("http://", "https://")):
            try:
                results = probe_url(target, nbytes=args.bytes, timeout=args.timeout,
                                    prefer_media=args.media, insecure=args.insecure,
                                    limit=args.limit)
            except Exception as exc:                        # noqa: BLE001
                print(f"{target}\n  could not fetch: {type(exc).__name__}: {exc}\n")
                failed = True
                continue
            if not results:
                print(f"{target}\n  no renditions found\n")
                failed = True
                continue
            if args.verbose:
                print(target)
                for rend, rep, err in results:
                    print()
                    if rep is None:
                        print(f"  {rend.label}: unreachable -- {err}")
                    else:
                        print(format_report(f"  {rend.label}  [{rend.url}]", rep))
            else:
                print(format_probe(target, results))
            print()
            continue

        try:
            with open(target, "rb") as fh:
                blob = fh.read(max(args.bytes, 4 << 20))
        except OSError as exc:
            print(f"{target}\n  {exc.strerror}\n")
            failed = True
            continue
        print(format_report(target, sniff(blob)))
        print()

    sys.exit(1 if failed else 0)
