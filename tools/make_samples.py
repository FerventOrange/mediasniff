"""Derive the corpus entries that no public server hands out directly.

packed_audio.aac  -- a real HLS packed-audio segment, built by demuxing the
                     AAC elementary stream back out of ts_muxed.ts and
                     prefixing the ID3 PRIV timestamp tag the HLS spec
                     requires. The audio bytes are genuine, not synthetic.
enc_aes128.ts     -- ts_muxed.ts under AES-128-CBC, i.e. what a segment
                     looks like under EXT-X-KEY:METHOD=AES-128.
webm_*.webm       -- hand-built minimal WebM init segments. Synthetic, but
                     structurally valid EBML; no public DASH-WebM vector was
                     reachable to use instead.
"""

# pylint: disable=missing-function-docstring

import os
import struct
import subprocess

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")


def packed_audio() -> None:
    src = os.path.join(HERE, "ts_muxed.ts")
    if not os.path.exists(src):
        print("  skip packed_audio (ts_muxed.ts missing)")
        return
    with open(src, "rb") as fh:
        d = fh.read()
    es = bytearray()
    for off in range(0, len(d) - 188 + 1, 188):
        p = d[off : off + 188]
        if p[0] != 0x47:
            continue
        if (((p[1] & 0x1F) << 8) | p[2]) != 0x22:  # the AAC PID
            continue
        afc = (p[3] >> 4) & 3
        i = 4 + (1 + p[4] if afc in (2, 3) else 0)
        if afc == 2 or i >= 188:
            continue
        if p[1] & 0x40:  # PES header present
            if p[i : i + 3] != b"\x00\x00\x01":
                continue
            i += 9 + p[i + 8]
        es += p[i:]

    # ID3v2.4 tag holding the PRIV frame HLS mandates on packed audio.
    body = b"com.apple.streaming.transportStreamTimestamp\x00" + struct.pack(">Q", 900000)
    priv = b"PRIV" + struct.pack(">I", len(body)) + b"\x00\x00" + body
    n = len(priv)
    syncsafe = bytes([(n >> 21) & 0x7F, (n >> 14) & 0x7F, (n >> 7) & 0x7F, n & 0x7F])
    with open(os.path.join(HERE, "packed_audio.aac"), "wb") as fh:
        fh.write(b"ID3\x04\x00\x00" + syncsafe + priv + bytes(es))
    with open(os.path.join(HERE, "packed_audio_noid3.aac"), "wb") as fh:
        fh.write(bytes(es))
    print("  packed_audio.aac / packed_audio_noid3.aac")


def aes128() -> None:
    src = os.path.join(HERE, "ts_muxed.ts")
    if not os.path.exists(src):
        print("  skip enc_aes128.ts (ts_muxed.ts missing)")
        return
    key = os.urandom(16).hex()
    subprocess.run(
        [
            "openssl",
            "enc",
            "-aes-128-cbc",
            "-K",
            key,
            "-iv",
            "00" * 16,
            "-in",
            src,
            "-out",
            os.path.join(HERE, "enc_aes128.ts"),
        ],
        check=True,
    )
    print("  enc_aes128.ts")


def webm() -> None:
    def vint(n: int) -> bytes:
        for length in range(1, 9):
            if n < (1 << (7 * length)) - 1:
                return ((1 << (7 * length)) | n).to_bytes(length, "big")
        raise ValueError(n)

    def el(eid: str, payload: bytes) -> bytes:
        return bytes.fromhex(eid) + vint(len(payload)) + payload

    def u(v: int) -> bytes:
        return v.to_bytes(max(1, (v.bit_length() + 7) // 8), "big")

    header = el(
        "1A45DFA3",
        el("4286", u(1))
        + el("42F7", u(1))
        + el("42F2", u(4))
        + el("42F3", u(8))
        + el("4282", b"webm")
        + el("4287", u(2))
        + el("4285", u(2)),
    )
    info = el("1549A966", el("2AD7B1", u(1000000)) + el("4D80", b"mediasniff"))

    def trk(num: int, ttype: int, codec: str) -> bytes:
        return el(
            "AE",
            el("D7", u(num)) + el("73C5", u(num)) + el("83", u(ttype)) + el("86", codec.encode()),
        )

    variants = {
        "webm_muxed": trk(1, 1, "V_VP9") + trk(2, 2, "A_OPUS"),
        "webm_video": trk(1, 1, "V_VP9"),
        "webm_audio": trk(1, 2, "A_OPUS"),
    }
    for name, entries in variants.items():
        blob = header + el("18538067", info + el("1654AE6B", entries))
        with open(os.path.join(HERE, name + ".webm"), "wb") as fh:
            fh.write(blob)
        print(f"  {name}.webm")


def llhls_and_elementary():
    """An LL-HLS partial segment that is not the first part of its segment opens
    in the middle of an mdat, with no moof and no magic of any kind. Derive that
    shape from real segments, plus an Annex-B elementary stream."""
    for src, out, limit in (
        ("apple_vid_seg.mp4", "llhls_part_video.m4s", 120000),
        ("apple_ec3_seg.mp4", "llhls_part_audio.m4s", 60000),
    ):
        path = os.path.join(HERE, src)
        if not os.path.exists(path):
            print(f"  skip {out} ({src} missing)")
            continue
        with open(path, "rb") as fh:
            blob = fh.read()
        start = blob.index(b"mdat") + 4
        with open(os.path.join(HERE, out), "wb") as fh:
            fh.write(blob[start : start + limit])
        print(f"  {out}")

    path = os.path.join(HERE, "apple_vid_seg.mp4")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            blob = fh.read()
        buf = blob[blob.index(b"mdat") + 4 :][:200000]
        out, p = bytearray(), 0
        while p + 4 <= len(buf):
            n = struct.unpack_from(">I", buf, p)[0]
            if n == 0 or p + 4 + n > len(buf):
                break
            out += b"\x00\x00\x00\x01" + buf[p + 4 : p + 4 + n]
            p += 4 + n
        with open(os.path.join(HERE, "annexb.h264"), "wb") as fh:
            fh.write(bytes(out))
        print("  annexb.h264")


def flv():
    """FLV as produced by RTMP ingest, in all three shapes."""

    def build(has_v, has_a, tags):
        flags = (1 if has_v else 0) | (4 if has_a else 0)
        d = bytearray(b"FLV\x01" + bytes([flags]) + struct.pack(">I", 9) + struct.pack(">I", 0))
        for ttype, payload in tags:
            d += bytes([ttype]) + len(payload).to_bytes(3, "big") + b"\x00" * 7
            d += payload + struct.pack(">I", 11 + len(payload))
        return bytes(d)

    vtag = b"\x17\x01\x00\x00\x00" + b"\x00" * 40  # H.264 keyframe
    atag = b"\xaf\x01" + b"\x21\x11\x45" * 8  # AAC raw
    for name, hv, ha, tags in (
        ("rtmp_muxed.flv", True, True, [(9, vtag), (8, atag)] * 12),
        ("rtmp_video.flv", True, False, [(9, vtag)] * 20),
        ("rtmp_audio.flv", False, True, [(8, atag)] * 20),
    ):
        with open(os.path.join(HERE, name), "wb") as fh:
            fh.write(build(hv, ha, tags))
        print(f"  {name}")


if __name__ == "__main__":
    os.makedirs(HERE, exist_ok=True)
    packed_audio()
    aes128()
    webm()
    llhls_and_elementary()
    flv()
