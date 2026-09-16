"""Regression suite for mediasniff.

Fetch the corpus first:  ./tools/fetch_samples.sh

Every expectation below is the source manifest's own CODECS / contentType
declaration. The manifest is the answer key and is never an input to sniff().
"""

import os
import struct

import pytest

import mediasniff as ms

SAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def _fixture(name):
    """Checked-in captures from live streams that cannot be re-fetched.
    See tests/fixtures/README.md for provenance."""
    with open(os.path.join(FIXTURES, name), "rb") as fh:
        return fh.read()

# (file, expected verdict prefix, expected container)
CASES = [
    # --- MPEG-TS -----------------------------------------------------------
    # Two segments from one packager, 60536 bytes each, names differing only by
    # an infix. This is the case the project exists to solve.
    ("ts_muxed.ts", "muxed (audio+video)", "mpeg-ts"),
    ("ts_audioonly.ts", "audio-only", "mpeg-ts"),
    # Live webcam whose PMT declares an ID3 metadata PID carrying nothing.
    ("live_summit.ts", "video-only", "mpeg-ts"),
    # --- fMP4 init segments (moov present, so hdlr is authoritative) --------
    ("init_video.dash", "video-only", "isobmff"),
    ("init_audio.dash", "audio-only", "isobmff"),
    ("apple_vid_init.mp4", "video-only", "isobmff"),
    ("apple_ec3_init.mp4", "audio-only", "isobmff"),
    # --- fMP4 media segments (moof only, nothing declares the track kind) ---
    ("seg_video.dash", "video-only", "isobmff"),
    ("seg_audio.dash", "audio-only", "isobmff"),
    ("apple_vid_seg.mp4", "video-only", "isobmff"),
    ("apple_ec3_seg.mp4", "audio-only", "isobmff"),
    ("apple_ac3_seg.mp4", "audio-only", "isobmff"),
    # All-intra trickplay: sample table reads like VBR audio; only the mdat
    # payload proves it is video.
    ("apple_iframe_seg.mp4", "video-only", "isobmff"),
    # --- packed audio, subtitles, manifests, opaque ------------------------
    ("packed_audio.aac", "audio-only", "adts"),
    ("packed_audio_noid3.aac", "audio-only", "adts"),
    ("apple_vtt_seg.mp4", "subtitles/text-only", "webvtt"),
    ("enc_aes128.ts", "opaque high-entropy", "opaque"),
    ("master.m3u8", "playlist", "m3u8-playlist"),
    ("tos.mpd", "manifest", "dash-mpd"),
    ("webm_muxed.webm", "muxed (audio+video)", "webm/matroska"),
    ("webm_video.webm", "video-only", "webm/matroska"),
    ("webm_audio.webm", "audio-only", "webm/matroska"),
    # --- DRM: CENC leaves the container and sample table in the clear -------
    ("drm_avc_init.mp4", "video-only", "isobmff"),
    ("drm_hevc_init.mp4", "video-only", "isobmff"),
    ("drm_aac_init.mp4", "audio-only", "isobmff"),
    ("drm_avc_seg.m4s", "video-only", "isobmff"),
    ("drm_hevc_seg.m4s", "video-only", "isobmff"),
    ("drm_aac_seg.m4s", "audio-only", "isobmff"),
    ("drm_wvtt_init.mp4", "subtitles/text-only", "isobmff"),
    ("drm_stpp_seg.m4s", "subtitles/text-only", "isobmff"),
    ("drm_webm_audio.webm", "audio-only", "webm/matroska"),
    ("drm_webm_video.webm", "video-only", "webm/matroska"),
    # --- live-specific shapes ----------------------------------------------
    ("rtmp_muxed.flv", "muxed (audio+video)", "flv"),
    ("rtmp_video.flv", "video-only", "flv"),
    ("rtmp_audio.flv", "audio-only", "flv"),
    ("llhls_part_video.m4s", "video-only", "headless media payload"),
    ("annexb.h264", "video-only", "annex-b elementary stream"),
    ("drm_wvtt_seg.m4s", "subtitles/text-only", "isobmff"),
]

# A fragment must be classifiable from a short Range request: moof precedes
# mdat, and HLS requires a PAT+PMT at the head of every TS segment.
PREFIXES = (4096, 16384, 65536, 262144)


def _load(name):
    path = os.path.join(SAMPLES, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} missing -- run ./tools/fetch_samples.sh")
    with open(path, "rb") as fh:
        return fh.read()


@pytest.mark.parametrize("name,verdict,container", CASES)
def test_classification(name, verdict, container):
    rep = ms.sniff(_load(name))
    assert rep.container == container
    assert rep.verdict.startswith(verdict)


@pytest.mark.parametrize("name,verdict,container", [c for c in CASES if "m3u8" not in c[0]])
def test_classification_from_short_prefix(name, verdict, container):
    """The verdict must not depend on having the whole fragment."""
    blob = _load(name)
    for n in PREFIXES:
        if n >= len(blob):
            break
        assert ms.sniff(blob[:n]).verdict.startswith(verdict), f"{name} wrong at {n}B"


def test_declared_and_observed_are_tracked_separately():
    """The webcam's PMT declares an ID3 PID that carries no packets."""
    rep = ms.sniff(_load("live_summit.ts"))
    ghost = [t for t in rep.tracks if t.declared and not t.observed]
    assert ghost, "expected a declared-but-absent track"
    assert all("ABSENT" in t.note for t in ghost)


def test_encrypted_segment_is_not_guessed():
    """Full-segment AES-128 must report opaque, never a fabricated track list --
    and must present itself as an inference, because entropy alone cannot tell
    ciphertext from any other compressed payload."""
    rep = ms.sniff(_load("enc_aes128.ts"))
    assert rep.encrypted == "aes-128-full"
    assert rep.confidence == "low"
    assert "inferred" in rep.drm.scheme
    assert not rep.tracks


def test_continuous_stream_joined_midframe_is_not_called_encrypted():
    """An Icecast MP3 read joins mid-frame, so offset 0 is meaningless and the
    first syncword can be thousands of bytes in. Without resync these landed on
    the entropy heuristic and were reported as AES-128 encrypted -- 320 kbps MP3
    measures ~7.99 bits/byte, indistinguishable from ciphertext by entropy."""
    body = _load("packed_audio_noid3.aac")
    for lead in (157, 402, 3857):
        blob = bytes(((i * 97 + 13) % 256) for i in range(lead)) + body
        rep = ms.sniff(blob)
        assert rep.verdict.startswith("audio-only"), f"lead {lead}: {rep.verdict}"
        assert rep.encrypted == "", f"lead {lead} wrongly claimed encryption"


def test_icy_metadata_interleaved_audio():
    """A SHOUTcast client that requests metadata gets StreamTitle blocks injected
    *into* the audio every icy-metaint bytes -- often ~1 KB, leaving room for only
    two frames between them. No consecutive-run check can survive that, so the
    detector measures frame coverage with resync instead. This capture was being
    reported as AES-128 encrypted."""
    blob = _fixture("icy_interleaved_aac.bin")
    assert b"StreamTitle" in blob, "fixture should contain injected ICY metadata"
    cover, frames, _first = ms._frame_coverage(blob, ms._adts_frame_len, ms._adts_key)
    assert cover > 0.9 and frames > 50
    rep = ms.sniff(blob)
    assert rep.verdict.startswith("audio-only")
    assert rep.encrypted == ""


def test_coverage_threshold_sits_in_a_real_gap():
    """Guards the margin itself. Audio that reaches the coverage path scores
    97-100%; the worst non-audio case (Annex-B H.264 misread as MPEG audio)
    scores 40%. If a change erodes either side, this fails before the wild
    sweep has to find it."""
    def best(blob):
        return max(ms._frame_coverage(blob, fn, kf, sy)[0] for fn, kf, sy in (
            (ms._adts_frame_len, ms._adts_key, b"\xff"),
            (ms._mpeg_frame_len, ms._mpeg_key, b"\xff"),
            (ms._ac3_frame_len, ms._ac3_key, b"\x0b")))

    for name in ("packed_audio_noid3.aac", "apple_ec3_seg.mp4"):
        assert best(_load(name)) > 0.90, name
    assert best(_fixture("icy_interleaved_aac.bin")) > 0.90
    assert best(_fixture("id3_padded_aac.bin")) > 0.90
    # non-audio must stay far below
    assert best(_load("annexb.h264")) < 0.55
    assert best(_load("enc_aes128.ts")) < 0.55


def test_annexb_needs_more_than_one_start_code():
    """Digital silence in MP3 is runs of 0x00, so 00 00 01 appears at the head
    of audio often enough to matter. One start code classified a live MP3
    station as video."""
    audio = _load("packed_audio_noid3.aac")
    rep = ms.sniff(b"\x00\x00\x01\x65" + audio)
    assert rep.verdict.startswith("audio-only"), rep.verdict
    # a genuine Annex-B stream still resolves
    assert ms.sniff(_load("annexb.h264")).container == "annex-b elementary stream"


def test_frame_coverage_rejects_random_data():
    """The counterpart: coverage must stay far below threshold on noise, or the
    fix would just trade one false positive for another."""
    for _ in range(10):
        noise = os.urandom(65536)
        for fn, kf in ((ms._adts_frame_len, ms._adts_key),
                       (ms._mpeg_frame_len, ms._mpeg_key)):
            cover, _n, _o = ms._frame_coverage(noise, fn, kf)
            # Observed max over 30 trials is ~12%; the detector fires at 55%.
            assert cover < 0.35, f"noise reached {cover:.0%} coverage"
    # Real ciphertext must also stay far below threshold.
    cover, _n, _o = ms._frame_coverage(_load("enc_aes128.ts"), ms._adts_frame_len, ms._adts_key)
    assert cover < 0.35


def test_ac3_needs_frame_validation_not_just_a_syncword():
    """0x0B77 turns up in any binary every ~64 KB. Accepting it on sight made an
    obfuscated video stream report as AC-3 audio, so frame sizes are computed
    from the ATSC A/52 table and chained like every other codec."""
    assert ms._ac3_frame_len(bytes.fromhex("0b7700000c1008"), 0) == 384
    assert ms._ac3_frame_len(bytes.fromhex("0b77000003ff08"), 0) is None   # frmsizecod > 37
    assert ms._ac3_frame_len(bytes.fromhex("0b76000000000000"), 0) is None  # wrong syncword
    for _ in range(10):
        cover, _n, _o = ms._frame_coverage(os.urandom(65536), ms._ac3_frame_len,
                                           ms._ac3_key, b"\x0b")
        assert cover < 0.35


def test_id3_tag_length_is_not_trusted_blindly():
    """A real packager's ID3 size left an 880-byte gap before the first ADTS
    frame. Trusting the tag length alone gave 'packed audio, unknown codec' at
    low confidence; resyncing past it identifies the codec properly."""
    rep = ms.sniff(_fixture("id3_padded_aac.bin"))
    assert rep.verdict.startswith("audio-only")
    assert rep.confidence == "high"
    assert "aac" in rep.tracks[0].codec


def test_audio_coverage_outranks_a_loose_nal_chain():
    """A length-prefixed NAL "chain" is four arbitrary bytes read as a length
    plus one read as a header -- compressed audio satisfies it by chance often
    enough that a live MP3 station was classified as video. Strong evidence
    (a validated frame run over most of the buffer) must be checked first."""
    audio = _load("packed_audio_noid3.aac")
    # A prefix that reads as a plausible length-prefixed NAL: length 200, then
    # a byte whose low 5 bits are a valid AVC nal_unit_type.
    decoy = struct.pack(">I", 200) + b"\x65" + b"\x00" * 199
    rep = ms.sniff(decoy + audio)
    assert rep.verdict.startswith("audio-only"), rep.verdict


def test_mpeg_audio_frame_lengths():
    """Frame length must be computed, not guessed: resync depends on the next
    syncword landing exactly where the length says it will."""
    assert ms._mpeg_audio_frame(bytes.fromhex("fffb9064"))[0] == 417   # MPEG1 L3 128k/44.1k
    assert ms._mpeg_audio_frame(bytes.fromhex("fff35064"))[0] == 130   # MPEG2 L3 40k/22.05k
    assert ms._mpeg_audio_frame(bytes.fromhex("ffe00000")) is None     # reserved version
    assert ms._mpeg_audio_frame(bytes.fromhex("fffbf064")) is None     # bad bitrate index


def test_moof_only_audio_has_no_syncword_to_find():
    """CMAF stores raw AAC access units, so the sample table is the only signal.
    Guards the regression that made payload sniffing look sufficient."""
    blob = _load("seg_audio.dash")
    mdat = blob[blob.index(b"mdat") + 4 :]
    assert not ms._adts_chain(mdat[:8192])
    assert ms.sniff(blob).verdict.startswith("audio-only")


def test_all_intra_video_is_not_mistaken_for_audio():
    """An I-frame-only trickplay track has uniform sample durations and no
    keyframe peak -- statistically indistinguishable from VBR audio. The
    chain-validated NAL run in the mdat is what settles it, and direct
    observation must outrank the sample-table inference."""
    blob = _load("apple_iframe_seg.mp4")
    trafs = ms._scan_trafs(blob)
    assert len(trafs) == 1
    kind, _codec, _why, score = ms._classify_traf(trafs[0])
    assert kind is ms.Kind.UNKNOWN, "sample table alone must not claim this one"
    assert -2.0 < score < 2.0
    assert ms.sniff(blob).verdict.startswith("video-only")


def test_cenc_does_not_block_classification():
    """Common Encryption keeps boxes, handlers and the sample table in the
    clear so a player can parse without a key -- so can we."""
    init = ms.sniff(_load("drm_avc_init.mp4"))
    assert init.encrypted == "cenc"
    assert init.drm.scheme == "cenc"
    assert init.drm.key_ids, "default_KID is readable from tenc"
    assert init.verdict.startswith("video-only")
    assert init.confidence == "high"

    seg = ms.sniff(_load("drm_avc_seg.m4s"))
    assert seg.drm.per_sample, "senc/saiz mark per-sample encryption"
    assert seg.verdict.startswith("video-only")


def test_encrypted_webm_reports_key_id():
    rep = ms.sniff(_load("drm_webm_video.webm"))
    assert rep.drm.key_ids
    assert "AES" in rep.drm.scheme
    assert rep.verdict.startswith("video-only")


def test_subtitles_in_a_drm_stream_are_left_clear():
    """Packagers routinely protect a/v and leave text tracks in the clear."""
    rep = ms.sniff(_load("drm_wvtt_init.mp4"))
    assert not rep.drm.scheme
    assert rep.verdict.startswith("subtitles")


def test_cmaf_fragment_carrying_ttml_is_not_reported_as_raw_ttml():
    """Container magic must outrank content heuristics: this is an ISOBMFF
    fragment with a text track, not a bare TTML file."""
    rep = ms.sniff(_load("drm_stpp_seg.m4s"))
    assert rep.container == "isobmff"
    assert rep.verdict.startswith("subtitles")


def test_headless_partial_segment_is_classified():
    """An LL-HLS part that is not the first of its segment has no moof and no
    magic -- it opens mid-mdat."""
    blob = _load("llhls_part_video.m4s")
    assert blob[:8] != b"\x00\x00\x00\x18"
    rep = ms.sniff(blob)
    assert rep.verdict.startswith("video-only")
    assert rep.confidence == "medium"


def test_empty_webvtt_cue_is_not_read_as_a_nal_unit():
    """An 8-byte `vtte` box satisfies a naive length-prefixed-NAL check, which
    made a 180-byte subtitle segment report as video. Sample-box detection runs
    first, and a lone tiny NAL is no longer sufficient evidence on its own."""
    assert not ms._nal_chain(b"\x00\x00\x00\x08vtte")
    assert ms._vtt_sample_chain(b"\x00\x00\x00\x08vtte")
    rep = ms.sniff(_load("drm_wvtt_seg.m4s"))
    assert rep.verdict.startswith("subtitles")


def test_nal_codec_guess_admits_ambiguity():
    """One NAL header byte cannot separate H.264 from HEVC. Parameter-set units
    can; an SEI-only run cannot, and must not be guessed at."""
    assert ms._guess_nal_codec([0x06, 0x67, 0x68, 0x65]) == "h264"
    assert ms._guess_nal_codec([0x4E, 0x40, 0x42, 0x44]) == "hevc"
    assert ms._guess_nal_codec([0x06, 0x06]) == "h264/hevc"
    assert ms._guess_nal_codec([0x41, 0x41]) == "h264/hevc"


def test_flv_header_and_tags_are_cross_checked():
    rep = ms.sniff(_load("rtmp_muxed.flv"))
    assert {t.kind for t in rep.tracks} == {ms.Kind.VIDEO, ms.Kind.AUDIO}
    assert all(t.declared and t.observed for t in rep.tracks)


def test_lone_sync_byte_is_not_mpeg_ts():
    """0x47 occurs in any binary roughly once per 256 bytes. A single one, with
    no second packet to corroborate it, previously locked the TS parser on --
    real gzip-compressed playlists were being reported as MPEG-TS."""
    # Deterministic on purpose: random filler can itself contain a plausible
    # packet header and make this test flaky.
    filler = bytes((i * 37 + 11) % 256 for i in range(360))
    filler = filler.replace(b"\x47", b"\x46")          # exactly one 0x47, placed below
    blob = filler[:283] + b"\x47" + filler[284:]
    assert len([b for b in blob if b == 0x47]) == 1
    assert ms._ts_layout(blob) is None
    # header validation, not just the sync byte
    assert not ms._ts_packet_ok(b"\x47\x80\x00\x10" + b"\x00" * 184, 0)   # error flag
    assert not ms._ts_packet_ok(b"\x47\x00\x00\x00" + b"\x00" * 184, 0)   # afc == 0
    assert ms._ts_packet_ok(b"\x47\x40\x00\x10" + b"\x00" * 184, 0)


def test_gzipped_fragments_are_unwrapped():
    """Some origins serve playlists gzipped without the client decoding it."""
    import gzip as _gzip
    inner = b"#EXTM3U\n#EXT-X-VERSION:3\nindex.m3u8\n"
    rep = ms.sniff(_gzip.compress(inner))
    assert rep.is_playlist
    assert "gzip-compressed" in rep.evidence[0]


def test_prefix_does_not_disprove_a_declared_track():
    """A 7 Mbit/s live TS segment can carry 1 MB of video before its first audio
    packet. On a truncated fragment, a declared track that has not appeared yet
    must not be treated as absent."""
    blob = _load("ts_muxed.ts")
    assert not ms.sniff(blob).truncated
    prefix = blob[: 188 * 20 + 40]              # deliberately mid-packet
    rep = ms.sniff(prefix)
    assert rep.truncated
    assert rep.verdict.startswith("muxed")


# --- regressions found by tools/wild.py against real-world streams ---------

def test_real_llhls_part_without_a_keyframe_is_still_video():
    """A 0.2 s LL-HLS part holds ~5 inter frames and no keyframe, so its sample
    sizes are flat -- statistically identical to audio. Two bugs met here: the
    sample table claimed audio, and the mdat NAL check that should have
    overridden it was rejecting the 11 KB slice NALs for exceeding the sniff
    window. Real 2 Mbit/s video has NALs bigger than any sane window."""
    for name in ("llhls_part_video_nokeyframe.m4s", "llhls_part_video_interframes.m4s"):
        rep = ms.sniff(_fixture(name))
        assert rep.verdict.startswith("video-only"), f"{name}: {rep.verdict}"


def test_real_llhls_audio_part_still_classifies():
    """The low-sample-count guard must not cost us real audio parts."""
    assert ms.sniff(_fixture("llhls_part_audio.m4s")).verdict.startswith("audio-only")
    assert ms.sniff(_fixture("llhls_init_video.mp4")).verdict.startswith("video-only")


def test_nal_chain_accepts_a_nal_larger_than_the_window():
    """The regression in one line: a single 11 KB NAL seen through an 8 KB
    window is normal, not evidence against."""
    big = struct.pack(">I", 11516) + b"\x61" + b"\x00" * 200
    assert ms._nal_chain(big, probes=4)


def test_multiprogram_scrambled_multiplex():
    """A 36-program DVB multiplex under conditional access. Only the first four
    PMTs were being parsed, and scrambled PES headers defeat the stream_id
    fallback -- but the PSI stays in the clear, so the tracks are still typed."""
    rep = ms.sniff(_fixture("mpts_dvbcsa_36program.ts"))
    assert rep.encrypted == "dvb-csa"
    assert rep.verdict.startswith("muxed")
    kinds = {t.kind for t in rep.tracks}
    assert ms.Kind.VIDEO in kinds and ms.Kind.AUDIO in kinds


def test_gzipped_playlist_fixture():
    rep = ms.sniff(_fixture("gzipped_playlist.m3u8.gz"))
    assert rep.is_playlist and "gzip" in rep.evidence[0]


def test_declared_audio_that_never_arrives():
    """A complete segment whose manifest AND PMT both declare AAC, but which
    carries 350 video packets and no audio at all. On a complete fragment the
    bytes win and the disagreement is surfaced; on a prefix of the same segment
    the declaration wins, because a prefix cannot prove absence."""
    blob = _fixture("ts_audio_declared_absent.ts")
    full = ms.sniff(blob)
    assert not full.truncated
    assert full.verdict.startswith("video-only")
    assert "declares audio but carries none" in full.verdict
    ghost = [t for t in full.tracks if t.kind is ms.Kind.AUDIO]
    assert ghost and ghost[0].declared and not ghost[0].observed

    prefix = ms.sniff(blob[: len(blob) // 2 + 7])
    assert prefix.truncated
    assert prefix.verdict.startswith("muxed")
    assert "not seen in this prefix" in prefix.verdict


def test_non_media_bodies_are_named_not_shrugged_at():
    """Dead stream URLs answer with something, and naming it is more useful than
    "unknown". One origin in the wild sample literally returns a shrug."""
    assert ms.sniff(b"<!DOCTYPE html><html>").container == "html"
    assert ms.sniff(b'{"error":"not found"}').container == "json"
    assert ms.sniff("\u00af\\_(\u30c4)_/\u00af".encode()).container == "text"
    assert ms.sniff(b"Not Found").container == "text"
    # and must not swallow real media
    assert ms.sniff(_load("ts_muxed.ts")).container == "mpeg-ts"
    assert ms.sniff(_load("packed_audio.aac")).container == "adts"


def test_truncated_input_does_not_raise():
    """Byte-for-byte truncation of every sample must degrade, never crash."""
    for name, _, _ in CASES:
        blob = _load(name)
        for n in (1, 2, 7, 8, 9, 15, 64, 187, 188, 189, 1000):
            ms.sniff(blob[:n])


def test_garbage_input_does_not_raise():
    for blob in (b"", b"\x00" * 1024, b"\x47" * 1024, os.urandom(4096), b"ID3" + b"\xff" * 99):
        ms.sniff(blob)
