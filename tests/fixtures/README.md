# Fixtures

Unlike `samples/`, these are checked in: they were captured from live streams
that rotate their segments out within seconds, so `tools/fetch_samples.sh`
cannot reproduce them. Each one caused a real bug.

| File | Source | Bug it caught |
|---|---|---|
| `llhls_part_video_nokeyframe.m4s` | apa.tv LL-HLS, `EXT-X-PART` | 0.2 s part, 5 inter frames, no keyframe. `_nal_chain` rejected its 11 KB slice NALs for being larger than the sniff window, so nothing could type it. |
| `llhls_part_video_interframes.m4s` | apa.tv LL-HLS | Same shape; the sample table read as audio (cv 0.13, peak 1.2) and was confidently **misclassified as audio-only**. |
| `llhls_part_audio.m4s` | apa.tv LL-HLS | The audio counterpart, to prove the low-sample-count guard did not break real audio. |
| `llhls_init_video.mp4` | apa.tv LL-HLS | `EXT-X-MAP` init for the above. |
| `mpts_dvbcsa_36program.ts` | public broadcast multiplex | 36-program MPTS with DVB-CSA conditional access. Only the first 4 PMTs were parsed, and scrambled PES headers defeated the stream_id fallback. |
| `gzipped_playlist.m3u8.gz` | pbs.org | A gzipped playlist that a single stray `0x47` byte caused to be reported as MPEG-TS. |

| `icy_interleaved_aac.bin` | SHOUTcast AAC+ station | A client requesting `Icy-MetaData` gets `StreamTitle='...'` blocks injected *into* the audio every ~1 KB, leaving room for only two frames between them. No consecutive-frame check survives that; it was reported as AES-128 encrypted. |
| `id3_padded_aac.bin` | rds.radio HLS `.aac` | The ID3 tag length does not reach the first ADTS frame -- an 880-byte gap follows it. Trusting the tag length gave "unknown codec" at low confidence. |

| `ts_audio_declared_absent.ts` | mycloudstream.io live TV | Both the HLS manifest (`CODECS="avc1.64001f,mp4a.40.2"`) and the PMT declare an AAC stream, but the segment carries 350 video packets and **zero** audio packets. The case the declared-vs-observed split exists for. |

Captured 2026-09-16. All from publicly listed, freely-available streams.

**Licensing:** these captures are fragments of third-party broadcast and radio
streams, retained as test vectors only. They are not covered by the project's
MIT license and are not redistributable as media. Each is a few seconds long,
and several are unplayable in isolation (no init segment, or CSA-scrambled).
