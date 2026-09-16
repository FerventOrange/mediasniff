#!/usr/bin/env bash
# Rebuild the test corpus from public test vectors.
#
# The point of the corpus is that filenames and extensions are useless in it:
#   *.dash    -- two init segments and two media segments, identical magic
#                and identical ftyp/styp brands, one pair audio one pair video
#   *.ts      -- a muxed segment and an audio-only segment, same byte length,
#                names differing only by an infix
#   *.mp4     -- includes a file that is actually WebVTT text
#
# Ground truth comes from each manifest's own CODECS / contentType attributes,
# which the sniffer never sees.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/samples" && cd "$ROOT/samples"

US="https://demo.unified-streaming.com/k8s/features/stable/video/tears-of-steel/tears-of-steel.ism"
AP="https://devstreaming-cdn.apple.com/videos/streaming/examples/adv_dv_atmos"

get() { curl -fsS -m 45 "$1" -o "$2" && echo "  $2"; }

echo "Unified Streaming -- tears-of-steel (TS muxed vs audio-only, CMAF a/v):"
get "$US/.mpd"                                                    tos.mpd
get "$US/tears-of-steel-audio_eng=64008-video_eng=401000-1.ts"    ts_muxed.ts
get "$US/tears-of-steel-audio_eng=64008-1.ts"                     ts_audioonly.ts
get "$US/tears-of-steel-video_eng=401000.dash"                    init_video.dash
get "$US/tears-of-steel-audio_eng=64008.dash"                     init_audio.dash
get "$US/tears-of-steel-video_eng=401000-0.dash"                  seg_video.dash
get "$US/tears-of-steel-audio_eng=64008-0.dash"                   seg_audio.dash

echo "Apple -- adv_dv_atmos (fMP4 with EC-3/AC-3 and WebVTT):"
get "$AP/main.m3u8" apple.m3u8
pull() {   # $1 = rendition playlist path, $2 = output prefix
  local d; d=$(dirname "$1")
  curl -fsS -m 30 "$AP/$1" -o /tmp/_pl.m3u8
  local init seg
  init=$(grep -oE 'URI="[^"]*"' /tmp/_pl.m3u8 | head -1 | sed 's/URI="//;s/"//' || true)
  seg=$(grep -v '^#' /tmp/_pl.m3u8 | grep -v '^$' | head -1)
  [ -n "$init" ] && get "$AP/$d/$init" "${2}_init.mp4"
  [ -n "$seg"  ] && get "$AP/$d/$seg"  "${2}_seg.mp4"
}
pull "Job932393e2-1e4f-4fdb-ab59-0d201f752656-107660254-Transcode_audio_full_en_atmos_0_1-en_audio/prog_index.m3u8" apple_ec3
pull "Job932393e2-1e4f-4fdb-ab59-0d201f752656-107660254-Transcodeaudio_en_surround51dd_audio/prog_index.m3u8"       apple_ac3
pull "Job932393e2-1e4f-4fdb-ab59-0d201f752656-107660254-Convertsubtitles_vttv2_en_any-en/prog_index.m3u8"           apple_vtt
VID=$(grep -A1 '^#EXT-X-STREAM-INF' apple.m3u8 | grep -vE '^#|^--' | head -1)
pull "$VID" apple_vid
# All-intra trickplay: the adversarial video case. Every sample is a keyframe,
# so the sample table has none of the usual video tells and reads like VBR audio.
IFR=$(grep '^#EXT-X-I-FRAME-STREAM-INF' apple.m3u8 | head -1 | grep -oE 'URI="[^"]*"' | sed 's/URI="//;s/"//')
[ -n "$IFR" ] && pull "$IFR" apple_iframe

echo "Axinom MultiDRM -- CENC under Widevine + PlayReady:"
AX="https://media.axprod.net/TestVectors/v7-MultiDRM-SingleKey"
for pair in "1:drm_avc" "8:drm_hevc" "15:drm_aac" "18:drm_wvtt" "19:drm_stpp"; do
  id="${pair%%:*}"; nm="${pair##*:}"
  get "$AX/$id/init.mp4"  "${nm}_init.mp4"
  get "$AX/$id/0001.m4s"  "${nm}_seg.m4s"
done

echo "Shaka angel-one -- encrypted WebM (VP9 + Opus):"
SH="https://storage.googleapis.com/shaka-demo-assets/angel-one-widevine"
get "$SH/a-eng-0096k-libopus-2c.webm" drm_webm_audio.webm
get "$SH/v-0144p-0100k-vp9.webm"      drm_webm_video.webm

echo "Liberty Mountain webcam -- live HLS, video-only with a dead ID3 PID:"
LM="https://live8.brownrice.com:444/libertysummit/libertysummit.stream"
get "$LM/main_playlist.m3u8" master.m3u8
CL=$(grep -v '^#' master.m3u8 | head -1)
curl -fsS -m 30 "$LM/$CL" -o /tmp/_cl.m3u8
get "$LM/$(grep -v '^#' /tmp/_cl.m3u8 | head -1)" live_summit.ts

echo "Derived samples (packed audio, AES-128, WebM):"
python3 "$ROOT/tools/make_samples.py"
echo "done."
