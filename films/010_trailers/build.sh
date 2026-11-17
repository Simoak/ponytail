#!/bin/bash
# Reproducible build: lossless segments -> join -> one final encode (grain once) + covers.
set -e; cd "$(dirname "$0")"
python3 audio.py
ls -d seg >/dev/null; if [ "$1" != encode ]; then rm -f seg/*.mkv; seq 0 8 | xargs -P 4 -I{} python3 render.py seg {}; fi
for i in $(seq -f "%02g" 0 8); do echo "file 'seg$i.mkv'"; done > seg/list.txt
python3 covers.py
ffmpeg -y -loglevel error -f concat -safe 0 -i seg/list.txt -c copy master_lossless.mkv
for A in full sfx; do
  OUT=ORRIKAN_010_TRAILERS_$( [ $A = full ] && echo FULL || echo SFX ).mp4
  VF="noise=c0s=3:c0f=t+a:all_seed=10,lutyuv=y='clip(val,16,235)'"
  X="-c:v libx264 -preset slow -tune grain -x264-params mbtree=0 -b:v 12M -maxrate 14M -bufsize 24M -g 60 -sc_threshold 0 -profile:v high -level 4.2 -pix_fmt yuv420p -color_primaries bt709 -color_trc bt709 -colorspace bt709 -color_range tv"
  ffmpeg -y -loglevel error -i master_lossless.mkv -vf "$VF" $X -pass 1 -passlogfile seg/p$A -an -f mp4 /dev/null
  ffmpeg -y -loglevel error -i master_lossless.mkv -i audio/$A.wav -vf "$VF" $X -pass 2 -passlogfile seg/p$A -c:a aac -b:a 320k -ar 48000 -ac 2 -movflags +faststart -shortest $OUT
done
