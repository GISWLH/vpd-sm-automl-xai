#!/bin/bash
# Resume PKU NDVI consolidated zips through this machine: curl from Zenodo, append over ssh to xuanze, md5 check there.
R=/data/lhwang/vpd_sm_repro/raw/pku_ndvi
one() {
  k=$1; m=$2; u=$3
  for try in 1 2 3 4 5 6 7 8 9 10; do
    n=$(ssh -o BatchMode=yes xuanze-neiwang "stat -c %s $R/$k" 2>/dev/null)
    curl -sSL --max-time 7200 -r "$n-" "$u" | ssh -o BatchMode=yes xuanze-neiwang "cat >> $R/$k" 2>/dev/null
    s=$(ssh -o BatchMode=yes xuanze-neiwang "md5sum $R/$k | cut -d' ' -f1" 2>/dev/null)
    [ "$s" = "$m" ] && { echo "OK $k"; return 0; }
    echo "retry $try $k size $(ssh -o BatchMode=yes xuanze-neiwang "stat -c %s $R/$k" 2>/dev/null)"
    sleep 15
  done
  echo "BAD $k"
}
ssh -o BatchMode=yes xuanze-neiwang "grep zip $R/urls2.txt" 2>/dev/null > /tmp/urls2.txt
while read k m u; do one "$k" "$m" "$u" & done < /tmp/urls2.txt
wait
echo DONE
