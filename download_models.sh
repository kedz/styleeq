#!/usr/bin/env bash

GDURL="https://docs.google.com/uc?export=download"

downloadfile () {
    gid=$1
    fn=$2
    CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt \
            --keep-session-cookies --no-check-certificate \
            "$GDURL&id=$gid" -O- \
            | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')

    wget --load-cookies /tmp/cookies.txt \
        "${GDURL}&confirm=${CONFIRM}&id=$gid" \
        -O $fn && rm -rf /tmp/cookies.txt

    tar zxvf $fn

}

# Models 
#https://drive.google.com/file/d/1av8YTsMd5KXlOjp5TsHtSUmHIWv4G6eK/view?usp=sharing
MODELS_ID=1av8YTsMd5KXlOjp5TsHtSUmHIWv4G6eK
downloadfile $MODELS_ID style_models.tar.gz
