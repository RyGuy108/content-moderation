#!/usr/bin/env bash
# Fetch third-party data and model weights that are too big (or not ours) to commit.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw models
# MS-COCO 2014 val captions (CC BY 4.0): everyday scene descriptions, used as benign prompts.
[ -f data/raw/coco_captions_val2014.json ] || curl -fsSL -o data/raw/coco_captions_val2014.json \
  https://raw.githubusercontent.com/tylin/coco-caption/master/annotations/captions_val2014.json
# dlib face detector / landmarks / ResNet face embedding (public domain), via the PyPI sdist.
if [ ! -f models/dlib_face_recognition_resnet_model_v1.dat ]; then
  tmp=$(mktemp -d)
  pip download --no-deps --no-binary :all: -d "$tmp" face_recognition_models==0.3.0 -q
  tar xzf "$tmp"/face_recognition_models-0.3.0.tar.gz -C "$tmp"
  cp "$tmp"/face_recognition_models-0.3.0/face_recognition_models/models/{shape_predictor_5_face_landmarks.dat,dlib_face_recognition_resnet_model_v1.dat,mmod_human_face_detector.dat} models/
  rm -rf "$tmp"
fi
# OpenCV res10 SSD face detector (also used by the media preprocessor).
[ -f models/res10_300x300_ssd_iter_140000_fp16.caffemodel ] || curl -fsSL -o models/res10_300x300_ssd_iter_140000_fp16.caffemodel \
  https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
[ -f models/deploy.prototxt ] || curl -fsSL -o models/deploy.prototxt \
  https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt
echo ok
