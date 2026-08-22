# Real-video validation assets

## person_bottle2.jpg / hydration.jpg

- Source: Wikimedia Commons — "Hydration (23379336205).jpg"
  https://commons.wikimedia.org/wiki/File:Hydration_(23379336205).jpg
- License: freely licensed (CC BY 2.0 via Flickr upload to Commons); see the
  Commons file page for the exact license and author attribution.
- Use: base photograph for real-video validation scenarios. Real YOLO detects
  `person` (0.94) and `bottle` (0.57) in this image; MoveNet produces usable
  wrist/shoulder keypoints on the person crop.

The positive/negative/ambiguous validation videos are generated FROM this real
photograph at test time (see tests/test_validation_scenarios.py and
tests/test_real_video_demo.py) so every frame contains real, re-detected
content — no synthetic track injection anywhere.
