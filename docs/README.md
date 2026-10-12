# YOCO project website

Static project page for **YOCO: You Only Calibrate Once! Fast Mocap Calibration for Dexterous Teleoperation**, CoRL 2026.

## Preview

From the repository root:

```bash
python3 -m http.server 8000 --directory docs --bind 127.0.0.1
```

Open http://127.0.0.1:8000/. No package installation or build step is required.

## GitHub Pages

Publish this directory from the `release/yoco-website` branch:

1. Push the branch to `origin`.
2. In the repository's **Settings → Pages**, select **Deploy from a branch**.
3. Select **release/yoco-website** and **/docs**, then save.

The standard project URL is https://franklin-zhang0.github.io/Calibration_net/. The `.nojekyll` file keeps the site static. All CSS, JavaScript, figures, PDFs, and videos are local, and asset paths are relative so the page also works under the repository URL prefix.

## Content and sources

- Paper metadata, author affiliations, abstract, Tables 1–3, and task protocol: [arXiv:2610.11657v1](https://arxiv.org/abs/2610.11657v1), submitted 8 October 2026. A copy of the supplied public PDF is included in `assets/papers/yoco.pdf`.
- Overview and method figures: the supplied paper figures. Calibration-pose photos and the teleoperation summary chart: supplied presentation assets, checked against the public paper.
- Overview video: supplied `corl_supp_video.mov`.
- Task videos: supplied `71_1780338891_raw_X0mmMg_mosaic.mp4` (bottle cap), `IMG_8322_F34462CB-1C42-4FCD-A270-CD6D1AB90AD6.mp4` (earbud), `70_1780338841_raw_CbU2Mk_mosaic.mp4` (glue), and `68_1780338784_raw_MjQ87H_mosaic.mp4` (tissue box).
- Videos are re-encoded as H.264 MP4 with `yuv420p` and fast-start metadata; task clips are 1280×720. Source mosaics are preserved. Videos are silent and use click-to-play controls, with no automatic downloads on page load.
- Visual reference: [SCIZOR](https://ut-austin-rpl.github.io/SCIZOR/). Layout and styles were authored for this page; no third-party site assets or JavaScript libraries are bundled.

Edit the text and result tables in `index.html`, styles in `assets/css/style.css`, and tab/citation behavior in `assets/js/main.js`. Both result tables remain visible when JavaScript is disabled.

The adaptation-time columns report aggregate H100 evaluation time. Physical-task results measure complete calibration-and-retargeting pipelines. See the paper for the experimental scope and failure-time convention.
