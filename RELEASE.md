# YOCO website release

- Branch: `release/yoco-website`.
- Base: `origin/baselines` at `965c0257bf0e70777675c609b35289fb7b18e1e1`, the newest remote branch tip by commit date after fetching on 12 October 2026.
- Website: `docs/`, ready for GitHub Pages from the branch's `/docs` directory.
- Research implementation, model assets, configurations, portable training/evaluation scripts, and existing tests are retained.

## Cleanup

Removed the 13.7 MB example HDF5 dataset from `visualization/`; users should provide their own dataset path. Removed three local macOS orchestration wrappers (`run_all_finetune_baselines_mac.sh`, `run_full_finetune_hypernet_visualize_mac.sh`, and `run_hypernet_finetune_and_export_mac.sh`); the underlying Python training, evaluation, and export tools remain available. One wrapper included an installation-specific Python path.

Added ignore patterns for local environments, operating-system files, and agent/editor state. The website includes only assets it actually references, with no node modules, generated caches, or original large video masters.

Local untracked files that predated this task are preserved and are not included in the release commit. Research branches retain the original tracked files through Git history.

## Validation scope

The website is checked for local asset and anchor resolution, browser rendering at desktop and phone sizes, result-tab and citation interactions, and MP4 decoding. Research algorithms are unchanged; this website release is not a new training or hardware-evaluation result.
