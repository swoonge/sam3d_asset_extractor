import pytest

from sam3d_asset_extractor.cli import build_parser


def test_parser_includes_required_flags():
    parser = build_parser()
    options: set[str] = set()
    for action in parser._actions:
        for flag in action.option_strings:
            options.add(flag)
    assert "--image" in options
    assert "--output-dir" in options
    assert "--sam2-mode" in options
    assert "--sam3d-input" in options
    assert "--decimate" in options
    assert "--no-decimate" in options


def test_parser_rejects_missing_required():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_rejects_missing_depth():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--image", "x.jpg", "--output-dir", "out", "--cam-k", "k.txt"])


def test_parser_rejects_missing_cam_k():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--image", "x.jpg", "--output-dir", "out", "--depth-image", "d.png"])


def _min_argv() -> list[str]:
    """Minimum required argv: image + output-dir + depth-image + cam-k."""
    return [
        "--image", "x.jpg",
        "--output-dir", "out",
        "--depth-image", "d.png",
        "--cam-k", "k.txt",
    ]


def test_parser_parses_minimum():
    parser = build_parser()
    args = parser.parse_args(_min_argv())
    assert args.image.name == "x.jpg"
    assert args.depth_image.name == "d.png"
    assert args.cam_k.name == "k.txt"
    assert args.sam2_mode == "auto"
    assert args.sam3d_input == "full"
    assert args.decimate is True


def test_parser_mode_overrides():
    parser = build_parser()
    args = parser.parse_args(_min_argv() + [
        "--sam2-mode", "manual",
        "--sam3d-input", "cropped",
        "--no-decimate",
        "--decimate-target-faces", "5000",
    ])
    assert args.sam2_mode == "manual"
    assert args.sam3d_input == "cropped"
    assert args.decimate is False
    assert args.decimate_target_faces == 5000
