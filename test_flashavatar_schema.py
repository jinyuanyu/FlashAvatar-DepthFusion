import json
import os
import re
import unittest
from pathlib import Path

from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parent
SCHEMA_PATH = REPO_ROOT / "data_schema.json"

_ENV_TRUE = {"1", "true", "yes", "on"}


def _is_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _ENV_TRUE


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _torch_load_cpu(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _candidate_roots(env_name: str, *defaults: Path):
    raw = os.environ.get(env_name)
    if raw:
        return [Path(raw)]
    return list(defaults)


def _resolve_identity_root(identity: str, roots):
    for root in roots:
        candidate = root / identity
        if candidate.exists():
            return candidate
    return roots[0] / identity


def _resolve_config_path(identity: str, roots):
    for root in roots:
        candidate = root / f"{identity}.yml"
        if candidate.exists():
            return candidate
    return roots[0] / f"{identity}.yml"


def _parse_simple_yaml(path: Path):
    parsed = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip().strip("'\"")
    return parsed


def _sorted_paths(folder: Path, pattern: str):
    return sorted(folder.glob(pattern))


def _assert_contiguous_names(testcase, paths, regex, artifact_name):
    testcase.assertTrue(paths, f"{artifact_name} is empty")
    extracted = []
    for path in paths:
        match = regex.fullmatch(path.name)
        testcase.assertIsNotNone(
            match,
            f"{artifact_name} has unexpected filename: {path.name}",
        )
        extracted.append(int(match.group(1)))
    testcase.assertEqual(
        extracted,
        list(range(len(paths))),
        f"{artifact_name} indices must be contiguous and start at 00000",
    )


def _mask_values(path: Path):
    with Image.open(path) as image:
        colors = image.getcolors(maxcolors=3)
    return None if colors is None else {value for _, value in colors}


class TestFlashAvatarSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = _load_json(SCHEMA_PATH)
        cls.dataset_root = Path(
            os.environ.get("FLASHAVATAR_DATASET_ROOT", REPO_ROOT / "dataset")
        )
        cls.tracker_output_roots = _candidate_roots(
            "FLASHAVATAR_TRACKER_OUTPUT_ROOT",
            REPO_ROOT / "metrical-tracker" / "output",
            REPO_ROOT.parent / "metrical-tracker" / "output",
        )
        cls.tracker_input_roots = _candidate_roots(
            "FLASHAVATAR_TRACKER_INPUT_ROOT",
            REPO_ROOT / "metrical-tracker" / "input",
            REPO_ROOT.parent / "metrical-tracker" / "input",
        )
        cls.tracker_config_roots = _candidate_roots(
            "FLASHAVATAR_TRACKER_CONFIG_ROOT",
            REPO_ROOT / "metrical-tracker" / "configs" / "actors",
            REPO_ROOT.parent / "metrical-tracker" / "configs" / "actors",
        )
        cls.require_tracker_workspace = _is_truthy(
            "FLASHAVATAR_REQUIRE_TRACKER_WORKSPACE"
        )

        raw_ids = os.environ.get("FLASHAVATAR_TEST_IDS", "").strip()
        if raw_ids:
            cls.identities = [item.strip() for item in raw_ids.split(",") if item.strip()]
        elif cls.dataset_root.exists():
            cls.identities = sorted(
                path.name for path in cls.dataset_root.iterdir() if path.is_dir()
            )
        else:
            cls.identities = []

        cls.required_frame_shapes = cls.schema["final_flashavatar_layout"]["artifacts"][
            "checkpoint_frames"
        ]["minimum_required_keys"]

    def test_schema_metadata(self):
        self.assertEqual(self.schema["schema_version"], "1.0.0")
        self.assertEqual(
            self.schema["pipeline_name"], "flashavatar_mp4_preprocess"
        )

    def test_dataset_identities_exist(self):
        self.assertTrue(
            self.dataset_root.exists(),
            f"dataset root does not exist: {self.dataset_root}",
        )
        self.assertTrue(self.identities, "no identities found under dataset root")

    def test_flashavatar_contracts(self):
        img_regex = re.compile(r"(\d{5})\.jpg")
        neck_regex = re.compile(r"(\d{5})_neckhead\.png")
        mouth_regex = re.compile(r"(\d{5})_mouth\.png")
        frame_regex = re.compile(r"(\d{5})\.frame")

        for identity in self.identities:
            with self.subTest(identity=identity):
                dataset_dir = self.dataset_root / identity
                imgs_dir = dataset_dir / "imgs"
                alpha_dir = dataset_dir / "alpha"
                parsing_dir = dataset_dir / "parsing"

                self.assertTrue(dataset_dir.exists(), f"missing dataset dir: {dataset_dir}")
                self.assertTrue(imgs_dir.exists(), f"missing imgs dir: {imgs_dir}")
                self.assertTrue(alpha_dir.exists(), f"missing alpha dir: {alpha_dir}")
                self.assertTrue(parsing_dir.exists(), f"missing parsing dir: {parsing_dir}")

                img_paths = _sorted_paths(imgs_dir, "*.jpg")
                alpha_paths = _sorted_paths(alpha_dir, "*.jpg")
                neck_paths = _sorted_paths(parsing_dir, "*_neckhead.png")
                mouth_paths = _sorted_paths(parsing_dir, "*_mouth.png")

                self.assertGreater(len(img_paths), 0, "image count must be > 0")
                self.assertEqual(len(img_paths), len(alpha_paths), "count(imgs) != count(alpha)")
                self.assertEqual(
                    len(img_paths),
                    len(neck_paths),
                    "count(imgs) != count(parsing_neckhead)",
                )
                self.assertEqual(
                    len(img_paths),
                    len(mouth_paths),
                    "count(imgs) != count(parsing_mouth)",
                )

                _assert_contiguous_names(self, img_paths, img_regex, "imgs")
                _assert_contiguous_names(self, alpha_paths, img_regex, "alpha")
                _assert_contiguous_names(
                    self, neck_paths, neck_regex, "parsing_neckhead"
                )
                _assert_contiguous_names(
                    self, mouth_paths, mouth_regex, "parsing_mouth"
                )

                for index in range(len(img_paths)):
                    frame5 = f"{index:05d}"
                    image_path = imgs_dir / f"{frame5}.jpg"
                    alpha_path = alpha_dir / f"{frame5}.jpg"
                    neck_path = parsing_dir / f"{frame5}_neckhead.png"
                    mouth_path = parsing_dir / f"{frame5}_mouth.png"

                    self.assertTrue(image_path.exists(), f"missing image: {image_path}")
                    self.assertTrue(alpha_path.exists(), f"missing alpha: {alpha_path}")
                    self.assertTrue(neck_path.exists(), f"missing neckhead: {neck_path}")
                    self.assertTrue(mouth_path.exists(), f"missing mouth: {mouth_path}")

                    with Image.open(image_path) as image:
                        image_size = image.size
                        self.assertEqual(
                            len(image.getbands()),
                            3,
                            f"imgs must be 3-channel: {image_path}",
                        )

                    with Image.open(alpha_path) as alpha:
                        self.assertEqual(
                            len(alpha.getbands()),
                            1,
                            f"alpha must be single-channel: {alpha_path}",
                        )
                        self.assertEqual(
                            alpha.size,
                            image_size,
                            f"alpha resolution mismatch for frame {frame5}",
                        )

                    with Image.open(neck_path) as neck:
                        self.assertEqual(
                            len(neck.getbands()),
                            1,
                            f"neckhead must be single-channel: {neck_path}",
                        )
                        self.assertEqual(
                            neck.size,
                            image_size,
                            f"neckhead resolution mismatch for frame {frame5}",
                        )

                    with Image.open(mouth_path) as mouth:
                        self.assertEqual(
                            len(mouth.getbands()),
                            1,
                            f"mouth must be single-channel: {mouth_path}",
                        )
                        self.assertEqual(
                            mouth.size,
                            image_size,
                            f"mouth resolution mismatch for frame {frame5}",
                        )

                    neck_values = _mask_values(neck_path)
                    mouth_values = _mask_values(mouth_path)
                    self.assertIsNotNone(
                        neck_values,
                        f"neckhead has more than two unique values: {neck_path}",
                    )
                    self.assertIsNotNone(
                        mouth_values,
                        f"mouth has more than two unique values: {mouth_path}",
                    )
                    self.assertTrue(
                        neck_values <= {0, 255},
                        f"neckhead must be binary: {neck_path}",
                    )
                    self.assertTrue(
                        mouth_values <= {0, 255},
                        f"mouth must be binary: {mouth_path}",
                    )

                tracker_output_root = _resolve_identity_root(
                    identity, self.tracker_output_roots
                )
                checkpoint_dir = tracker_output_root / "checkpoint"
                checkpoint_paths = (
                    _sorted_paths(checkpoint_dir, "*.frame")
                    if checkpoint_dir.exists()
                    else []
                )

                config_path = _resolve_config_path(identity, self.tracker_config_roots)
                tracker_input_root = _resolve_identity_root(
                    identity, self.tracker_input_roots
                )
                identity_npy_path = tracker_input_root / "identity.npy"

                if config_path.exists():
                    config = _parse_simple_yaml(config_path)
                    self.assertEqual(
                        config_path.stem,
                        identity,
                        f"tracker config stem must equal identity: {config_path}",
                    )
                    self.assertEqual(
                        config.get("actor"),
                        f"./input/{identity}",
                        f"unexpected actor field in {config_path}",
                    )
                    self.assertEqual(
                        config.get("save_folder"),
                        "./output/",
                        f"unexpected save_folder in {config_path}",
                    )
                    self.assertEqual(
                        config.get("begin_frames"),
                        "1",
                        f"begin_frames must be 1 in {config_path}",
                    )
                elif self.require_tracker_workspace:
                    self.fail(f"missing tracker config: {config_path}")

                if tracker_input_root.exists():
                    self.assertTrue(
                        identity_npy_path.exists(),
                        f"missing identity.npy: {identity_npy_path}",
                    )
                elif self.require_tracker_workspace:
                    self.fail(f"missing tracker input dir: {tracker_input_root}")

                if checkpoint_paths:
                    _assert_contiguous_names(
                        self, checkpoint_paths, frame_regex, "checkpoint_frames"
                    )
                    self.assertEqual(
                        len(checkpoint_paths) + 1,
                        len(img_paths),
                        "count(checkpoint_frames) + 1 must equal count(imgs)",
                    )

                    for frame_index, frame_path in enumerate(checkpoint_paths):
                        payload = _torch_load_cpu(frame_path)
                        self.assertIn("flame", payload, f"missing flame in {frame_path}")
                        self.assertIn("opencv", payload, f"missing opencv in {frame_path}")

                        for key, expected_shape in self.required_frame_shapes.items():
                            if "." in key:
                                parent, child = key.split(".", 1)
                                self.assertIn(
                                    parent,
                                    payload,
                                    f"missing {parent} in {frame_path}",
                                )
                                self.assertIn(
                                    child,
                                    payload[parent],
                                    f"missing {key} in {frame_path}",
                                )
                                actual_shape = tuple(payload[parent][child].shape)
                            else:
                                self.assertIn(
                                    key,
                                    payload,
                                    f"missing {key} in {frame_path}",
                                )
                                value = payload[key]
                                actual_shape = tuple(
                                    getattr(value, "shape", (len(value),))
                                )

                            self.assertEqual(
                                actual_shape,
                                tuple(expected_shape),
                                f"{key} shape mismatch in {frame_path}",
                            )

                        img_size = payload["img_size"]

                        expected_image_path = imgs_dir / f"{frame_index + 1:05d}.jpg"
                        self.assertTrue(
                            expected_image_path.exists(),
                            f"missing mapped image for {frame_path}: {expected_image_path}",
                        )
                        with Image.open(expected_image_path) as mapped_image:
                            expected_size = mapped_image.size

                        img_size_pair = tuple(int(value) for value in img_size)
                        self.assertEqual(
                            img_size_pair,
                            expected_size,
                            f"img_size does not match mapped image for {frame_path}",
                        )

if __name__ == "__main__":
    unittest.main()
