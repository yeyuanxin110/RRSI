"""Match two images and save a matching visualization."""

import argparse
from pathlib import Path
import sys

import cv2 as cv
import numpy as np
import pydegensac
import torch

from model.sample import Transformer
from tools_.desc_tools import preprocess_Transformer_layer
from tools_.detect_tools import keypoint_detect
from tools_.model_tools import load_descriptor
from tools_.utils import expand_to_square
from visualization import draw_matches
from rrsi.sampling import cartesian_to_logpolar


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return number


def parse_args():
    package_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Match two modality images and save a yellow-line matching image."
    )
    parser.add_argument("--image-a", type=Path, required=True)
    parser.add_argument("--image-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("matches.png"))
    parser.add_argument("--checkpoint", type=Path, default=package_dir / "weights" / "rrsi.pt")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--max-keypoints", type=positive_int, default=7500)
    parser.add_argument("--batch-size", type=positive_int, default=128,
                        help="Number of patches per descriptor inference batch.")
    parser.add_argument("--show", action="store_true", help="Display the saved matching image.")
    return parser.parse_args()


def read_image(path):
    if not path.is_file():
        raise FileNotFoundError("Image not found: {}".format(path))
    # imdecode also supports Windows paths containing Chinese characters.
    image = cv.imdecode(np.fromfile(str(path), dtype=np.uint8), cv.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image: {}".format(path))
    return image


def resolve_device(name):
    if name == "auto":
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices are cpu and cuda.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable. Use --device cpu.")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError("CUDA device index is out of range: {}".format(index))
        device = torch.device("cuda", index)
    return device


def save_image(path, image):
    suffix = path.suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        raise ValueError("Output must have a PNG, JPEG, BMP or TIFF extension.")
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv.imencode(suffix, image)
    if not ok:
        raise RuntimeError("Could not encode output image: {}".format(path))
    encoded.tofile(str(path))


def sample_image_patches(image, device, max_keypoints, batch_size):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    square = expand_to_square(gray)
    _, groups, _, _ = keypoint_detect(square, num_max=max_keypoints)
    points = groups[0]
    height, width = gray.shape
    points = points[(points[:, 0] < width) & (points[:, 1] < height)]
    if len(points) == 0:
        raise ValueError("No keypoints were found in one of the images.")
    sampler = Transformer(
        transform="STN", coords="linear", resolution=128,
        SIFTscale=1500 * 128 / square.shape[0],
    )
    batches = []
    for start in range(0, len(points), batch_size):
        inputs = preprocess_Transformer_layer(square, points[start:start + batch_size], "image", device)
        batches.append(sampler(inputs))
    return points, torch.cat(batches)


@torch.inference_mode()
def match_images(image_a, image_b, model, device, max_keypoints=7500, batch_size=128):
    keypoints_a, cartesian_a = sample_image_patches(image_a, device, max_keypoints, batch_size)
    keypoints_b, cartesian_b = sample_image_patches(image_b, device, max_keypoints, batch_size)
    logpolar_a = cartesian_to_logpolar(cartesian_a)
    logpolar_b = cartesian_to_logpolar(cartesian_b)
    output = model(cartesian_a, logpolar_a, cartesian_b, logpolar_b)
    features_a = output.descriptor_a.float().cpu().numpy()
    features_b = output.descriptor_b.float().cpu().numpy()
    matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    matches = matcher.match(
        np.asarray(features_a, dtype=np.float32),
        np.asarray(features_b, dtype=np.float32),
    )
    if len(matches) < 4:
        raise ValueError("At least four descriptor matches are needed for geometric filtering.")
    points_a = np.asarray([keypoints_a[m.queryIdx] for m in matches], dtype=np.float32)
    points_b = np.asarray([keypoints_b[m.trainIdx] for m in matches], dtype=np.float32)
    homography, mask = pydegensac.findHomography(points_a, points_b, 3.0)
    if homography is None or mask is None or not np.isfinite(homography).all():
        raise ValueError("Geometric filtering could not estimate a valid homography.")
    inlier_matches = [match for match, keep in zip(matches, np.asarray(mask).reshape(-1)) if keep]
    if len(inlier_matches) < 4:
        raise ValueError("Geometric filtering retained fewer than four matches.")
    canvas = draw_matches(keypoints_a, keypoints_b, inlier_matches, image_a, image_b)
    return canvas, len(matches), len(inlier_matches)


def main():
    args = parse_args()
    try:
        if args.output.resolve() in (args.image_a.resolve(), args.image_b.resolve()):
            raise ValueError("Choose an output path different from the two input paths.")
        device = resolve_device(args.device)
        np.random.seed(2026)
        torch.manual_seed(2026)
        cv.setRNGSeed(2026)
        image_a = read_image(args.image_a)
        image_b = read_image(args.image_b)
        model = load_descriptor(args.checkpoint, device=device)
        print("Running descriptor inference on {}...".format(device), flush=True)
        canvas, matches, inliers = match_images(
            image_a, image_b, model, device,
            max_keypoints=args.max_keypoints, batch_size=args.batch_size,
        )
        save_image(args.output, canvas)
        print("Matches: {} | Geometric inliers: {}".format(matches, inliers))
        print("Saved: {}".format(args.output.resolve()))
        if args.show:
            cv.imshow("RRSI matches", canvas)
            cv.waitKey(0)
            cv.destroyAllWindows()
    except (OSError, ValueError, RuntimeError, cv.error, np.linalg.LinAlgError) as error:
        print("Error: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
