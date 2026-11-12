"""Match images across scale changes and save a matching visualization."""

import argparse
from pathlib import Path
import sys

import cv2 as cv
import numpy as np
import pydegensac
import torch

from infer import positive_int, read_image, resolve_device, save_image
from model.sample import Transformer
from rrsi.sampling import cartesian_to_logpolar
from tools_.desc_tools import preprocess_Transformer_layer
from tools_.detect_tools import FAST_gradient_kps_detect
from tools_.model_tools import load_descriptor
from tools_.utils import expand_to_square
from visualization import draw_matches


def parse_args():
    package_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Match two modality images using an image pyramid."
    )
    parser.add_argument("--image-a", type=Path, required=True)
    parser.add_argument("--image-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("matches_scale.png"))
    parser.add_argument("--checkpoint", type=Path, default=package_dir / "weights" / "rrsi.pt")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--max-keypoints", type=positive_int, default=2000,
                        help="Keypoint budget per detected pyramid level.")
    parser.add_argument("--batch-size", type=positive_int, default=64,
                        help="Number of patches per sampling batch.")
    parser.add_argument("--show", action="store_true", help="Display the saved matching image.")
    return parser.parse_args()


def detect_pyramid(image, max_keypoints):
    image = image.astype(np.float32)
    levels, groups, scales = [], [], []
    for scale in (1.0, 0.5, 0.25):
        level = cv.resize(image, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
        points = FAST_gradient_kps_detect(level, num_max=max_keypoints)
        levels.append(level)
        groups.append(points)
        scales.append(scale)
    for scale in (2.0, 4.0):
        level = cv.resize(image, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
        levels.append(level)
        groups.append(groups[0] * scale)
        scales.append(scale)
    return levels, groups, scales


def image_features(image, model, encoder, device, max_keypoints, batch_size):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    square = expand_to_square(gray)
    height, width = gray.shape
    levels, groups, scales = detect_pyramid(square, max_keypoints)
    keypoints, cartesian_features, logpolar_features = [], [], []
    for level, points, scale in zip(levels, groups, scales):
        original_points = points / scale
        valid = ((original_points[:, 0] >= 0) & (original_points[:, 0] < width)
                 & (original_points[:, 1] >= 0) & (original_points[:, 1] < height))
        points, original_points = points[valid], original_points[valid]
        if len(points) == 0:
            continue
        keypoints.append(original_points)
        sampler = Transformer(
            transform="STN", coords="linear", resolution=128,
            SIFTscale=1500 * 128 / level.shape[0],
        )
        batches = []
        for start in range(0, len(points), batch_size):
            inputs = preprocess_Transformer_layer(level, points[start:start + batch_size], "image", device)
            batches.append(sampler(inputs))
        cartesian = torch.cat(batches)
        logpolar = cartesian_to_logpolar(cartesian)
        cartesian_features.append(model.encode(encoder, cartesian))
        logpolar_features.append(model.encode(encoder, logpolar))
        del cartesian, logpolar, batches
    if not keypoints:
        raise ValueError("No keypoints were found in one of the images.")
    return np.concatenate(keypoints), torch.cat(cartesian_features), torch.cat(logpolar_features)


@torch.inference_mode()
def match_images(image_a, image_b, model, device, max_keypoints=1000, batch_size=64):
    keypoints_a, cartesian_a, logpolar_a = image_features(
        image_a, model, model.encoder_a, device, max_keypoints, batch_size,
    )
    keypoints_b, cartesian_b, logpolar_b = image_features(
        image_b, model, model.encoder_b, device, max_keypoints, batch_size,
    )
    output = model.forward_features(cartesian_a[None], logpolar_a[None], cartesian_b[None], logpolar_b[None])
    features_a = output.descriptor_a[0].float().cpu().numpy()
    features_b = output.descriptor_b[0].float().cpu().numpy()
    matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    matches = matcher.match(
        np.asarray(features_a, dtype=np.float32),
        np.asarray(features_b, dtype=np.float32),
    )
    if len(matches) < 4:
        raise ValueError("At least four descriptor matches are needed for geometric filtering.")
    points_a = np.asarray([keypoints_a[m.queryIdx] for m in matches], dtype=np.float32)
    points_b = np.asarray([keypoints_b[m.trainIdx] for m in matches], dtype=np.float32)
    # The reprojection tolerance is measured in the second image's pixels.
    image_size_ratio = max(image_b.shape[:2]) / max(image_a.shape[:2])
    homography, mask = pydegensac.findHomography(points_a, points_b, 3.0 * max(1.0, image_size_ratio))
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
        print("Running multi-scale descriptor inference on {}...".format(device), flush=True)
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
