from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
from multiprocessing import Pool
import sys, os
from tqdm.auto import tqdm
import cv2


def parse_args() -> Namespace:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("input", nargs="+", help="Input video files")

    return parser.parse_args()


def process(args) -> None:  # type: ignore
    input_video, output_dir = args

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"File {input_video} could not be opened")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_filename = os.path.join(output_dir, os.path.basename(input_video))
    print(f"Saving video file to {out_filename}")
    writer = cv2.VideoWriter(
        out_filename,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            # EOF
            break

        # Mask left person (passenger)
        frame = cv2.rectangle(
            frame,
            (0, 200),
            (620, 1300),
            (114, 114, 114),  # values used in Yolo-NAS training for background
            -1,
        )

        # Mask right person (driver)
        frame = cv2.rectangle(
            frame,
            (900, 190),
            (1600, 1300),
            (114, 114, 114),  # values used in Yolo-NAS training for background
            -1,
        )

        writer.write(frame)

    writer.release()
    cap.release()


def main(args: Namespace) -> int:
    # Sanity check
    for input_video in args.input:
        if not os.path.exists(input_video):
            raise RuntimeError(f"File {input_video} does not exist")

        if not os.path.isfile(input_video):
            raise RuntimeError(f"File {input_video} is not a file")

    os.makedirs(args.output, exist_ok=True)

    input_map = [(input_file, args.output) for input_file in args.input]

    pool = Pool(processes=16)
    list(tqdm(pool.imap(process, input_map), desc="Processing videos"))
    return 0


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(args))
