from misc.generate_test_sequences import generate_videos
from misc.find_test_segments import BLOCK_LIST, get_mimicry_normal, get_mimicry_random_alignment, get_non_mimicry_normal, get_non_mimicry_random_alignment
from misc.shared import BASE_DIR
import torch
import shutil

OUTPUT_DIR = BASE_DIR / "outputs/super_new"

from collections import defaultdict

a = defaultdict(list)


def find_overlapping_sequences():
    for file_ in (base_dir / "mimicry_gt").glob("*.mp4"):
        *_, session, start, stop = file_.stem.split("_")
        frames = set(range(int(start), int(stop)))
        if any([x[0] & frames for x in a[session]]):
            print(file_.stem, start, stop)
            for x in a[session]:
                print("->", x[1], min(x[0]), max(x[0]))
            print("----------")
        a[session].append((frames, file_.stem))


def cache_sequence(name, func, args=[]):
    json_file = name.with_suffix(suffix=".pt")
    if not json_file.exists():
        sequences = func(*args)
        torch.save(sequences, json_file)
    else:
        sequences = torch.load(json_file)
    return sequences


def create_mimicry_random_alignment(name):
    output_name = OUTPUT_DIR / f"{name}_random_align"
    sequences = cache_sequence(
        output_name, get_mimicry_random_alignment, (OUTPUT_DIR / f"{name}_gt",),
    )

    generate_videos(sequences, output_name)


def create_mimicry_gt(name):
    output_name = OUTPUT_DIR / f"{name}_gt"
    sequences = cache_sequence(output_name, get_mimicry_normal)
    total_seq = []
    try:
        shutil.rmtree(output_name)
    except FileNotFoundError:
        pass
    output_name.mkdir(exist_ok=True)
    orig = OUTPUT_DIR / f"{name}_gt_original"

    for seq in sequences:
        seq = list(seq)
        if seq[0] not in BLOCK_LIST:

            *_, start, stop = seq[0].split(".")[0].split("_")

            print(seq[0], int(stop) - int(start))
            total_seq.append(seq)

            orig_file = (orig / seq[0]).with_suffix(".mp4")

            if orig_file.exists():
                shutil.copy(
                    orig_file, (output_name / seq[0]).with_suffix(".mp4"),
                )
                (output_name / "meta").mkdir(exist_ok=True)
                shutil.copy(
                    (orig_file.parent / "meta" / orig_file.stem).with_suffix(".txt"),
                    (output_name / "meta" / orig_file.stem).with_suffix(".txt"),
                )

    generate_videos(total_seq, output_name)


def create_non_mimicry_random_alignment(name):
    output_name = OUTPUT_DIR / f"{name}_no_mimicry_rand_align"
    sequences = cache_sequence(output_name, get_non_mimicry_random_alignment, (OUTPUT_DIR / f"{name}_random_align",))
    generate_videos(sequences, output_name)


def create_non_mimicry(name):
    output_name = OUTPUT_DIR / f"{name}_no_mimicry_gt"
    sequences = cache_sequence(output_name, get_non_mimicry_normal, (OUTPUT_DIR / f"{name}_no_mimicry_rand_align",))
    generate_videos(sequences, output_name)

if __name__ == "__main__":
    name = "mimicry"
    # create_mimicry_gt(name)
    # create_mimicry_random_alignment(name)
    create_non_mimicry_random_alignment(name)
    create_non_mimicry(name)
