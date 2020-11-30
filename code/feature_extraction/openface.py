import shutil
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path

import docker
from misc.shared import DATASET_DIR
from tqdm import tqdm


def extract_openface(fps):
    client = docker.from_env()
    msg = "Extracing openface from videos"

    for file in tqdm(list(DATASET_DIR.glob(f"*/*/video_{fps}fps.mp4")), desc=msg):
        openface_file = file.parent / f"openface_{fps}fps.csv"

        if openface_file.exists():
            continue

        with tempfile.TemporaryDirectory() as tmpd:
            shutil.copyfile(file, Path(tmpd) / "input.mp4")
            ctnr = client.containers.run(
                "algebr/openface:latest",
                tty=True,
                detach=True,
                volumes={tmpd: {"bind": "/data"}},
            )
            output_file = Path(f"/output/openface.csv")
            try:
                ctnr.exec_run("mkdir /output")
                ctnr.exec_run(
                    f"./build/bin/FeatureExtraction -f /data/input.mp4 -of {output_file.stem} -out_dir /output -2Dfp -3Dfp -pdmparams -pose -aus -gaze"
                )

                archive, stat = ctnr.get_archive(output_file)
                filelike = BytesIO(b"".join(b for b in archive))
                tar = tarfile.open(fileobj=filelike)

                with tempfile.TemporaryDirectory() as tmpd:
                    tar.extract(output_file.name, tmpd)
                    tmp_open_face_file = Path(tmpd) / output_file.name
                    openface_file.write_text(tmp_open_face_file.read_text().replace(", ", ","))

            finally:
                ctnr.stop()
            exit()
