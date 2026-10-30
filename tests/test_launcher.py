"""Slurm launch orchestration without allocating GPUs."""

import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest


class LauncherTests(unittest.TestCase):
    def test_launch_allocated_workers_and_publish_only_on_success(self):
        script = Path(__file__).resolve().parents[1] / "script/job.sh"
        with TemporaryDirectory(prefix="launcher_", dir=Path(__file__).parent) as temporary:
            root = Path(temporary).resolve()
            (root / "script").mkdir()
            (root / "script/worker.sh").write_text(
                'echo "$*" >> "$LAUNCH_LOG"\n'
                'if [ "$FAIL_WORKER" = 1 ] && [ "$1" != prepare ]; then exit 1; fi\n')

            for gpus, fail in ((1, False), (2, False), (4, False), (2, True)):
                with self.subTest(gpus=gpus, fail=fail):
                    collection = root / f"round_{gpus}_{fail}"
                    collection.mkdir()
                    log = collection / "launch.log"
                    env = {**os.environ, "SLURM_JOB_ID": "8695", "SLURM_JOB_NUM_NODES": "1",
                           "LAUNCH_LOG": str(log), "FAIL_WORKER": str(int(fail))}
                    result = subprocess.run(["bash", str(script), "train", str(root), str(collection),
                                             "previous", str(root), "1024", str(gpus)],
                                            env=env, capture_output=True)
                    self.assertEqual(result.returncode, int(fail), result.stderr)
                    calls = [line.split() for line in log.read_text().splitlines()]
                    self.assertEqual([call[0] for call in calls], ["prepare", "train"])
                    self.assertEqual(calls[1][-1], str(gpus))
                    self.assertEqual((collection / "training.ready").exists(), not fail)
            env["SLURM_JOB_NUM_NODES"] = "2"
            rejected = root / "rejected"
            rejected.mkdir()
            result = subprocess.run(["bash", str(script), "train", str(root), str(rejected),
                                     "previous", str(root), "512", "1"], env=env, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((rejected / "training.ready").exists())


if __name__ == "__main__":
    unittest.main()
