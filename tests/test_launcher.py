"""Slurm launch orchestration without allocating GPUs."""

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class LauncherTests(unittest.TestCase):
    def test_launch_allocated_workers_and_publish_only_on_success(self):
        script = Path(__file__).resolve().parents[1] / "script/job.sh"
        with TemporaryDirectory(prefix="launcher_", dir=Path(__file__).parent) as temporary:
            root = Path(temporary).resolve()
            commands = root / "bin"
            commands.mkdir()
            runner = commands / "srun"
            runner.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
with open(os.environ["LAUNCH_LOG"], "a") as stream:
    stream.write(json.dumps({"args": sys.argv[1:]}) + "\\n")
if os.environ["FAIL_WORKER"] == "1" and "prepare" not in sys.argv:
    sys.exit(1)
''')
            runner.chmod(0o755)

            for nodes, tasks, fail in ((1, 1, False), (1, 2, False), (1, 4, False), (1, 2, True)):
                with self.subTest(nodes=nodes, tasks=tasks, fail=fail):
                    collection = root / f"round_{nodes}_{tasks}_{fail}"
                    collection.mkdir()
                    log = collection / "launch.jsonl"
                    env = {**os.environ, "PATH": str(commands) + os.pathsep + os.environ["PATH"],
                           "SLURM_JOB_ID": "8695", "SLURM_JOB_NODELIST": "node[2-3]",
                           "SLURM_JOB_NUM_NODES": str(nodes), "LAUNCH_LOG": str(log),
                           "SLURM_NTASKS": str(tasks),
                           "SLURM_CPUS_PER_TASK": "4",
                           "FAIL_WORKER": str(int(fail))}
                    result = subprocess.run(["bash", str(script), "train", str(root), str(collection),
                                             "previous", str(root), "1024"], env=env, capture_output=True)
                    self.assertEqual(result.returncode, int(fail), result.stderr)
                    calls = [json.loads(line) for line in log.read_text().splitlines()]
                    self.assertEqual(len(calls), 2)
                    self.assertIn(f"--ntasks={nodes}", calls[0]["args"])
                    self.assertIn("prepare", calls[0]["args"])
                    self.assertIn("--ntasks=1", calls[1]["args"])
                    self.assertIn(f"--gpus-per-task=h100:{tasks}", calls[1]["args"])
                    self.assertEqual(calls[1]["args"][-1], str(tasks))
                    self.assertIn("--gpu-bind=none", calls[1]["args"])
                    self.assertIn("--kill-on-bad-exit=1", calls[1]["args"])
                    self.assertEqual((collection / "training.ready").exists(), not fail)
            env["SLURM_JOB_NUM_NODES"] = "2"
            rejected = root / "rejected"
            rejected.mkdir()
            result = subprocess.run(["bash", str(script), "train", str(root), str(rejected),
                                     "previous", str(root), "512"], env=env, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((rejected / "training.ready").exists())


if __name__ == "__main__":
    unittest.main()
