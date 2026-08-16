from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any


class TrainingBridge:
    def __init__(self, repo: Path):
        executable = repo / "node_modules" / ".bin" / ("tsx.cmd" if os.name == "nt" else "tsx")
        self.process = subprocess.Popen(
            [str(executable), "training-server.ts"],
            cwd=repo,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.process.stdin or not self.process.stdout:
            raise RuntimeError("training bridge pipes are unavailable")
        self.process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            stderr = self.process.stderr.read() if self.process.stderr else ""
            raise RuntimeError(f"training bridge stopped unexpectedly: {stderr}")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "training bridge returned an unknown error"))
        return response

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.request({"cmd": "close"})
            except Exception:
                self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()

    def __enter__(self) -> "TrainingBridge":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
