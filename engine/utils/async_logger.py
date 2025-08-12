from __future__ import annotations

import json
import threading
import time
import queue
from pathlib import Path


class AsyncJsonlLogger:
    """Minimal async JSONL writer with batching and background thread."""
    def __init__(self, base_dir: Path, batch_size: int = 256, flush_ms: int = 200):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = int(batch_size)
        self.flush_ms = int(flush_ms)
        self._q: "queue.Queue[tuple[str,str]]" = queue.Queue(maxsize=10000)
        self._buffers: dict[str, list[str]] = {}
        self._files: dict[str, any] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="JsonlWriter", daemon=True)
        self._thread.start()

    def emit(self, stream: str, event: dict):
        try:
            if 'type' not in event:
                event['type'] = stream
            line = json.dumps(event, ensure_ascii=True, separators=(",", ":")) + "\n"
            self._q.put_nowait((stream, line))
        except Exception:
            pass

    def _run(self):
        last = time.time()
        while not self._stop.is_set() or not self._q.empty():
            try:
                stream, line = self._q.get(timeout=0.05)
                buf = self._buffers.setdefault(stream, [])
                buf.append(line)
                if len(buf) >= self.batch_size:
                    self._flush(stream)
            except queue.Empty:
                pass
            now = time.time()
            if (now - last) * 1000.0 >= self.flush_ms:
                for s in list(self._buffers.keys()):
                    if self._buffers[s]:
                        self._flush(s)
                last = now
        for s in list(self._buffers.keys()):
            if self._buffers[s]:
                self._flush(s)
        for f in self._files.values():
            try:
                f.flush(); f.close()
            except Exception:
                pass

    def _flush(self, stream: str):
        try:
            fh = self._files.get(stream)
            if fh is None:
                path = self.base_dir / f"{stream}.jsonl"
                fh = open(path, 'a', encoding='utf-8')
                self._files[stream] = fh
            buf = self._buffers.get(stream, [])
            if buf:
                fh.writelines(buf)
                fh.flush()
                self._buffers[stream] = []
        except Exception:
            self._buffers[stream] = []

    def shutdown(self):
        try:
            self._stop.set()
            self._thread.join(timeout=3.0)
        except Exception:
            pass


