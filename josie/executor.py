"""One worker thread for ALL MLX/Metal work (ASR, TTS) so models never contend for the GPU."""
import asyncio, functools
from concurrent.futures import ThreadPoolExecutor


class SerialExecutor:
    def __init__(self):
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")

    async def run(self, fn, *a, **k):
        return await asyncio.get_running_loop().run_in_executor(self._pool, functools.partial(fn, *a, **k))
