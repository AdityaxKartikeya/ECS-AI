"""WebSocket client: streams taps to the server, receives predictions instantly."""
import asyncio
import json

import websockets


class MetalClient:
    def __init__(self, cfg):
        self.cfg = cfg
        url = cfg["server_url"].rstrip("/")
        self.uri = f"{url}/ws/ingest?api_key={cfg['api_key']}"

    async def __aenter__(self):
        self.ws = await websockets.connect(self.uri, max_size=4 * 1024 * 1024,
                                           ping_interval=20, ping_timeout=20)
        return self

    async def __aexit__(self, *exc):
        await self.ws.close()

    async def send_tap(self, pcm_bytes, object_id, tap_id, sample_rate=None):
        sr = sample_rate or self.cfg["sample_rate"]
        await self.ws.send(json.dumps({
            "type": "meta",
            "object_id": object_id,
            "tap_id": str(tap_id),
            "sample_rate": int(sr),
            "device_id": self.cfg.get("device_id", "unknown"),
        }))
        await self.ws.send(pcm_bytes)
        return json.loads(await self.ws.recv())

    async def send_batch(self, list_of_pcm, object_id, sample_rate=None):
        sr = sample_rate or self.cfg["sample_rate"]
        await self.ws.send(json.dumps({
            "type": "batch",
            "n": len(list_of_pcm),
            "object_id": object_id,
            "sample_rate": int(sr),
            "device_id": self.cfg.get("device_id", "unknown"),
        }))
        for pcm in list_of_pcm:
            await self.ws.send(pcm)
        return json.loads(await self.ws.recv())


async def _demo(cfg):
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from record import record_one
    from validate import check_valid

    async with MetalClient(cfg) as client:
        print("connected:", client.uri)
        for i in range(int(cfg.get("taps_per_object", 5))):
            got = record_one(cfg)
            if got is None:
                print("capture failed, skipping")
                continue
            sig, sr = got
            ok, reason = check_valid(sig, sr)
            if not ok:
                print(f"local reject: {reason}")
                continue
            resp = await client.send_tap(sig.tobytes(), "demo_obj", i, sr)
            print(f"tap {i}: {resp.get('quality')} conf={resp.get('confidence')} "
                  f"ok={resp.get('features_ok')} reason={resp.get('reason')}")


if __name__ == "__main__":
    import json
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "config.json")) as f:
        cfg = json.load(f)
    asyncio.run(_demo(cfg))
