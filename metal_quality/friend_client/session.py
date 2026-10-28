"""Main run loop: tap N times per object, get instant predictions, then a final batch verdict."""
import asyncio
import glob
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from record import record_one           # noqa: E402
from validate import check_valid        # noqa: E402
from ws_client import MetalClient       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def object_id_from_name(fname):
    """obj_001_tap_03.wav -> obj_001 ; metal_obj0_t0.wav -> metal_obj0"""
    stem = os.path.splitext(os.path.basename(fname))[0]
    m = re.match(r"(.+?)[._-]?tap[._-]?\d*$", stem, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.match(r"(.+?)[._-]?t\d+$", stem, re.IGNORECASE)
    return m.group(1) if m else stem


def save_log(cfg, object_id, tap_id, pcm, resp):
    d = os.path.join(cfg.get("local_log_dir", "./logs"))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{object_id}_{tap_id}.npy"), "wb") as f:
        np.save(f, pcm)
    with open(os.path.join(d, "session_log.jsonl"), "a") as f:
        f.write(json.dumps({"ts": time.time(), "object_id": object_id,
                            "tap_id": tap_id, "response": resp}) + "\n")


async def run(cfg):
    mode = cfg.get("input_mode", "serial")
    taps_per_object = int(cfg.get("taps_per_object", 5))
    files = []
    if mode == "wav_folder":
        files = sorted(glob.glob(os.path.join(cfg.get("wav_folder", "./taps"), "*.wav")))
        if not files:
            print("no wav files found in", cfg.get("wav_folder"))
            return

    # group wav files by object id
    groups = {}
    for fp in files:
        groups.setdefault(object_id_from_name(fp), []).append(fp)

    async with MetalClient(cfg) as client:
        print("connected to", cfg["server_url"])
        if mode == "wav_folder":
            for obj, fps in groups.items():
                collected, payloads = 0, []
                for fp in fps:
                    if collected >= taps_per_object:
                        break
                    got = record_one(cfg, wav_path=fp)
                    if got is None:
                        print("read failed:", fp)
                        continue
                    sig, sr = got
                    ok, reason = check_valid(sig, sr)
                    if not ok:
                        print(f"local reject {os.path.basename(fp)}: {reason}")
                        continue
                    resp = await client.send_tap(sig.tobytes(), obj, collected, sr)
                    save_log(cfg, obj, collected, sig, resp)
                    print(f"  {obj} tap {collected}: {resp.get('quality')} "
                          f"conf={resp.get('confidence')} ({resp.get('reason')})")
                    payloads.append(sig.tobytes())
                    collected += 1
                if payloads:
                    final = await client.send_batch(payloads, obj)
                    print(f"FINAL {obj}: {final.get('quality')} "
                          f"conf={final.get('confidence')} "
                          f"n_valid={final.get('n_valid')}/{final.get('n_taps')}")
        else:
            obj = "live_obj"
            payloads = []
            for i in range(taps_per_object):
                input("Press Enter then tap the object...")
                got = record_one(cfg)
                if got is None:
                    print("no data from ESP32 - check cable")
                    continue
                sig, sr = got
                ok, reason = check_valid(sig, sr)
                if not ok:
                    print(f"local reject: {reason} (re-tap)")
                    continue
                resp = await client.send_tap(sig.tobytes(), obj, i, sr)
                save_log(cfg, obj, i, sig, resp)
                print(f"  tap {i}: {resp.get('quality')} conf={resp.get('confidence')} "
                      f"({resp.get('reason')})")
                payloads.append(sig.tobytes())
            if payloads:
                final = await client.send_batch(payloads, obj)
                print(f"FINAL {obj}: {final.get('quality')} conf={final.get('confidence')}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None, help="override server_url")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--wav-folder", default=None)
    ap.add_argument("--mode", default=None, choices=["serial", "wav_folder"])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    with open(os.path.join(HERE, "config.json")) as f:
        cfg = json.load(f)
    if args.url:
        cfg["server_url"] = args.url
    if args.api_key:
        cfg["api_key"] = args.api_key
    if args.wav_folder:
        cfg["wav_folder"] = args.wav_folder
    if args.mode:
        cfg["input_mode"] = args.mode
    if args.device:
        cfg["device_id"] = args.device
    asyncio.run(run(cfg))
