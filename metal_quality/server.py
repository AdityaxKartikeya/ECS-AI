"""FastAPI prediction server for metal type / quality from tap audio."""
import io, json, os, time, uuid, asyncio
from collections import deque
from datetime import datetime

import numpy as np
import pickle
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Header, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse

from features import (extract, extract_full, peer_features, check_valid,
                      FEATURE_ORDER, PEER_FEATURE_ORDER, FS_DEFAULT)

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")
LOGS = os.path.join(BASE, "logs")
os.makedirs(LOGS, exist_ok=True)

API_KEY = os.environ.get("API_KEY", "dev-secret-change-me")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.7"))
MIN_TAPS_FOR_PURITY = int(os.environ.get("MIN_TAPS", "3"))

# ---- live state for dashboard ----
RECENT = deque(maxlen=500)
SUBSCRIBERS = []          # SSE queues
WS_CLIENTS = []           # dashboard websocket clients

def broadcast(rec):
    for q in list(SUBSCRIBERS):
        try:
            q.put_nowait(rec)
        except Exception:
            pass
    dead = []
    for ws in list(WS_CLIENTS):
        try:
            # fire-and-forget; store coroutine not awaited here, use loop
            pass
        except Exception:
            dead.append(ws)
    # schedule async sends
    try:
        loop = asyncio.get_event_loop()
        for ws in list(WS_CLIENTS):
            loop.create_task(_ws_send(ws, rec))
    except RuntimeError:
        pass

async def _ws_send(ws, rec):
    try:
        await ws.send_json({"type": "prediction", "data": rec})
    except Exception:
        if ws in WS_CLIENTS:
            WS_CLIENTS.remove(ws)

# ---- load artifacts ----
def load_artifacts():
    with open(os.path.join(MODELS, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODELS, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODELS, "feature_columns.json")) as f:
        feat_cols = json.load(f)
    with open(os.path.join(MODELS, "label_classes.json")) as f:
        classes = json.load(f)
    return model, scaler, feat_cols, classes

model, scaler, FEAT_COLS, CLASSES = load_artifacts()
COL_INDEX = {name: i for i, name in enumerate(FEATURE_ORDER)}

app = FastAPI(title="Metal Quality Prediction Server")

# ---- peer bridge: subscribe to the friend's /features stream, mirror it here ----
# His server only publishes; this task is the read side. Each tap event it sees is
# converted into our record shape and pushed into RECENT + broadcast, so the local
# dashboard shows his taps at the same instant his side produces them.
PEER_URL = os.environ.get("PEER_FEATURES_URL", "ws://10.67.123.28:6699/features")
PEER_KEY = os.environ.get("PEER_API_KEY", "3c97e78e50f8e35a5580b83d3cff77fd")
PEER_BRIDGE = os.environ.get("PEER_BRIDGE", "1") == "1"
PEER_RETRY_S = float(os.environ.get("PEER_RETRY_S", "5"))
_peer_counts = {}


def peer_record(ev):
    """His tap event -> the record shape our dashboard already renders.

    He runs no ML, so quality/confidence are null/0 and marked peer:true —
    the dashboard keeps those out of confidence stats and styles them apart.
    """
    try:
        t = datetime.fromisoformat(str(ev.get("ts")).replace("Z", "+00:00")).timestamp()
    except Exception:
        t = time.time()
    oid = ev.get("object_id") or "peer"
    n = _peer_counts.get(oid, 0) + 1
    _peer_counts[oid] = n
    ok = bool(ev.get("features_ok"))
    return {"material": None, "quality": None, "confidence": 0.0,
            "features_ok": ok,
            "reason": ev.get("reason") or ("ok" if ok else "rejected"),
            "n_taps": 1, "peer": True, "object_id": oid, "tap_id": str(n),
            "device_id": ev.get("device_id") or "peer", "timestamp": t,
            "features": ev.get("features") or {},
            "sample_rate": ev.get("sample_rate"), "n_samples": ev.get("n_samples"),
            "duration_s": ev.get("duration_s")}


async def peer_pump():
    """Live WS poller on his output endpoint -> our dashboard, immediately."""
    import websockets
    uri = PEER_URL if "api_key=" in PEER_URL else f"{PEER_URL}?api_key={PEER_KEY}"
    while True:
        try:
            async with websockets.connect(uri, max_size=8 * 1024 * 1024,
                                          ping_interval=20, ping_timeout=20,
                                          open_timeout=10) as ws:
                first = json.loads(await ws.recv())
                print(f"[peer] bridge connected -> {first}", flush=True)
                async for msg in ws:
                    ev = json.loads(msg)
                    if ev.get("action") != "tap":
                        continue
                    rec = peer_record(ev)
                    RECENT.appendleft(rec)
                    broadcast(rec)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[peer] bridge down ({e}); retry in {PEER_RETRY_S:.0f}s", flush=True)
        await asyncio.sleep(PEER_RETRY_S)


@app.on_event("startup")
async def _start_peer_bridge():
    if PEER_BRIDGE:
        asyncio.create_task(peer_pump())


def auth(x_api_key):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="bad api key")


def decode_audio(raw: bytes, filename: str, sr_expected: int):
    """Return (sig: np.ndarray int/float, sr)."""
    import io as _io
    name = (filename or "").lower()
    if name.endswith(".wav"):
        from scipy.io import wavfile
        sr, data = wavfile.read(_io.BytesIO(raw))
        data = np.asarray(data)
        if data.ndim > 1:
            data = data[:, 0]
        return data, int(sr)
    # assume raw int16 little-endian PCM
    sig = np.frombuffer(raw, dtype="<i2")
    return sig, sr_expected


def resample_if_needed(sig, sr, target):
    if sr == target:
        return sig, sr
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(sr, target)
    sig = resample_poly(sig, target // g, sr // g)
    return sig, target


def extract_vector(sig, sr):
    """Return (model_vector, peer_features, reason).

    model_vector keeps our 27-column order for predict(); peer_features is the
    20-key dict we publish to the dashboard / peer, in PEER_FEATURE_ORDER.
    """
    vec, feats, reason = extract_full(sig, sr)
    if vec is None:
        return None, {}, reason
    # reorder to FEAT_COLS
    full = {name: float(vec[i]) for i, name in enumerate(FEATURE_ORDER)}
    v = np.array([full[c] for c in FEAT_COLS], dtype=np.float64)
    return v, peer_features(feats), "ok"


def mean_peer_features(pfs):
    """Average the 20-key peer dict across taps."""
    if not pfs:
        return {}
    return {k: float(np.mean([p[k] for p in pfs])) for k in PEER_FEATURE_ORDER}


def predict_from_vector(v):
    Xs = scaler.transform(v.reshape(1, -1))
    proba = model.predict_proba(Xs)[0]
    idx = int(np.argmax(proba))
    return CLASSES[idx], float(proba[idx]), {CLASSES[i]: float(proba[i]) for i in range(len(CLASSES))}


def log_request(object_id, tap_id, device_id, raw, result):
    day = datetime.now().strftime("%Y%m%d")
    ddir = os.path.join(LOGS, day)
    os.makedirs(ddir, exist_ok=True)
    uid = uuid.uuid4().hex[:8]
    tag = f"{object_id}_{tap_id}_{uid}"
    with open(os.path.join(ddir, tag + ".pcm"), "wb") as f:
        f.write(raw)
    with open(os.path.join(ddir, "session_log.jsonl"), "a") as f:
        f.write(json.dumps({"ts": time.time(), "object_id": object_id, "tap_id": tap_id,
                           "device_id": device_id, "result": result}) + "\n")


@app.get("/")
def root():
    return {"service": "Metal Quality Prediction Server", "status": "running",
            "endpoints": ["/health", "/predict", "/predict_batch"],
            "classes": CLASSES, "n_features": len(FEAT_COLS)}

@app.get("/health")
def health():
    return {"status": "ok", "classes": CLASSES, "n_features": len(FEAT_COLS)}


@app.post("/predict")
async def predict(
    audio: UploadFile = File(...),
    object_id: str = Form("unknown"),
    tap_id: str = Form("0"),
    sample_rate: str = Form(str(FS_DEFAULT)),
    device_id: str = Form("unknown"),
    x_api_key: str = Header(None),
):
    auth(x_api_key)
    sr_expected = int(sample_rate)
    raw = await audio.read()
    try:
        sig, sr = decode_audio(raw, audio.filename, sr_expected)
        sig, sr = resample_if_needed(sig, sr, FS_DEFAULT)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"decode error: {e}")

    v, pf, reason = extract_vector(sig, sr)
    if v is None:
        result = {"material": None, "quality": None, "confidence": 0.0,
                  "features_ok": False, "reason": reason, "n_taps": 1}
        rec = dict(result); rec["timestamp"]=time.time(); rec["object_id"]=object_id; rec["tap_id"]=tap_id; rec["device_id"]=device_id; rec["features"]={}
        RECENT.appendleft(rec); broadcast(rec)
        log_request(object_id, tap_id, device_id, raw, result)
        return JSONResponse(result)

    label, conf, proba = predict_from_vector(v)
    result = {"material": label.split("_")[0] if "_" in label else label,
              "quality": label, "confidence": round(conf, 4),
              "features_ok": True, "reason": "ok", "n_taps": 1,
              "probabilities": proba}
    if conf < CONFIDENCE_THRESHOLD:
        result["reason"] = "low_confidence"
    feat = pf
    rec = dict(result); rec["timestamp"]=time.time(); rec["object_id"]=object_id; rec["tap_id"]=tap_id; rec["device_id"]=device_id; rec["features"]=feat
    RECENT.appendleft(rec); broadcast(rec)
    log_request(object_id, tap_id, device_id, raw, result)
    return JSONResponse(result)


@app.post("/predict_batch")
async def predict_batch(
    audio: list[UploadFile] = File(...),
    object_id: str = Form("unknown"),
    sample_rate: str = Form(str(FS_DEFAULT)),
    device_id: str = Form("unknown"),
    x_api_key: str = Header(None),
):
    auth(x_api_key)
    sr_expected = int(sample_rate)
    vectors = []
    pfs = []
    reasons = []
    raws = []
    for uf in audio:
        raw = await uf.read()
        raws.append(raw)
        try:
            sig, sr = decode_audio(raw, uf.filename, sr_expected)
            sig, sr = resample_if_needed(sig, sr, FS_DEFAULT)
        except Exception as e:
            reasons.append(f"decode_error:{e}")
            continue
        v, pf, reason = extract_vector(sig, sr)
        if v is None:
            reasons.append(reason)
            continue
        vectors.append(v)
        pfs.append(pf)
        reasons.append("ok")

    if not vectors:
        result = {"material": None, "quality": None, "confidence": 0.0,
                  "features_ok": False, "reason": "no_valid_taps",
                  "n_taps": len(audio), "n_valid": 0, "tap_reasons": reasons}
        rec = dict(result); rec["timestamp"]=time.time(); rec["object_id"]=object_id; rec["tap_id"]="batch"; rec["device_id"]=device_id; rec["features"]={}
        RECENT.appendleft(rec); broadcast(rec)
        for r in raws:
            log_request(object_id, "batch", device_id, r, result)
        return JSONResponse(result)

    avg = np.mean(np.stack(vectors, axis=0), axis=0)
    label, conf, proba = predict_from_vector(avg)
    result = {"material": label.split("_")[0] if "_" in label else label,
              "quality": label, "confidence": round(conf, 4),
              "features_ok": True, "reason": "ok" if conf >= CONFIDENCE_THRESHOLD else "low_confidence",
              "n_taps": len(audio), "n_valid": len(vectors),
              "tap_reasons": reasons, "probabilities": proba}
    feat = mean_peer_features(pfs)
    rec = dict(result); rec["timestamp"]=time.time(); rec["object_id"]=object_id; rec["tap_id"]="batch"; rec["device_id"]=device_id; rec["features"]=feat
    RECENT.appendleft(rec); broadcast(rec)
    for r in raws:
        log_request(object_id, "batch", device_id, r, result)
    return JSONResponse(result)


@app.get("/dashboard")
def dashboard():
    return FileResponse(os.path.join(BASE, "static", "dashboard.html"))

@app.get("/recent")
def recent_endpoint(limit: int = 100):
    items = list(RECENT)[:limit]
    return {"items": items, "count": len(RECENT)}

@app.get("/stats")
def stats():
    total = len(RECENT)
    valid = [r for r in RECENT if r.get("features_ok")]
    rej = [r for r in RECENT if not r.get("features_ok")]
    qcounts = {}
    for r in valid:
        if r.get("quality"):
            qcounts[r.get("quality")] = qcounts.get(r.get("quality"), 0) + 1
    rcounts = {}
    for r in rej:
        rcounts[r.get("reason")] = rcounts.get(r.get("reason"), 0) + 1
    objs = {}
    for r in RECENT:
        o = objs.setdefault(r.get("object_id"), {"taps":0,"valid":0,"peer":0,"conf_sum":0.0})
        o["taps"] += 1
        if r.get("peer"):
            o["peer"] += 1
        elif r.get("features_ok"):
            o["valid"] += 1
            o["conf_sum"] += r.get("confidence", 0)
    for o in objs.values():
        o["avg_conf"] = (o["conf_sum"]/o["valid"]) if o["valid"] else 0.0
    scored = [r for r in valid if not r.get("peer")]
    avg_conf = (sum(r.get("confidence",0) for r in scored)/len(scored)) if scored else 0.0
    return {"total": total, "valid": len(valid), "rejected": len(rej),
            "avg_confidence": round(avg_conf,4),
            "quality_counts": qcounts, "rejection_counts": rcounts,
            "objects": objs, "classes": CLASSES}

@app.get("/events")
async def events():
    q = asyncio.Queue()
    SUBSCRIBERS.append(q)
    async def gen():
        try:
            while True:
                rec = await q.get()
                yield f"data: {json.dumps(rec)}\n\n"
        finally:
            try:
                SUBSCRIBERS.remove(q)
            except ValueError:
                pass
    return StreamingResponse(gen(), media_type="text/event-stream")

@app.websocket("/ws/dashboard")
async def ws_dashboard(websocket: WebSocket):
    await websocket.accept()
    WS_CLIENTS.append(websocket)
    # send recent snapshot
    try:
        await websocket.send_json({"type": "snapshot", "items": list(RECENT)[:100]})
        while True:
            await websocket.receive_text()  # keep alive / ignore
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in WS_CLIENTS:
            WS_CLIENTS.remove(websocket)

@app.websocket("/ws/ingest")
async def ws_ingest(websocket: WebSocket, api_key: str = ""):
    if api_key != API_KEY:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive()
            if "text" in msg and msg["text"] is not None:
                try:
                    meta = json.loads(msg["text"])
                except Exception:
                    await websocket.send_json({"error": "bad meta json"})
                    continue
                if meta.get("type") not in ("meta", "batch"):
                    await websocket.send_json({"error": "expected meta or batch frame first"})
                    continue
                if meta.get("type") == "batch":
                    n = int(meta.get("n", 0))
                    object_id = meta.get("object_id", "unknown")
                    sr_expected = int(meta.get("sample_rate", FS_DEFAULT))
                    device_id = meta.get("device_id", "unknown")
                    vecs, pfs_i, reasons, raws = [], [], [], []
                    for _ in range(max(0, n)):
                        raw_i = await websocket.receive_bytes()
                        raws.append(raw_i)
                        try:
                            sig_i, sr_i = decode_audio(raw_i, "tap.pcm", sr_expected)
                            sig_i, sr_i = resample_if_needed(sig_i, sr_i, FS_DEFAULT)
                        except Exception as e:
                            reasons.append(f"decode_error:{e}")
                            continue
                        v_i, pf_i, r_i = extract_vector(sig_i, sr_i)
                        if v_i is None:
                            reasons.append(r_i)
                            continue
                        vecs.append(v_i)
                        pfs_i.append(pf_i)
                        reasons.append("ok")
                    if not vecs:
                        result = {"material": None, "quality": None, "confidence": 0.0,
                                  "features_ok": False, "reason": "no_valid_taps",
                                  "n_taps": n, "n_valid": 0, "tap_reasons": reasons}
                    else:
                        avg_i = np.mean(np.stack(vecs, axis=0), axis=0)
                        label_i, conf_i, proba_i = predict_from_vector(avg_i)
                        result = {"material": label_i.split("_")[0] if "_" in label_i else label_i,
                                  "quality": label_i, "confidence": round(conf_i, 4),
                                  "features_ok": True,
                                  "reason": "ok" if conf_i >= CONFIDENCE_THRESHOLD else "low_confidence",
                                  "n_taps": n, "n_valid": len(vecs),
                                  "tap_reasons": reasons, "probabilities": proba_i}
                    feat_i = mean_peer_features(pfs_i)
                    rec = dict(result); rec["timestamp"] = time.time(); rec["object_id"] = object_id
                    rec["tap_id"] = "batch"; rec["device_id"] = device_id; rec["features"] = feat_i
                    RECENT.appendleft(rec); broadcast(rec)
                    for raw_i in raws:
                        log_request(object_id, "batch", device_id, raw_i, result)
                    await websocket.send_json(rec)
                    continue
                # wait for binary audio frame
                binmsg = await websocket.receive_bytes()
                object_id = meta.get("object_id", "unknown")
                tap_id = meta.get("tap_id", "0")
                sr_expected = int(meta.get("sample_rate", FS_DEFAULT))
                device_id = meta.get("device_id", "unknown")
                raw = binmsg
                try:
                    sig, sr = decode_audio(raw, "tap.pcm", sr_expected)
                    sig, sr = resample_if_needed(sig, sr, FS_DEFAULT)
                except Exception as e:
                    await websocket.send_json({"features_ok": False, "reason": f"decode_error:{e}"})
                    continue
                v, pf, reason = extract_vector(sig, sr)
                if v is None:
                    rec = {"material": None, "quality": None, "confidence": 0.0,
                           "features_ok": False, "reason": reason, "n_taps": 1,
                           "timestamp": time.time(), "object_id": object_id,
                           "tap_id": tap_id, "device_id": device_id, "features": {}}
                    RECENT.appendleft(rec); broadcast(rec)
                    await websocket.send_json(rec)
                    continue
                label, conf, proba = predict_from_vector(v)
                result = {"material": label.split("_")[0] if "_" in label else label,
                          "quality": label, "confidence": round(conf, 4),
                          "features_ok": True,
                          "reason": "ok" if conf >= CONFIDENCE_THRESHOLD else "low_confidence",
                          "n_taps": 1, "probabilities": proba}
                feat = pf
                rec = dict(result); rec["timestamp"] = time.time(); rec["object_id"] = object_id
                rec["tap_id"] = tap_id; rec["device_id"] = device_id; rec["features"] = feat
                RECENT.appendleft(rec); broadcast(rec)
                log_request(object_id, tap_id, device_id, raw, result)
                await websocket.send_json(rec)
            elif "bytes" in msg and msg["bytes"] is not None:
                # stray binary without meta
                await websocket.send_json({"error": "send meta text frame first"})
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
