"""Face detection, embeddings and labeling.

There is no model training: pretrained insightface (buffalo_l) embeddings, a
person = the mean of their labeled embeddings, a suggestion = cosine similarity
to those means. Every confirmed label immediately improves future suggestions.
"""

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ndrive import core

try:
    from insightface.app import FaceAnalysis
except ImportError:  # dev/test boxes without the ML stack — labeling UI still works, scanning is disabled
    FaceAnalysis = None

log = logging.getLogger("ndrive")

IGNORE = "(not a face)"
STRANGER = "(stranger)"  # real face, unknown person: archived, but never suggested or filterable

# Detection runs on a downscaled copy (cheap — the detector resizes to DET_SIZE anyway), but the
# recognition crop is warped out of the ORIGINAL pixels: shrinking a face to 112px keeps detail,
# upscaling a 50px face into 112px invents it. Measured before this: a 135px face in a 12MP photo
# ended up at 50px and matched its own person no better than a stranger did.
DET_SIZE = 1024  # detector input; larger finds the small faces in group shots
LOAD_MAX_SIDE = 6000  # only a memory guard for absurd images; normal photos are used at full size
# Measured: impostors peak at ~0.26 cosine, true matches sit at p10=0.40, median 0.68.
SUGGEST_MIN_SIM = 0.40
SUGGEST_MARGIN = 0.05  # and the runner-up must be clearly behind, else we ask rather than guess
KEEP_LABEL_IOU = 0.4  # a re-detected face overlapping this much is the same face: carry its label over
# Measured on the family library: labelled faces sit at p5=64px, so 50 drops background noise
# (6.5% of the queue) while keeping everything anyone actually bothered to name.
MIN_FACE_PX = 50
_analyzer = None


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        analyzer = FaceAnalysis(
            name="buffalo_l",
            root=str(core.CACHE / "insightface"),  # on the storage volume: survives container rebuilds
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        analyzer.prepare(ctx_id=0, det_size=(DET_SIZE, DET_SIZE))
        _analyzer = analyzer
    return _analyzer


_scan_lock = threading.Lock()  # ponytail: one sweep at a time — insightface is CPU-hungry and shares the box
_rescan = threading.Event()


def _todo() -> list[tuple[str, float]]:
    with core.db() as con:
        done = {r["path"]: r["mtime"] for r in con.execute("SELECT path, mtime FROM face_scan")}
        return [
            (r["path"], r["mtime"])
            for r in con.execute("SELECT path, mtime FROM photos")
            if Path(r["path"]).suffix.lower() in core.IMAGE_EXTS and done.get(r["path"]) != r["mtime"]
        ]


def pending_count() -> int:
    return len(_todo())


def scan_async() -> None:
    """Fire-and-forget sweep, called after uploads."""
    threading.Thread(target=sweep, daemon=True, name="ndrive-faces").start()


def sweep() -> None:
    """Scan everything pending, one sweep at a time.

    Repeats only when another upload arrived mid-sweep — never loops on `pending_count`,
    which would spin forever on a photo that can't be scanned (corrupt file, no ML stack).
    """
    if not _scan_lock.acquire(blocking=False):
        _rescan.set()  # a sweep is running: ask it for one more pass so our upload isn't missed
        return
    try:
        while True:
            _rescan.clear()
            scan_faces()
            if not _rescan.is_set():
                return
    finally:
        _scan_lock.release()


def scan_faces() -> None:
    """Detect + embed faces for photos not scanned yet (or changed since)."""
    if FaceAnalysis is None:
        log.warning("insightface not installed — face scanning disabled")
        return
    todo = _todo()
    for rel, mtime in todo:
        try:
            _scan_one(rel, mtime)
        except Exception as exc:  # noqa: BLE001 — one unreadable photo must not stop the sweep
            log.warning(f"face scan failed for {rel}: {exc}")
    if todo:
        log.info(f"face scan done: {len(todo)} new photo(s)")


@dataclass
class _Det:
    """A detection carried between the two models; embedding is filled in by the recogniser."""

    bbox: np.ndarray
    kps: np.ndarray
    embedding: np.ndarray = field(default=None)

    @property
    def normed_embedding(self) -> np.ndarray:
        return self.embedding / (np.linalg.norm(self.embedding) or 1.0)


def _detect_and_embed(full_bgr: np.ndarray, small_bgr: np.ndarray, scale: float) -> list[_Det]:
    analyzer = _get_analyzer()
    boxes, kpss = analyzer.det_model.detect(small_bgr, max_num=0, metric="default")
    out = []
    for i in range(len(boxes)):
        if kpss is None:  # no landmarks means no alignment, and an unaligned crop embeds badly
            continue
        box = boxes[i][:4] * scale  # back to original-pixel coords
        if max(box[2] - box[0], box[3] - box[1]) < MIN_FACE_PX:
            continue  # too small to recognise, and nobody wants to label a face in the background
        det = _Det(bbox=box, kps=kpss[i] * scale)
        analyzer.models["recognition"].get(full_bgr, det)  # warps 112x112 out of the ORIGINAL pixels
        out.append(det)
    return out


def _scan_one(rel: str, mtime: float) -> None:
    abs_ = core.resolve(rel)
    if not abs_.is_file():
        return
    with Image.open(abs_) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        if max(img.size) > LOAD_MAX_SIDE:
            img = ImageOps.contain(img, (LOAD_MAX_SIDE, LOAD_MAX_SIDE))
        full = np.asarray(img)[:, :, ::-1]  # insightface wants BGR
        scale = max(max(img.size) / DET_SIZE, 1.0)
        small = (
            np.asarray(img.resize((round(img.width / scale), round(img.height / scale))))[:, :, ::-1]
            if scale > 1
            else full
        )
    found = _detect_and_embed(full, small, scale)
    with core.db() as con:
        old = [
            (tuple(int(v) for v in r["bbox"].split(",")), r["id"], r["label"])
            for r in con.execute("SELECT id, bbox, label FROM faces WHERE path=?", (rel,))
        ]
        for _, face_id, _label in old:
            (core.CACHE / "img" / f"face-{face_id}.jpg").unlink(missing_ok=True)  # ids get reused; drop stale crops
        con.execute("DELETE FROM faces WHERE path=?", (rel,))
        for f in found:
            box = tuple(round(float(v)) for v in f.bbox)  # already in original-pixel coords
            label = next(
                (
                    lab
                    for b, _i, lab in sorted(old, key=lambda o: -_iou(box, o[0]))
                    if lab and _iou(box, b) >= KEEP_LABEL_IOU
                ),
                None,
            )
            con.execute(
                "INSERT OR REPLACE INTO faces(path, bbox, embedding, label) VALUES(?,?,?,?)",
                (rel, ",".join(str(v) for v in box), f.normed_embedding.astype(np.float32).tobytes(), label),
            )
        con.execute("INSERT OR REPLACE INTO face_scan VALUES(?,?)", (rel, mtime))


def _iou(a: tuple, b: tuple) -> float:
    """Overlap between two boxes — used to recognise 'the same face' across a re-scan."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if not inter:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


# --- labeling --------------------------------------------------------------


def _centroids() -> dict[str, np.ndarray]:
    with core.db() as con:
        rows = con.execute(
            "SELECT label, embedding FROM faces WHERE label IS NOT NULL AND label NOT IN (?, ?)", (IGNORE, STRANGER)
        ).fetchall()
    grouped: dict[str, list[np.ndarray]] = {}
    for r in rows:
        grouped.setdefault(r["label"], []).append(np.frombuffer(r["embedding"], dtype=np.float32))
    centroids = {}
    for name, vecs in grouped.items():
        c = np.mean(vecs, axis=0)
        centroids[name] = c / (np.linalg.norm(c) or 1.0)
    return centroids


def unlabeled(limit: int = 30) -> list[dict]:
    """Unlabeled faces with the model's best guess, grouped per suggested person.

    Scoring every pending face (not just the page's worth) is what lets us serve them in
    blocks of the same person: a wrong one stands out from its neighbours at a glance.
    """
    centroids = _centroids()
    with core.db() as con:
        rows = con.execute("SELECT id, path, embedding FROM faces WHERE label IS NULL").fetchall()
    scored = []
    for r in rows:
        vec = np.frombuffer(r["embedding"], dtype=np.float32)
        ranked = sorted(((float(vec @ c), name) for name, c in centroids.items()), reverse=True)
        suggest, sim = None, 0.0
        if ranked:
            sim, best = ranked[0]
            runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
            # guess only when it is both confident and clearly ahead — a wrong pre-selection is
            # worse than none, because confirming it poisons the person's average
            if sim >= SUGGEST_MIN_SIM and sim - runner_up >= SUGGEST_MARGIN:
                suggest = best
        scored.append({"id": r["id"], "path": r["path"], "suggest": suggest, "score": sim})
    # named blocks first (surest name, surest face), the "no idea" ones last
    scored.sort(key=lambda f: (f["suggest"] is None, f["suggest"] or "", -f["score"]))
    return scored[:limit]


def grouped(limit: int = 30) -> list[dict]:
    """The same faces, bundled into per-person blocks for the labeling page."""
    groups: list[dict] = []
    for f in unlabeled(limit):
        if not groups or groups[-1]["suggest"] != f["suggest"]:
            groups.append({"suggest": f["suggest"], "faces": []})
        groups[-1]["faces"].append(f)
    return groups


def unlabeled_count() -> int:
    with core.db() as con:
        return con.execute("SELECT COUNT(*) FROM faces WHERE label IS NULL").fetchone()[0]


def set_label(face_id: int, label: str) -> None:
    with core.db() as con:
        con.execute("UPDATE faces SET label=? WHERE id=?", (label.strip(), face_id))


def label_options() -> list[str]:
    """Names offered on the labeling page: every account plus everyone already labeled."""
    return sorted(set(core.users()) | set(people()), key=str.lower)


def people() -> list[str]:
    with core.db() as con:
        rows = con.execute(
            "SELECT DISTINCT label FROM faces WHERE label IS NOT NULL AND label NOT IN (?, ?) ORDER BY label",
            (IGNORE, STRANGER),
        )
        return [r[0] for r in rows]


def crop(face_id: int) -> Path:
    """Cached crop of one face (with margin) for the labeling page."""
    with core.db() as con:
        row = con.execute("SELECT path, bbox FROM faces WHERE id=?", (face_id,)).fetchone()
    if not row:
        raise FileNotFoundError(f"no face {face_id}")
    out = core.CACHE / "img" / f"face-{face_id}.jpg"
    if not out.exists():
        x1, y1, x2, y2 = (int(v) for v in row["bbox"].split(","))
        pad = round(0.25 * max(x2 - x1, y2 - y1))
        with Image.open(core.resolve(row["path"])) as img:
            img = ImageOps.exif_transpose(img)  # bbox coords are in transposed space, same as detection
            face = img.crop((max(0, x1 - pad), max(0, y1 - pad), min(img.width, x2 + pad), min(img.height, y2 + pad)))
            face.thumbnail((200, 200))
            face.convert("RGB").save(out, "JPEG", quality=85)
    return out
