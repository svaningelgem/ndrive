"""Face detection, embeddings and labeling.

There is no model training: pretrained insightface (buffalo_l) embeddings, a
person = the mean of their labeled embeddings, a suggestion = cosine similarity
to those means. Every confirmed label immediately improves future suggestions.
"""

import logging
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
SUGGEST_MIN_SIM = 0.25  # cosine similarity below this → no pre-selected guess
DETECT_MAX_SIDE = 1600
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
        analyzer.prepare(ctx_id=0, det_size=(640, 640))
        _analyzer = analyzer
    return _analyzer


def scan_faces() -> None:
    """Detect + embed faces for photos not scanned yet (or changed since)."""
    if FaceAnalysis is None:
        log.warning("insightface not installed — face scanning disabled")
        return
    with core.db() as con:
        done = {r["path"]: r["mtime"] for r in con.execute("SELECT path, mtime FROM face_scan")}
        todo = [
            (r["path"], r["mtime"])
            for r in con.execute("SELECT path, mtime FROM photos")
            if Path(r["path"]).suffix.lower() in core.IMAGE_EXTS and done.get(r["path"]) != r["mtime"]
        ]
    for rel, mtime in todo:
        try:
            _scan_one(rel, mtime)
        except Exception as exc:  # noqa: BLE001 — one unreadable photo must not stop the sweep
            log.warning(f"face scan failed for {rel}: {exc}")
    if todo:
        log.info(f"face scan done: {len(todo)} new photo(s)")


def _scan_one(rel: str, mtime: float) -> None:
    abs_ = core.resolve(rel)
    if not abs_.is_file():
        return
    with Image.open(abs_) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        scale = max(img.size) / DETECT_MAX_SIDE
        if scale > 1:
            img = img.resize((round(img.width / scale), round(img.height / scale)))
        else:
            scale = 1.0
        arr = np.asarray(img)[:, :, ::-1]  # insightface wants BGR
    found = _get_analyzer().get(arr)
    with core.db() as con:
        con.execute("DELETE FROM faces WHERE path=?", (rel,))
        for f in found:
            bbox = ",".join(str(round(float(v) * scale)) for v in f.bbox)  # back to original-pixel coords
            con.execute(
                "INSERT OR REPLACE INTO faces(path, bbox, embedding, label) VALUES(?,?,?,NULL)",
                (rel, bbox, f.normed_embedding.astype(np.float32).tobytes()),
            )
        con.execute("INSERT OR REPLACE INTO face_scan VALUES(?,?)", (rel, mtime))


# --- labeling --------------------------------------------------------------


def _centroids() -> dict[str, np.ndarray]:
    with core.db() as con:
        rows = con.execute(
            "SELECT label, embedding FROM faces WHERE label IS NOT NULL AND label != ?", (IGNORE,)
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
    """Unlabeled faces with the model's best guess (nearest labeled person by cosine)."""
    centroids = _centroids()
    with core.db() as con:
        rows = con.execute("SELECT id, path, embedding FROM faces WHERE label IS NULL LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        vec = np.frombuffer(r["embedding"], dtype=np.float32)
        best, sim = None, 0.0
        for name, centroid in centroids.items():
            s = float(vec @ centroid)
            if s > sim:
                best, sim = name, s
        out.append({"id": r["id"], "path": r["path"], "suggest": best if sim >= SUGGEST_MIN_SIM else None})
    return out


def unlabeled_count() -> int:
    with core.db() as con:
        return con.execute("SELECT COUNT(*) FROM faces WHERE label IS NULL").fetchone()[0]


def set_label(face_id: int, label: str) -> None:
    with core.db() as con:
        con.execute("UPDATE faces SET label=? WHERE id=?", (label.strip(), face_id))


def people() -> list[str]:
    with core.db() as con:
        rows = con.execute(
            "SELECT DISTINCT label FROM faces WHERE label IS NOT NULL AND label != ? ORDER BY label", (IGNORE,)
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
