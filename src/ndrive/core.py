"""Storage, users, likes and the photo index.

Layout under HOME (default ./storage):
    data/<user>/...   the shared tree, served over WebDAV; first path segment = owner
    cache/            rebuildable: sqlite index + jpeg renditions; `rescan` recreates it
    trash/<stamp>/    soft-deleted files, dropped by `ndrive purge-trash`
"""

import hashlib
import hmac
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import imagehash
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()
log = logging.getLogger("ndrive")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".3gp", ".avi"}
MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS
PHASH_DUP_DISTANCE = 6  # hamming distance on the 64-bit phash; bursts may trip this — we warn, never block
EXIF_DT_ORIGINAL, EXIF_DT, EXIF_DT_DIGITIZED, EXIF_IFD = 36867, 306, 36868, 0x8769
THUMB_SIDE = 320
VIEW_SIDE = 2048
STREAM_MAX_HEIGHT = 1080  # originals within these bounds stream as-is; anything bigger gets a web rendition
STREAM_MAX_BITRATE = 8_000_000

HOME: Path = Path()
DATA: Path = Path()
CACHE: Path = Path()
TRASH: Path = Path()
_DB: Path = Path()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
    username TEXT PRIMARY KEY, salt BLOB NOT NULL, pw_hash BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS photos(
    path TEXT PRIMARY KEY, owner TEXT NOT NULL, taken_at TEXT NOT NULL, uploaded_at TEXT NOT NULL,
    width INTEGER, height INTEGER, size INTEGER NOT NULL, mtime REAL NOT NULL, phash TEXT, dup_of TEXT);
CREATE TABLE IF NOT EXISTS likes(
    username TEXT NOT NULL, path TEXT NOT NULL, PRIMARY KEY(username, path));
CREATE TABLE IF NOT EXISTS faces(
    id INTEGER PRIMARY KEY, path TEXT NOT NULL, bbox TEXT NOT NULL, embedding BLOB NOT NULL, label TEXT,
    UNIQUE(path, bbox));
CREATE TABLE IF NOT EXISTS face_scan(path TEXT PRIMARY KEY, mtime REAL NOT NULL);
"""


def configure(home: str | Path) -> None:
    global HOME, DATA, CACHE, TRASH, _DB
    HOME = Path(home).resolve()
    DATA, CACHE, TRASH = HOME / "data", HOME / "cache", HOME / "trash"
    _DB = CACHE / "ndrive.db"
    for d in (DATA, CACHE / "img", CACHE / "vid", TRASH):
        d.mkdir(parents=True, exist_ok=True)
    with db() as con:
        con.executescript(_SCHEMA)
    verify_user.cache_clear()


@contextmanager
def db():
    con = sqlite3.connect(_DB)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=5000")
    try:
        yield con
        con.commit()
    finally:
        con.close()


# --- users & ownership -----------------------------------------------------


def _hash(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)


def add_user(username: str, password: str) -> None:
    if not username.replace("-", "").isalnum() or username.startswith("."):
        raise ValueError("username must be alphanumeric — it becomes the folder name")
    existing = canonical_user(username)
    if existing is not None and existing != username:
        raise ValueError(f"user already exists as {existing!r} (usernames are case-insensitive)")
    salt = os.urandom(16)
    with db() as con:
        con.execute("INSERT OR REPLACE INTO users VALUES(?,?,?)", (username, salt, _hash(password, salt)))
    (DATA / username).mkdir(exist_ok=True)
    verify_user.cache_clear()


@lru_cache(
    maxsize=256
)  # ponytail: caches plaintext creds in-process so scrypt (~50ms) runs once per user, not per request
def verify_user(username: str, password: str) -> str | None:
    """Case-insensitive login; returns the canonical (stored) username on success, else None."""
    with db() as con:
        row = con.execute(
            "SELECT username, salt, pw_hash FROM users WHERE username=? COLLATE NOCASE", (username,)
        ).fetchone()
    if row is None or not hmac.compare_digest(_hash(password, row["salt"]), row["pw_hash"]):
        return None
    return row["username"]


def canonical_user(username: str) -> str | None:
    with db() as con:
        row = con.execute("SELECT username FROM users WHERE username=? COLLATE NOCASE", (username,)).fetchone()
    return row[0] if row else None


def rename_user(old: str, new: str) -> None:
    """Rename an account and its folder; photos, likes and face data follow."""
    if not new.replace("-", "").isalnum() or new.startswith("."):
        raise ValueError("username must be alphanumeric — it becomes the folder name")
    if not (DATA / old).is_dir():
        raise ValueError(f"no such user: {old}")
    if (DATA / new).exists():
        raise ValueError(f"{new} already exists")
    (DATA / old).rename(DATA / new)
    with db() as con:
        con.execute("UPDATE users SET username=? WHERE username=?", (new, old))
        con.execute("UPDATE likes SET username=? WHERE username=?", (new, old))
    rename_paths(old, new)
    verify_user.cache_clear()


def users() -> list[str]:
    return sorted(d.name for d in DATA.iterdir() if d.is_dir())


def photo_counts() -> dict[str, int]:
    """How many pictures each account has — accounts with none are worth flagging in the filter."""
    with db() as con:
        return {r["owner"]: r["n"] for r in con.execute("SELECT owner, COUNT(*) n FROM photos GROUP BY owner")}


def can_write(username: str, rel_path: str) -> bool:
    """The entire permission model: you may only touch paths under your own top folder."""
    parts = [p for p in rel_path.split("/") if p]
    return bool(parts) and parts[0] == username and not any(p in {".", ".."} for p in parts)


def resolve(rel: str) -> Path:
    p = (DATA / rel).resolve()
    if not p.is_relative_to(DATA):
        raise ValueError(f"path escapes storage: {rel}")
    return p


# --- photo index -----------------------------------------------------------


def _taken_at(img: Image.Image, mtime: float) -> str:
    exif = img.getexif()
    ifd = exif.get_ifd(EXIF_IFD)
    raw = (
        ifd.get(EXIF_DT_ORIGINAL)
        or exif.get(EXIF_DT_ORIGINAL)
        or exif.get(EXIF_DT)
        or ifd.get(EXIF_DT_DIGITIZED)
        or exif.get(EXIF_DT_DIGITIZED)
    )
    if raw:
        try:
            return datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S").isoformat(sep=" ")
        except ValueError:
            pass  # malformed EXIF (WhatsApp etc.) — fall back to mtime
    return datetime.fromtimestamp(mtime).isoformat(sep=" ", timespec="seconds")


def index_file(rel: str) -> list[str]:
    """(Re)index one file; returns paths of near-duplicate existing photos."""
    abs_ = resolve(rel)
    ext = abs_.suffix.lower()
    if ext not in MEDIA_EXTS or not abs_.is_file():
        return []
    st = abs_.stat()
    width = height = phash = None
    taken = datetime.fromtimestamp(st.st_mtime).isoformat(sep=" ", timespec="seconds")
    thumb = _rendition_path(rel, st.st_mtime, THUMB_SIDE)
    try:
        if ext in VIDEO_EXTS:
            meta = _ffprobe(abs_)
            taken = _video_taken(meta) or taken
            width, height = _video_dims(meta)
            if not thumb.exists():
                _grab_frame(abs_, thumb, THUMB_SIDE)
        else:
            with Image.open(abs_) as img:
                width, height = img.size
                taken = _taken_at(img, st.st_mtime)
                phash = str(imagehash.phash(img))
                # we paid for the full decode (HEICs are expensive) — write the gallery thumb now, for free
                if not thumb.exists():
                    _write_rendition(img, thumb, THUMB_SIDE)
    except Exception as exc:  # noqa: BLE001 — corrupt or half-uploaded: keep it listed, unhashed
        log.warning(f"cannot read media {rel}: {exc}")
    dups = _near_duplicates(phash, rel)
    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    with db() as con:
        con.execute(
            "INSERT OR REPLACE INTO photos VALUES(?,?,?, COALESCE((SELECT uploaded_at FROM photos WHERE path=?), ?),"
            " ?,?,?,?,?,?)",
            (
                rel,
                rel.split("/")[0],
                taken,
                rel,
                now,
                width,
                height,
                st.st_size,
                st.st_mtime,
                phash,
                dups[0] if dups else None,
            ),
        )
    if ext in VIDEO_EXTS:
        transcode_async(rel)
    return dups


def _ffprobe(abs_: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(abs_)],
        capture_output=True,
        check=True,
        timeout=60,
    )
    return json.loads(out.stdout)


def _video_taken(meta: dict) -> str | None:
    # ponytail: no THM-sidecar fallback for 2000s-era camera AVIs — phone clips carry creation_time; rest gets mtime
    tag_sets = [meta.get("format", {}).get("tags", {})] + [s.get("tags", {}) for s in meta.get("streams", [])]
    for tags in tag_sets:
        for key, value in tags.items():
            if key.lower() == "creation_time":
                try:
                    dt = datetime.fromisoformat(str(value))
                except ValueError:
                    continue
                return dt.astimezone().replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")
    return None


def _video_dims(meta: dict) -> tuple[int | None, int | None]:
    for stream in meta.get("streams", []):
        if stream.get("codec_type") == "video":
            return stream.get("width"), stream.get("height")
    return None, None


_transcode_lock = threading.Lock()  # ponytail: one ffmpeg at a time — this box also runs the family's mail


def _video_rendition_path(rel: str, mtime: float) -> Path:
    key = hashlib.sha1(f"{rel}:{mtime}:stream".encode()).hexdigest()
    return CACHE / "vid" / f"{key}.mp4"


def stream_source(rel: str) -> Path:
    """What the lightbox should play: the web rendition when it exists, else the original."""
    abs_ = resolve(rel)
    if abs_.suffix.lower() in VIDEO_EXTS:
        rendition = _video_rendition_path(rel, abs_.stat().st_mtime)
        if rendition.exists():
            return rendition
    return abs_


def transcode_async(rel: str) -> None:
    threading.Thread(target=_locked_transcode, args=(rel,), daemon=True, name="ndrive-transcode").start()


def _locked_transcode(rel: str) -> None:
    with _transcode_lock:
        try:
            _maybe_transcode(rel)
        except Exception as exc:  # noqa: BLE001 — a broken clip must not stop the sweep
            log.warning(f"transcode failed for {rel}: {exc}")


def _maybe_transcode(rel: str) -> None:
    """Phone originals (4K/HEVC, 50-100 Mbit) choke browsers and home downlinks; YouTube-grade
    streaming needs a 1080p H.264 rendition. Originals that already fit stream untouched."""
    abs_ = resolve(rel)
    if not abs_.is_file():
        return
    out = _video_rendition_path(rel, abs_.stat().st_mtime)
    if out.exists():
        return
    meta = _ffprobe(abs_)
    stream = next((s for s in meta.get("streams", []) if s.get("codec_type") == "video"), {})
    bitrate = int(meta.get("format", {}).get("bit_rate") or 0)
    if (
        stream.get("codec_name") == "h264"
        and (stream.get("height") or 0) <= STREAM_MAX_HEIGHT
        and 0 < bitrate <= STREAM_MAX_BITRATE
    ):
        return
    log.info(f"transcoding {rel} for streaming")
    tmp = out.parent / (out.stem + ".part.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(abs_),
            "-vf",
            "scale=1920:1920:force_original_aspect_ratio=decrease:force_divisible_by=2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            "-y",
            str(tmp),
        ],
        capture_output=True,
        check=True,
        timeout=7200,
    )
    tmp.rename(out)  # atomic: never serve a half-written rendition


def transcode_all() -> None:
    """Catch-up sweep for videos without a streaming rendition (new uploads enqueue themselves)."""
    with db() as con:
        rels = [r["path"] for r in con.execute("SELECT path FROM photos ORDER BY path")]
    for rel in rels:
        if Path(rel).suffix.lower() in VIDEO_EXTS:
            _locked_transcode(rel)


def _grab_frame(src: Path, out: Path, max_side: int) -> None:
    for seek in ("1", "0"):  # 1s in for a representative frame; retry at 0 for clips shorter than that
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-ss",
                    seek,
                    "-i",
                    str(src),
                    "-frames:v",
                    "1",
                    "-vf",
                    f"scale='min({max_side},iw)':-2",
                    "-y",
                    str(out),
                ],
                capture_output=True,
                check=True,
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError):
            out.unlink(missing_ok=True)
            continue
        if out.exists() and out.stat().st_size:
            return
        out.unlink(missing_ok=True)
    raise RuntimeError(f"ffmpeg could not extract a frame from {src.name}")


def _near_duplicates(phash: str | None, rel: str) -> list[str]:
    if not phash:
        return []
    target = imagehash.hex_to_hash(phash)
    with db() as con:
        rows = con.execute("SELECT path, phash FROM photos WHERE phash IS NOT NULL AND path != ?", (rel,)).fetchall()
    # ponytail: linear scan per upload; a BK-tree if the library ever outgrows a family holiday
    return [r["path"] for r in rows if target - imagehash.hex_to_hash(r["phash"]) <= PHASH_DUP_DISTANCE]


def scan_all() -> None:
    """Reconcile the index with disk (files added/changed/removed outside our hooks)."""
    with db() as con:
        known = {r["path"]: (r["mtime"], r["size"]) for r in con.execute("SELECT path, mtime, size FROM photos")}
    found = set()
    for p in sorted(DATA.rglob("*")):
        if p.is_file() and p.suffix.lower() in MEDIA_EXTS:
            rel = p.relative_to(DATA).as_posix()
            found.add(rel)
            st = p.stat()
            if known.get(rel) != (st.st_mtime, st.st_size):
                index_file(rel)
    for gone in known.keys() - found:
        _forget(gone)
    log.info(f"scan done: {len(found)} pictures indexed")


def index_tree(rel: str) -> None:
    abs_ = resolve(rel)
    if abs_.is_dir():
        for p in abs_.rglob("*"):
            if p.is_file():
                index_file(p.relative_to(DATA).as_posix())
    else:
        index_file(rel)


def _forget(rel: str) -> None:
    with db() as con:
        for table in ("likes", "photos", "faces", "face_scan"):
            con.execute(f"DELETE FROM {table} WHERE path=?", (rel,))
        con.execute("UPDATE photos SET dup_of=NULL WHERE dup_of=?", (rel,))


# --- trash -----------------------------------------------------------------


def move_to_trash(rel: str) -> None:
    src = resolve(rel)
    dest = TRASH / datetime.now().strftime("%Y%m%d-%H%M%S") / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(src, dest)
    if "/" not in rel:  # someone trashed their whole root folder: keep an empty one so the account still works
        src.mkdir()
    with db() as con:
        for table in ("likes", "photos", "faces", "face_scan"):
            con.execute(f"DELETE FROM {table} WHERE path=? OR path LIKE ?", (rel, rel + "/%"))
        con.execute("UPDATE photos SET dup_of=NULL WHERE dup_of=? OR dup_of LIKE ?", (rel, rel + "/%"))


def purge_trash(days: int = 30) -> int:
    cutoff = datetime.now() - timedelta(days=days)
    removed = 0
    for stamp_dir in TRASH.iterdir():
        if datetime.fromtimestamp(stamp_dir.stat().st_mtime) < cutoff:
            shutil.rmtree(stamp_dir)
            removed += 1
    return removed


def rename_paths(old: str, new: str) -> None:
    """After a WebDAV MOVE: repoint index + likes; content unchanged so no re-hash."""
    with db() as con:
        rows = con.execute("SELECT path FROM photos WHERE path=? OR path LIKE ?", (old, old + "/%")).fetchall()
        for r in rows:
            np = new + r["path"][len(old) :]
            con.execute("DELETE FROM photos WHERE path=?", (np,))  # MOVE with Overwrite
            con.execute("UPDATE photos SET path=?, owner=? WHERE path=?", (np, np.split("/")[0], r["path"]))
            for table in ("likes", "faces", "face_scan"):
                con.execute(f"DELETE FROM {table} WHERE path=?", (np,))
                con.execute(f"UPDATE {table} SET path=? WHERE path=?", (np, r["path"]))
            con.execute("UPDATE photos SET dup_of=? WHERE dup_of=?", (np, r["path"]))


# --- likes & listing -------------------------------------------------------


def toggle_like(username: str, rel: str) -> tuple[bool, int]:
    with db() as con:
        removed = con.execute("DELETE FROM likes WHERE username=? AND path=?", (username, rel)).rowcount
        if not removed:
            con.execute("INSERT INTO likes VALUES(?,?)", (username, rel))
        count = con.execute("SELECT COUNT(*) FROM likes WHERE path=?", (rel,)).fetchone()[0]
    return not removed, count


def clear_dup(rel: str) -> None:
    """'Keep both' on the duplicates page: the pair was a legitimate near-match."""
    with db() as con:
        con.execute("UPDATE photos SET dup_of=NULL WHERE path=?", (rel,))


def pair_owners(rel: str) -> set[str]:
    """The owners involved in a duplicate pair — the only people allowed to resolve it."""
    with db() as con:
        row = con.execute(
            "SELECT p.owner AS a, q.owner AS b FROM photos p LEFT JOIN photos q ON q.path = p.dup_of WHERE p.path=?",
            (rel,),
        ).fetchone()
    return {v for v in (row["a"], row["b"]) if v} if row else set()


def list_photos(
    me: str,
    owner: str | None = None,
    sort: str = "asc",
    person: str | None = None,
    liked_only: bool = False,
) -> list[dict]:
    conditions = []
    if owner:
        conditions.append("p.owner = :owner")
    if person:
        conditions.append("p.path IN (SELECT path FROM faces WHERE label = :person)")
    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    having = "HAVING COUNT(l.username) > 0" if liked_only else ""
    order = {
        "asc": "p.taken_at ASC, p.path ASC",
        "owner": "p.owner ASC, p.taken_at DESC, p.path DESC",  # "sort by person" until face data exists
    }.get(sort, "p.taken_at DESC, p.path DESC")
    sql = f"""
        SELECT p.path, p.owner, p.taken_at, p.dup_of,
               COUNT(l.username) AS likes,
               COALESCE(MAX(CASE WHEN l.username = :me THEN 1 END), 0) AS liked
        FROM photos p LEFT JOIN likes l ON l.path = p.path
        {where}
        GROUP BY p.path {having} ORDER BY {order}"""
    with db() as con:
        rows = [dict(r) for r in con.execute(sql, {"me": me, "owner": owner, "person": person})]
    for r in rows:
        r["video"] = Path(r["path"]).suffix.lower() in VIDEO_EXTS
    return rows


def duplicate_pairs(user: str | None = None) -> list[tuple[dict, dict]]:
    """Flagged near-duplicates with their partner; scoped to pairs `user` can resolve when given."""
    with db() as con:
        rows = [dict(r) for r in con.execute("SELECT * FROM photos ORDER BY taken_at DESC, path DESC")]
    by_path = {r["path"]: r for r in rows}
    pairs = [(r, by_path[r["dup_of"]]) for r in rows if r["dup_of"] in by_path]
    if user:
        pairs = [(a, b) for a, b in pairs if user in (a["owner"], b["owner"])]
    return pairs


# --- renditions ------------------------------------------------------------


def _rendition_path(rel: str, mtime: float, max_side: int) -> Path:
    key = hashlib.sha1(f"{rel}:{mtime}:{max_side}".encode()).hexdigest()
    return CACHE / "img" / f"{key}.jpg"


def _write_rendition(img: Image.Image, out: Path, max_side: int) -> None:
    img = ImageOps.exif_transpose(img)
    img.thumbnail((max_side, max_side))
    img.convert("RGB").save(out, "JPEG", quality=85)


def rendition(rel: str, max_side: int) -> Path:
    """Cached JPEG rendition (thumb or view size); this is also how HEIC becomes browser-viewable.

    Gallery thumbs are normally pre-written by index_file; this decodes on demand for
    view-size requests and cache misses.
    """
    abs_ = resolve(rel)
    out = _rendition_path(rel, abs_.stat().st_mtime, max_side)
    if not out.exists():
        if abs_.suffix.lower() in VIDEO_EXTS:
            _grab_frame(abs_, out, max_side)
        else:
            with Image.open(abs_) as img:
                _write_rendition(img, out, max_side)
    return out


def unique_dest(parent_rel: str, filename: str) -> str:
    name = Path(filename).name
    if name in {"", ".", ".."}:
        name = "upload"
    stem, suffix = Path(name).stem, Path(name).suffix
    cand, i = name, 0
    while (DATA / parent_rel / cand).exists():
        i += 1
        cand = f"{stem}-{i}{suffix}"
    return f"{parent_rel}/{cand}"


def subfolders(owner: str) -> list[str]:
    root = DATA / owner
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_dir())
