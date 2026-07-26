import io
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ndrive import core, faces
from ndrive.app import create_app

ALICE = ("alice", "pw-alice")
BOB = ("bob", "pw-bob")


def jpeg_bytes(color: tuple[int, int, int], taken: str | None = None) -> bytes:
    img = Image.new("RGB", (64, 48), color)
    exif = Image.Exif()
    if taken:
        exif[306] = taken  # DateTime
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    return buf.getvalue()


def gradient_jpeg(size: tuple[int, int]) -> bytes:
    buf = io.BytesIO()
    Image.linear_gradient("L").resize(size).save(buf, "JPEG")
    return buf.getvalue()


def mandel_jpeg() -> bytes:
    # synthetic gradients/checkers all phash to the same degenerate value; the mandelbrot doesn't
    buf = io.BytesIO()
    Image.effect_mandelbrot((128, 96), (-2.0, -1.5, 1.0, 1.5), 40).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path: Path, mocker) -> TestClient:
    core.configure(tmp_path)
    core.add_user(*ALICE)
    core.add_user(*BOB)
    # no real background work by default: those threads would outlive tmp_path and hit a deleted database
    mocker.patch.object(faces, "scan_async")
    mocker.patch.object(core, "scan_all")
    mocker.patch.object(core, "transcode_all")
    return TestClient(create_app(tmp_path))


def test_verify_user(tmp_path: Path) -> None:
    core.configure(tmp_path)
    core.add_user("alice", "secret")
    assert core.verify_user("alice", "secret")
    assert not core.verify_user("alice", "wrong")
    assert not core.verify_user("ghost", "secret")


@pytest.mark.parametrize(
    ("user", "path", "ok"),
    [
        ("alice", "alice/a.jpg", True),
        ("alice", "alice/sub/b.jpg", True),
        ("alice", "bob/a.jpg", False),
        ("alice", "", False),
        ("alice", "alice/../bob/x.jpg", False),
    ],
)
def test_can_write(user: str, path: str, ok: bool) -> None:
    assert core.can_write(user, path) is ok


def test_case_insensitive_login(client: TestClient) -> None:
    assert core.verify_user("ALICE", "pw-alice") == "alice"  # canonical case comes back
    assert core.verify_user("Alice", "wrong") is None
    with pytest.raises(ValueError, match="case-insensitive"):
        core.add_user("ALICE", "whatever")

    r = client.post(
        "/api/upload", auth=("ALICE", "pw-alice"), files=[("files", ("c.jpg", jpeg_bytes((11, 12, 13)), "image/jpeg"))]
    )
    assert r.status_code == 200
    with core.db() as con:  # landed in the canonical folder
        assert con.execute("SELECT COUNT(*) FROM photos WHERE path='alice/c.jpg'").fetchone()[0] == 1
    assert client.put("/dav/alice/d.jpg", auth=("aLiCe", "pw-alice"), content=jpeg_bytes((14, 15, 16))).status_code in (
        200,
        201,
        204,
    )


def test_upload_gallery_and_exif_date(client: TestClient) -> None:
    r = client.post(
        "/api/upload",
        auth=ALICE,
        files=[("files", ("a.jpg", jpeg_bytes((200, 30, 30), taken="2026:07:01 10:00:00"), "image/jpeg"))],
    )
    assert r.status_code == 200
    assert r.json()["warnings"] == []
    with core.db() as con:
        row = con.execute("SELECT * FROM photos").fetchone()
    assert row["path"] == "alice/a.jpg"
    assert row["owner"] == "alice"
    assert row["taken_at"] == "2026-07-01 10:00:00"
    assert "alice/a.jpg" in client.get("/?owner=*", auth=BOB).text  # everyone-view shows all pictures
    assert "alice/a.jpg" not in client.get("/", auth=BOB).text  # default view: your own drive


def test_duplicate_warning(client: TestClient) -> None:
    client.post("/api/upload", auth=ALICE, files=[("files", ("a.jpg", gradient_jpeg((128, 96)), "image/jpeg"))])
    r = client.post("/api/upload", auth=ALICE, files=[("files", ("b.jpg", gradient_jpeg((100, 75)), "image/jpeg"))])
    assert "duplicate" in r.json()["warnings"][0]
    with core.db() as con:
        assert con.execute("SELECT dup_of FROM photos WHERE path='alice/b.jpg'").fetchone()[0] == "alice/a.jpg"


def test_duplicates_review_page(client: TestClient) -> None:
    client.post("/api/upload", auth=ALICE, files=[("files", ("a.jpg", gradient_jpeg((128, 96)), "image/jpeg"))])
    client.post("/api/upload", auth=ALICE, files=[("files", ("b.jpg", gradient_jpeg((100, 75)), "image/jpeg"))])
    assert len(core.duplicate_pairs()) == 1

    html = client.get("/duplicates", auth=ALICE).text
    assert "alice/a.jpg" in html
    assert "alice/b.jpg" in html
    assert "delete this one" in html
    assert "alice/a.jpg" not in client.get("/duplicates", auth=BOB).text  # pairs you can't resolve aren't shown

    # deleting the flagged copy's *partner* clears the dangling dup_of flag
    client.post("/api/delete", auth=ALICE, json={"paths": ["alice/a.jpg"]})
    assert core.duplicate_pairs() == []
    with core.db() as con:
        assert con.execute("SELECT dup_of FROM photos WHERE path='alice/b.jpg'").fetchone()[0] is None


def test_keep_both_resolves_pair(client: TestClient) -> None:
    client.post("/api/upload", auth=ALICE, files=[("files", ("a.jpg", gradient_jpeg((128, 96)), "image/jpeg"))])
    client.post("/api/upload", auth=BOB, files=[("files", ("b.jpg", gradient_jpeg((100, 75)), "image/jpeg"))])
    assert len(core.duplicate_pairs()) == 1  # duplicates are detected across accounts

    r = client.post("/api/keep-both", auth=BOB, json={"path": "bob/b.jpg"})
    assert r.status_code == 200
    assert core.duplicate_pairs() == []
    with core.db() as con:  # both photos still exist
        assert con.execute("SELECT COUNT(*) FROM photos").fetchone()[0] == 2


def test_keep_both_is_for_involved_owners_only(client: TestClient) -> None:
    client.post("/api/upload", auth=ALICE, files=[("files", ("a.jpg", gradient_jpeg((128, 96)), "image/jpeg"))])
    client.post("/api/upload", auth=ALICE, files=[("files", ("b.jpg", gradient_jpeg((100, 75)), "image/jpeg"))])
    assert client.post("/api/keep-both", auth=BOB, json={"path": "alice/b.jpg"}).status_code == 403
    assert core.duplicate_pairs("bob") == []  # not bob's to see
    assert len(core.duplicate_pairs("alice")) == 1
    assert len(core.duplicate_pairs()) == 1  # untouched
    assert client.post("/api/keep-both", auth=ALICE, json={"path": "alice/b.jpg"}).status_code == 200
    assert core.duplicate_pairs() == []


def test_liked_filter(client: TestClient) -> None:
    client.post("/api/upload", auth=ALICE, files=[("files", ("a.jpg", jpeg_bytes((1, 2, 3)), "image/jpeg"))])
    client.post("/api/upload", auth=ALICE, files=[("files", ("b.jpg", jpeg_bytes((4, 5, 6)), "image/jpeg"))])
    client.post("/api/like", auth=BOB, json={"path": "alice/a.jpg"})
    assert [p["path"] for p in core.list_photos("bob", liked_only=True)] == ["alice/a.jpg"]
    html = client.get("/?liked=1&owner=*", auth=BOB).text
    assert "alice/a.jpg" in html
    assert "alice/b.jpg" not in html


def test_default_sort_is_oldest_first(client: TestClient) -> None:
    old = jpeg_bytes((1, 1, 1), taken="2026:07:01 08:00:00")
    new = jpeg_bytes((2, 2, 2), taken="2026:07:09 08:00:00")
    client.post("/api/upload", auth=ALICE, files=[("files", ("new.jpg", new, "image/jpeg"))])
    client.post("/api/upload", auth=ALICE, files=[("files", ("old.jpg", old, "image/jpeg"))])
    html = client.get("/", auth=ALICE).text
    assert html.index("alice/old.jpg") < html.index("alice/new.jpg")


def test_rename_user(client: TestClient) -> None:
    client.post("/api/upload", auth=ALICE, files=[("files", ("a.jpg", jpeg_bytes((4, 5, 6)), "image/jpeg"))])
    client.post("/api/like", auth=BOB, json={"path": "alice/a.jpg"})
    _insert_face("alice/a.jpg", "1,1,20,20", _vec(2), label="mama")

    core.rename_user("alice", "alicia")
    assert (core.DATA / "alicia/a.jpg").exists()
    assert core.verify_user("alicia", "pw-alice")
    assert not core.verify_user("alice", "pw-alice")
    with core.db() as con:
        assert con.execute("SELECT owner FROM photos WHERE path='alicia/a.jpg'").fetchone()[0] == "alicia"
        assert con.execute("SELECT username FROM likes WHERE path='alicia/a.jpg'").fetchone()[0] == "bob"
        assert con.execute("SELECT label FROM faces WHERE path='alicia/a.jpg'").fetchone()[0] == "mama"


def test_distinct_photos_no_warning(client: TestClient) -> None:
    r1 = client.post("/api/upload", auth=ALICE, files=[("files", ("a.jpg", gradient_jpeg((128, 96)), "image/jpeg"))])
    r2 = client.post("/api/upload", auth=BOB, files=[("files", ("b.jpg", mandel_jpeg(), "image/jpeg"))])
    assert r1.json()["warnings"] == []
    assert r2.json()["warnings"] == []


def test_delete_ownership_and_trash(client: TestClient) -> None:
    client.post("/api/upload", auth=ALICE, files=[("files", ("a.jpg", jpeg_bytes((10, 200, 10)), "image/jpeg"))])
    r = client.post("/api/delete", auth=BOB, json={"paths": ["alice/a.jpg"]})
    assert r.json()["deleted"] == 0
    assert r.json()["skipped"] == 1
    assert "aren't yours" in r.json()["message"]
    assert (core.DATA / "alice/a.jpg").exists()

    r = client.post("/api/delete", auth=ALICE, json={"paths": ["alice/a.jpg"]})
    assert r.json()["deleted"] == 1
    assert not (core.DATA / "alice/a.jpg").exists()
    assert list(core.TRASH.rglob("a.jpg"))
    with core.db() as con:
        assert con.execute("SELECT COUNT(*) FROM photos").fetchone()[0] == 0


def test_like_toggle(client: TestClient) -> None:
    client.post("/api/upload", auth=ALICE, files=[("files", ("a.jpg", jpeg_bytes((1, 2, 3)), "image/jpeg"))])
    r = client.post("/api/like", auth=BOB, json={"path": "alice/a.jpg"})
    assert r.json() == {"liked": True, "count": 1}
    r = client.post("/api/like", auth=BOB, json={"path": "alice/a.jpg"})
    assert r.json() == {"liked": False, "count": 0}


def test_webdav_ownership(client: TestClient) -> None:
    body = jpeg_bytes((5, 5, 200))
    assert client.put("/dav/alice/x.jpg", auth=BOB, content=body).status_code == 403
    assert client.put("/dav/alice/x.jpg", auth=ALICE, content=body).status_code in (200, 201, 204)
    with core.db() as con:  # the PUT triggered indexing
        assert con.execute("SELECT owner FROM photos WHERE path='alice/x.jpg'").fetchone()[0] == "alice"
    assert client.get("/dav/alice/x.jpg", auth=BOB).status_code == 200  # read-all
    assert client.delete("/dav/alice/x.jpg", auth=BOB).status_code == 403
    assert client.delete("/dav/alice/x.jpg", auth=ALICE).status_code == 204  # soft delete
    assert not (core.DATA / "alice/x.jpg").exists()
    assert list(core.TRASH.rglob("x.jpg"))


def test_webdav_move_updates_index(client: TestClient) -> None:
    client.put("/dav/alice/x.jpg", auth=ALICE, content=jpeg_bytes((9, 9, 9)))
    r = client.request(
        "MOVE", "/dav/alice/x.jpg", auth=ALICE, headers={"Destination": "http://testserver/dav/alice/y.jpg"}
    )
    assert r.status_code in (201, 204)
    with core.db() as con:
        assert con.execute("SELECT path FROM photos").fetchone()[0] == "alice/y.jpg"

    client.put("/dav/alice/z.jpg", auth=ALICE, content=jpeg_bytes((9, 9, 9)))
    r = client.request(
        "MOVE", "/dav/alice/z.jpg", auth=ALICE, headers={"Destination": "http://testserver/dav/bob/z.jpg"}
    )
    assert r.status_code == 403  # cross-owner move refused


def test_mkdir_and_upload_to_folder(client: TestClient) -> None:
    r = client.post("/api/mkdir", auth=ALICE, json={"folder": "holiday"})
    assert r.status_code == 200
    assert (core.DATA / "alice/holiday").is_dir()
    assert client.post("/api/mkdir", auth=ALICE, json={"folder": "../bob"}).status_code == 400

    r = client.post(
        "/api/upload",
        auth=ALICE,
        data={"folder": "holiday"},
        files=[("files", ("a.jpg", jpeg_bytes((3, 2, 1)), "image/jpeg"))],
    )
    assert r.status_code == 200
    with core.db() as con:
        assert con.execute("SELECT owner FROM photos WHERE path='alice/holiday/a.jpg'").fetchone()[0] == "alice"


def test_sort_by_person(client: TestClient) -> None:
    late, early = jpeg_bytes((1, 1, 1), taken="2026:07:03 10:00:00"), jpeg_bytes((2, 2, 2), taken="2026:07:01 10:00:00")
    client.post("/api/upload", auth=BOB, files=[("files", ("b.jpg", late, "image/jpeg"))])
    client.post("/api/upload", auth=ALICE, files=[("files", ("a.jpg", early, "image/jpeg"))])
    assert [p["owner"] for p in core.list_photos("alice", sort="owner")] == ["alice", "bob"]
    assert [p["owner"] for p in core.list_photos("alice", sort="desc")] == ["bob", "alice"]


def test_taken_at_digitized_fallback(client: TestClient) -> None:
    img = Image.new("RGB", (32, 32), (7, 7, 7))
    exif = Image.Exif()
    exif[36868] = "2026:07:02 09:30:00"  # DateTimeDigitized only
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    (core.DATA / "alice/d.jpg").write_bytes(buf.getvalue())
    core.index_file("alice/d.jpg")
    with core.db() as con:
        assert (
            con.execute("SELECT taken_at FROM photos WHERE path='alice/d.jpg'").fetchone()[0] == "2026-07-02 09:30:00"
        )


def test_video_indexing_and_playback(client: TestClient) -> None:
    if not shutil.which("ffmpeg"):
        pytest.skip("no ffmpeg available")
    path = core.DATA / "alice/clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=64x48:rate=10",
            "-metadata",
            "creation_time=2026-07-05T12:00:00.000000Z",
            "-y",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    core.index_file("alice/clip.mp4")
    with core.db() as con:
        row = con.execute("SELECT * FROM photos WHERE path='alice/clip.mp4'").fetchone()
    expected = (datetime.fromisoformat("2026-07-05T12:00:00+00:00").astimezone().replace(tzinfo=None)).isoformat(
        sep=" ", timespec="seconds"
    )
    assert row["taken_at"] == expected
    assert row["width"] == 64
    assert row["phash"] is None
    with Image.open(core.rendition("alice/clip.mp4", core.THUMB_SIDE)) as frame:
        assert frame.format == "JPEG"
    html = client.get("/?owner=*", auth=BOB).text
    assert "alice/clip.mp4" in html
    assert "▶" in html
    r = client.get("/media/alice/clip.mp4", auth=BOB, headers={"Range": "bytes=0-99"})
    assert r.status_code == 206  # partial content — video seeking works


def test_video_stream_rendition(client: TestClient) -> None:
    if not shutil.which("ffmpeg"):
        pytest.skip("no ffmpeg available")
    # mpeg4 codec → outside the stream-as-is envelope → must get an h264 web rendition
    path = core.DATA / "alice/old.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=64x48:rate=10",
            "-c:v",
            "mpeg4",
            "-y",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    core.index_file("alice/old.mp4")
    core._locked_transcode("alice/old.mp4")
    src = core.stream_source("alice/old.mp4")
    assert src != core.resolve("alice/old.mp4")  # the rendition, not the original
    probe = core._ffprobe(src)
    assert probe["streams"][0]["codec_name"] == "h264"
    r = client.get("/stream/alice/old.mp4", auth=BOB, headers={"Range": "bytes=0-99"})
    assert r.status_code == 206

    # small h264 clip is already fine → original streams untouched
    path2 = core.DATA / "alice/ok.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=1:size=64x48:rate=10", "-y", str(path2)],
        check=True,
        capture_output=True,
    )
    core.index_file("alice/ok.mp4")
    core._locked_transcode("alice/ok.mp4")
    assert core.stream_source("alice/ok.mp4") == core.resolve("alice/ok.mp4")


def test_heic_rendition(client: TestClient) -> None:
    path = core.DATA / "alice/h.heic"
    try:
        Image.effect_mandelbrot((640, 480), (-2.0, -1.5, 1.0, 1.5), 40).convert("RGB").save(path)
    except Exception:  # noqa: BLE001 — libheif built without an HEVC encoder
        pytest.skip("no HEIC encoder available")
    core.index_file("alice/h.heic")
    with core.db() as con:
        assert con.execute("SELECT phash FROM photos WHERE path='alice/h.heic'").fetchone()[0]
    assert list((core.CACHE / "img").glob("*.jpg"))  # gallery thumb pre-written during indexing
    out = core.rendition("alice/h.heic", core.THUMB_SIDE)
    with Image.open(out) as thumb:
        assert thumb.format == "JPEG"
        assert max(thumb.size) <= core.THUMB_SIDE


def _vec_array(hot: int) -> np.ndarray:
    v = np.zeros(512, dtype=np.float32)
    v[hot] = 1.0
    return v


def _vec(hot: int) -> bytes:
    v = np.zeros(512, dtype=np.float32)
    v[hot] = 1.0
    return v.tobytes()


def _insert_face(path: str, bbox: str, embedding: bytes, label: str | None = None) -> int:
    with core.db() as con:
        cur = con.execute(
            "INSERT INTO faces(path, bbox, embedding, label) VALUES(?,?,?,?)", (path, bbox, embedding, label)
        )
        return cur.lastrowid


def test_face_labeling_flow(client: TestClient) -> None:
    client.post("/api/upload", auth=ALICE, files=[("files", ("a.jpg", jpeg_bytes((90, 60, 30)), "image/jpeg"))])
    _insert_face("alice/a.jpg", "1,1,20,20", _vec(0), label="mama")
    near = _insert_face("alice/a.jpg", "5,5,30,30", _vec(0))  # same direction as mama's centroid
    far = _insert_face("alice/a.jpg", "30,10,60,40", _vec(1))  # orthogonal — no guess

    guesses = {f["id"]: f["suggest"] for f in faces.unlabeled()}
    assert guesses == {near: "mama", far: None}

    html = client.get("/faces", auth=BOB).text
    assert f"/face/{near}" in html
    assert "selected>mama" in html  # the model's guess comes pre-selected
    assert ">alice<" in html  # account names are offered even before anyone labeled them
    assert ">bob<" in html

    crop = client.get(f"/face/{near}", auth=BOB)
    assert crop.status_code == 200
    with Image.open(io.BytesIO(crop.content)) as im:
        assert im.format == "JPEG"

    r = client.post("/api/face", auth=ALICE, json={"face_id": near, "label": "mama"})
    assert r.status_code == 200
    r = client.post("/api/face", auth=ALICE, json={"face_id": far, "label": "papa"})
    assert faces.people() == ["mama", "papa"]
    assert r.json()["remaining"] == 0

    # strangers are archived but never become suggestable people
    stranger = _insert_face("alice/a.jpg", "40,10,60,30", _vec(3))
    client.post("/api/face", auth=ALICE, json={"face_id": stranger, "label": faces.STRANGER})
    assert faces.people() == ["mama", "papa"]
    assert faces.unlabeled() == []

    # gallery filter by person
    assert "alice/a.jpg" in client.get("/?person=mama&owner=*", auth=BOB).text
    client.post("/api/upload", auth=BOB, files=[("files", ("b.jpg", mandel_jpeg(), "image/jpeg"))])
    photos = core.list_photos("bob", person="mama")
    assert [p["path"] for p in photos] == ["alice/a.jpg"]


def test_rescan_keeps_labels_and_drops_stale_crop(client: TestClient, mocker) -> None:
    """A re-scan re-detects the same faces; labels must survive it (matched by overlap)."""
    client.post("/api/upload", auth=ALICE, files=[("files", ("a.jpg", jpeg_bytes((10, 20, 30)), "image/jpeg"))])
    fake_model(mocker)
    faces.scan_faces()
    with core.db() as con:
        face_id = con.execute("SELECT id FROM faces WHERE path='alice/a.jpg'").fetchone()[0]
    faces.set_label(face_id, "oma")
    stale = core.CACHE / "img" / f"face-{face_id}.jpg"
    stale.write_bytes(b"old crop")

    with core.db() as con:  # force a re-scan of the same file
        con.execute("DELETE FROM face_scan")
    faces.scan_faces()

    with core.db() as con:
        rows = con.execute("SELECT id, label FROM faces WHERE path='alice/a.jpg'").fetchall()
    assert [r["label"] for r in rows] == ["oma"]  # label carried over to the re-detected face
    assert not stale.exists()  # cached crop of the old row is gone, so ids cannot show a wrong face


def test_suggestion_needs_confidence_and_margin(client: TestClient) -> None:
    client.post("/api/upload", auth=ALICE, files=[("files", ("a.jpg", jpeg_bytes((9, 9, 9)), "image/jpeg"))])
    _insert_face("alice/a.jpg", "1,1,20,20", _vec(0), label="mama")
    _insert_face("alice/a.jpg", "2,2,21,21", _vec(1), label="papa")

    weak = np.zeros(512, dtype=np.float32)  # 0.3 towards mama: above the old 0.25, below the real bar
    weak[0], weak[5] = 0.3, (1 - 0.3**2) ** 0.5
    weak_id = _insert_face("alice/a.jpg", "40,40,60,60", weak.tobytes())

    ambiguous = np.zeros(512, dtype=np.float32)  # equally close to both people
    ambiguous[0] = ambiguous[1] = 0.5**0.5
    amb_id = _insert_face("alice/a.jpg", "70,70,90,90", ambiguous.tobytes())

    guesses = {f["id"]: f["suggest"] for f in faces.unlabeled()}
    assert guesses[weak_id] is None  # not confident enough
    assert guesses[amb_id] is None  # confident but a coin flip between two people


def test_hard_pose_matches_its_own_kind_not_the_average(client: TestClient) -> None:
    """Sunglasses/profile shots look nothing like frontal ones; a per-person average would bury
    the single odd-looking example, so scoring must compare against individual labeled faces."""
    client.post("/api/upload", auth=ALICE, files=[("files", ("a.jpg", jpeg_bytes((8, 8, 8)), "image/jpeg"))])
    for i in range(5):  # five ordinary shots of mama
        _insert_face("alice/a.jpg", f"{i},1,{i + 20},20", _vec(0), label="mama")
    _insert_face("alice/a.jpg", "1,40,21,60", _vec(1), label="mama")  # one of her in sunglasses
    _insert_face("alice/a.jpg", "1,70,21,90", _vec(2), label="papa")

    same_sunglasses = _insert_face("alice/a.jpg", "40,40,60,60", _vec(1))
    guesses = {f["id"]: f["suggest"] for f in faces.unlabeled()}
    assert guesses[same_sunglasses] == "mama"

    # with an averaged face-per-person this scores ~0.20 and would have been left unrecognised
    known, names = faces._labeled()
    mama = known[[n == "mama" for n in names]]
    centroid = mama.mean(0) / np.linalg.norm(mama.mean(0))
    assert float(_vec_array(1) @ centroid) < faces.SUGGEST_MIN_SIM


def test_labeling_page_groups_by_suggested_person(client: TestClient) -> None:
    client.post("/api/upload", auth=ALICE, files=[("files", ("a.jpg", jpeg_bytes((9, 9, 9)), "image/jpeg"))])
    _insert_face("alice/a.jpg", "1,1,20,20", _vec(0), label="mama")
    _insert_face("alice/a.jpg", "2,2,21,21", _vec(1), label="papa")
    # two faces that look like mama, one like papa, one unrecognisable — deliberately interleaved
    m1 = _insert_face("alice/a.jpg", "10,10,30,30", _vec(0))
    p1 = _insert_face("alice/a.jpg", "31,10,50,30", _vec(1))
    m2 = _insert_face("alice/a.jpg", "51,10,70,30", _vec(0))
    unknown = _insert_face("alice/a.jpg", "71,10,90,30", _vec(7))

    groups = faces.grouped()
    assert [g["suggest"] for g in groups] == ["mama", "papa", None]  # named blocks first, unknowns last
    assert [f["id"] for f in groups[0]["faces"]] == [m1, m2]  # the two mamas arrive together
    assert [f["id"] for f in groups[1]["faces"]] == [p1]
    assert [f["id"] for f in groups[2]["faces"]] == [unknown]

    html = client.get("/faces", auth=BOB).text
    assert "✓ confirm all 2" in html  # one click confirms the whole block, each with its own dropdown
    assert html.index("probably <b>mama") < html.index("probably <b>papa")  # people come alphabetically


class FakeAnalyzer:
    """Stands in for insightface: a detector and a recogniser, same call shapes as the real ones."""

    def __init__(self) -> None:
        self.det_model = self
        self.models = {"recognition": self}
        self.detected_on: list[tuple[int, int]] = []
        self.embedded_on: list[tuple[int, int]] = []

    def detect(self, img: np.ndarray, max_num: int = 0, metric: str = "default"):
        self.detected_on.append(img.shape[:2])
        # 56px box: comfortably above MIN_FACE_PX even when the image is not downscaled
        return np.array([[2.0, 2.0, 58.0, 58.0, 0.99]], dtype=np.float32), np.zeros((1, 5, 2), dtype=np.float32)

    def get(self, img: np.ndarray, face) -> None:
        self.embedded_on.append(img.shape[:2])  # which image the crop came from is the whole point
        face.embedding = np.ones(512, dtype=np.float32)


def fake_model(mocker, inline: bool = False) -> FakeAnalyzer:
    """Pretend the ML stack is installed; optionally run sweeps inline instead of threaded."""
    analyzer = FakeAnalyzer()
    mocker.patch.object(faces, "FaceAnalysis", object())
    mocker.patch.object(faces, "_get_analyzer", return_value=analyzer)
    if inline:
        mocker.patch.object(faces, "scan_async", side_effect=faces.sweep)
    return analyzer


def test_upload_triggers_face_scan(client: TestClient, mocker) -> None:
    """Regression: face scanning used to run only at startup, so uploads were never scanned."""
    fake_model(mocker, inline=True)
    client.post("/api/upload", auth=ALICE, files=[("files", ("a.jpg", jpeg_bytes((10, 20, 30)), "image/jpeg"))])
    with core.db() as con:
        assert con.execute("SELECT COUNT(*) FROM faces WHERE path='alice/a.jpg'").fetchone()[0] == 1
    assert faces.pending_count() == 0

    client.put("/dav/alice/x.jpg", auth=ALICE, content=jpeg_bytes((40, 50, 60)))  # same for the drive
    with core.db() as con:
        assert con.execute("SELECT COUNT(*) FROM faces WHERE path='alice/x.jpg'").fetchone()[0] == 1


def test_scan_faces_with_mocked_model(client: TestClient, mocker) -> None:
    client.post("/api/upload", auth=ALICE, files=[("files", ("a.jpg", jpeg_bytes((10, 20, 30)), "image/jpeg"))])
    analyzer = fake_model(mocker)
    assert faces.pending_count() == 1

    faces.scan_faces()
    with core.db() as con:
        assert con.execute("SELECT COUNT(*) FROM faces").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM face_scan").fetchone()[0] == 1

    faces.scan_faces()  # unchanged photo → not scanned again
    assert len(analyzer.embedded_on) == 1


def test_embedding_uses_full_resolution_not_the_detection_copy(client: TestClient, mocker) -> None:
    """The bug that made big photos unrecognisable: the crop must come from the original pixels."""
    big = Image.effect_mandelbrot((4000, 3000), (-2.0, -1.5, 1.0, 1.5), 40).convert("RGB")
    buf = io.BytesIO()
    big.save(buf, "JPEG", quality=60)
    (core.DATA / "alice/big.jpg").write_bytes(buf.getvalue())
    core.index_file("alice/big.jpg")

    analyzer = fake_model(mocker)
    faces.scan_faces()

    assert analyzer.detected_on == [(768, 1024)]  # detector saw a copy shrunk to DET_SIZE on the long side
    assert analyzer.embedded_on == [(3000, 4000)]  # but the embedding came off the full-size image

    with core.db() as con:
        bbox = con.execute("SELECT bbox FROM faces WHERE path='alice/big.jpg'").fetchone()[0]
    x1, y1, x2, y2 = (int(v) for v in bbox.split(","))
    assert (x1, y1, x2, y2) == (8, 8, 227, 227)  # detector box scaled back to original coordinates
