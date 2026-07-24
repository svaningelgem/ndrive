import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ndrive import core
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
def client(tmp_path: Path) -> TestClient:
    core.configure(tmp_path)
    core.add_user(*ALICE)
    core.add_user(*BOB)
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
    assert "alice/a.jpg" in client.get("/", auth=BOB).text  # everyone sees everyone's pictures


def test_duplicate_warning(client: TestClient) -> None:
    client.post("/api/upload", auth=ALICE, files=[("files", ("a.jpg", gradient_jpeg((128, 96)), "image/jpeg"))])
    r = client.post("/api/upload", auth=ALICE, files=[("files", ("b.jpg", gradient_jpeg((100, 75)), "image/jpeg"))])
    assert "duplicate" in r.json()["warnings"][0]
    with core.db() as con:
        assert con.execute("SELECT dup_of FROM photos WHERE path='alice/b.jpg'").fetchone()[0] == "alice/a.jpg"


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
