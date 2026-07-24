"""FastAPI web UI + WsgiDAV mount, sharing one auth and one ownership rule."""

import base64
import json
import os
import shutil
import tempfile
import threading
import zipfile
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

from a2wsgi import WSGIMiddleware
from fastapi import Depends, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.background import BackgroundTask
from wsgidav.dc.base_dc import BaseDomainController
from wsgidav.fs_dav_provider import FilesystemProvider
from wsgidav.wsgidav_app import WsgiDAVApp

from ndrive import core, faces

security = HTTPBasic()
DAV_PREFIX = "/dav"
WRITE_METHODS = {"PUT", "MKCOL", "PROPPATCH", "DELETE", "MOVE", "COPY", "LOCK", "UNLOCK"}


def current_user(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    canonical = core.verify_user(credentials.username, credentials.password)
    if not canonical:
        raise HTTPException(401, "Bad credentials", headers={"WWW-Authenticate": 'Basic realm="ndrive"'})
    return canonical


class PathsPayload(BaseModel):
    paths: list[str]


class LikePayload(BaseModel):
    path: str


class MkdirPayload(BaseModel):
    folder: str


class FaceLabelPayload(BaseModel):
    face_id: int
    label: str


# --- WebDAV ----------------------------------------------------------------


class SqliteDomainController(BaseDomainController):
    def get_domain_realm(self, path_info, environ):
        return "ndrive"

    def require_authentication(self, realm, environ):
        return True

    def basic_auth_user(self, realm, user_name, password, environ):
        return core.verify_user(user_name, password) is not None

    def supports_http_digest_auth(self):
        return False

    def digest_auth_user(self, realm, user_name, environ):
        return False


def _basic_creds(environ) -> tuple[str, str]:
    header = environ.get("HTTP_AUTHORIZATION", "")
    if header.lower().startswith("basic "):
        try:
            user, _, pw = base64.b64decode(header[6:].strip()).decode(errors="replace").partition(":")
            return user, pw
        except ValueError:
            pass
    return "", ""


def _dest_rel(environ) -> str:
    dest = unquote(urlsplit(environ.get("HTTP_DESTINATION", "")).path)
    for prefix in (environ.get("SCRIPT_NAME", ""), DAV_PREFIX):
        if prefix and dest.startswith(prefix):
            dest = dest[len(prefix) :]
            break
    return dest.strip("/")


def _reply(start_response, status: str, body: bytes = b"", extra=()):
    headers = [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(body))), *extra]
    start_response(status, headers)
    return [body]


def _guarded(dav_app):
    """Ownership + soft-delete in front of WsgiDAV: writes only inside your own folder."""

    def app(environ, start_response):
        if not environ.get("SCRIPT_NAME"):
            environ["SCRIPT_NAME"] = DAV_PREFIX  # a2wsgi loses the mount prefix; keeps redirect URLs correct
        method = environ["REQUEST_METHOD"].upper()
        rel = environ.get("PATH_INFO", "").strip("/")
        if method not in WRITE_METHODS:
            return dav_app(environ, start_response)  # reads: everyone (WsgiDAV still checks the password)

        claimed, pw = _basic_creds(environ)
        user = core.canonical_user(claimed) if claimed else None  # case-insensitive; canonical case owns the folder
        if not user:
            return _reply(start_response, "401 Unauthorized", extra=(("WWW-Authenticate", 'Basic realm="ndrive"'),))
        targets = [rel] if method != "COPY" else []  # COPY reads the source, so only its destination must be yours
        if method in {"MOVE", "COPY"}:
            targets.append(_dest_rel(environ))
        if not all(core.can_write(user, t) for t in targets):
            return _reply(
                start_response, "403 Forbidden", b"You can only add, change or delete files inside your own folder."
            )

        if method == "DELETE":
            # handled here, not by WsgiDAV: soft-delete into trash/ (family + hard delete = tears)
            if not core.verify_user(claimed, pw):
                return _reply(start_response, "401 Unauthorized", extra=(("WWW-Authenticate", 'Basic realm="ndrive"'),))
            if not core.resolve(rel).exists():
                return _reply(start_response, "404 Not Found")
            core.move_to_trash(rel)
            return _reply(start_response, "204 No Content")

        seen = {}

        def capture(status, headers, exc_info=None):
            seen["status"] = status
            return start_response(status, headers, exc_info)

        chunks = list(dav_app(environ, capture))
        if seen.get("status", "").startswith("2"):
            if method == "PUT":
                core.index_file(rel)
            elif method == "MOVE":
                core.rename_paths(rel, _dest_rel(environ))
            elif method == "COPY":
                core.index_tree(_dest_rel(environ))
        return chunks

    return app


# --- app -------------------------------------------------------------------


def create_app(home: str | Path | None = None) -> FastAPI:
    core.configure(home or os.environ.get("NDRIVE_HOME", "storage"))
    dav = WsgiDAVApp(
        {
            "mount_path": DAV_PREFIX,  # drives href generation and Destination-header stripping
            "provider_mapping": {"/": FilesystemProvider(str(core.DATA))},
            "http_authenticator": {
                "domain_controller": SqliteDomainController,
                "accept_basic": True,
                "accept_digest": False,
                "default_to_digest": False,
            },
            "property_manager": True,  # Windows PROPPATCHes timestamps; in-memory is fine for a temporary drive
            "verbose": 1,
        }
    )
    app = FastAPI(title="ndrive")
    app.mount(DAV_PREFIX, WSGIMiddleware(_guarded(dav)))
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
    templates.env.filters["q"] = lambda value: quote(str(value))

    @app.get("/", response_class=HTMLResponse)
    def gallery(
        request: Request,
        user: str = Depends(current_user),
        owner: str | None = None,
        sort: str = "asc",
        person: str | None = None,
        liked: int = 0,
    ):
        photos = core.list_photos(user, owner=owner, sort=sort, person=person, liked_only=bool(liked))
        return templates.TemplateResponse(
            request,
            "gallery.html",
            {
                "user": user,
                "photos": photos,
                "owners": core.users(),
                "owner": owner or "",
                "sort": sort,
                "liked": liked,
                "my_folders": core.subfolders(user),
                "dup_count": len(core.duplicate_pairs(user)),
                "people": faces.people(),
                "person": person or "",
                "unlabeled_count": faces.unlabeled_count(),
                "paths_json": json.dumps([p["path"] for p in photos]),
            },
        )

    @app.get("/duplicates", response_class=HTMLResponse)
    def duplicates(request: Request, user: str = Depends(current_user)):
        return templates.TemplateResponse(request, "dups.html", {"user": user, "pairs": core.duplicate_pairs(user)})

    @app.get("/faces", response_class=HTMLResponse)
    def faces_page(request: Request, user: str = Depends(current_user)):
        return templates.TemplateResponse(
            request,
            "faces.html",
            {
                "user": user,
                "faces": faces.unlabeled(),
                "people": faces.label_options(),
                "total": faces.unlabeled_count(),
                "ignore": faces.IGNORE,
                "stranger": faces.STRANGER,
            },
        )

    @app.get("/face/{face_id}")
    def face_crop(face_id: int, user: str = Depends(current_user)):
        try:
            return FileResponse(faces.crop(face_id))
        except FileNotFoundError as exc:
            raise HTTPException(404) from exc

    @app.post("/api/face")
    def face_label(payload: FaceLabelPayload, user: str = Depends(current_user)):
        if not payload.label.strip():
            raise HTTPException(400, "Empty label")
        faces.set_label(payload.face_id, payload.label)
        return {"remaining": faces.unlabeled_count()}

    @app.get("/thumb/{rel:path}")
    def thumb(rel: str, user: str = Depends(current_user)):
        return FileResponse(core.rendition(rel, core.THUMB_SIDE))

    @app.get("/view/{rel:path}")
    def view(rel: str, user: str = Depends(current_user)):
        return FileResponse(core.rendition(rel, core.VIEW_SIDE))

    @app.get("/media/{rel:path}")
    def media(rel: str, user: str = Depends(current_user)):
        abs_ = core.resolve(rel)
        if not abs_.is_file():
            raise HTTPException(404)
        return FileResponse(abs_, filename=abs_.name)

    @app.get("/stream/{rel:path}")
    def stream(rel: str, user: str = Depends(current_user)):
        src = core.stream_source(rel)
        if not src.is_file():
            raise HTTPException(404)
        return FileResponse(src, media_type="video/mp4" if src.suffix == ".mp4" else None)

    @app.post("/api/mkdir")
    def mkdir(payload: MkdirPayload, user: str = Depends(current_user)):
        folder = payload.folder.strip().strip("/")
        rel = f"{user}/{folder}"
        if not folder or not core.can_write(user, rel) or any(p.startswith(".") or not p for p in rel.split("/")):
            raise HTTPException(400, "Invalid folder name")
        core.resolve(rel).mkdir(parents=True, exist_ok=True)
        return {"folder": folder}

    @app.post("/api/upload")
    def upload(files: list[UploadFile], folder: str = Form(""), user: str = Depends(current_user)):
        parent = f"{user}/{folder}".strip("/")
        if not core.can_write(user, parent) or not core.resolve(parent).is_dir():
            raise HTTPException(400, "Bad upload folder")
        warnings = []
        for f in files:
            rel = core.unique_dest(parent, f.filename or "upload")
            with core.resolve(rel).open("wb") as out:
                shutil.copyfileobj(f.file, out)
            if dups := core.index_file(rel):
                warnings.append(f"{Path(rel).name} looks like a duplicate of: {', '.join(dups)}")
        return {"count": len(files), "warnings": warnings}

    @app.post("/api/delete")
    def delete(payload: PathsPayload, user: str = Depends(current_user)):
        deleted = skipped = 0
        for rel in payload.paths:
            if not core.can_write(user, rel):
                skipped += 1
            elif core.resolve(rel).exists():
                core.move_to_trash(rel)
                deleted += 1
        message = f"Moved {deleted} picture(s) to the trash."
        if skipped:
            message += f" Skipped {skipped} — they aren't yours."
        return {"deleted": deleted, "skipped": skipped, "message": message}

    @app.post("/api/like")
    def like(payload: LikePayload, user: str = Depends(current_user)):
        liked, count = core.toggle_like(user, payload.path)
        return {"liked": liked, "count": count}

    @app.post("/api/keep-both")
    def keep_both(payload: LikePayload, user: str = Depends(current_user)):
        if user not in core.pair_owners(payload.path):
            raise HTTPException(403, "Only the owners involved can resolve this pair")
        core.clear_dup(payload.path)
        return {"ok": True}

    @app.post("/download")
    def download(user: str = Depends(current_user), paths: list[str] = Form(...)):
        tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)  # noqa: SIM115 — unlinked by BackgroundTask
        with zipfile.ZipFile(tmp, "w") as zf:
            for rel in paths:
                abs_ = core.resolve(rel)
                if abs_.is_file():
                    zf.write(abs_, arcname=rel)
        tmp.close()
        return FileResponse(tmp.name, filename="ndrive-selection.zip", background=BackgroundTask(os.unlink, tmp.name))

    def _startup_scan():
        core.scan_all()
        faces.scan_faces()
        core.transcode_all()

    threading.Thread(target=_startup_scan, daemon=True, name="ndrive-scan").start()
    return app
