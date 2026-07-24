"""CLI: serve / adduser / rescan / purge-trash."""

import argparse
import getpass
import logging
import os

import uvicorn

from ndrive import core, faces
from ndrive.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(prog="ndrive", description="Temporary family photo drive")
    parser.add_argument(
        "--home", default=os.environ.get("NDRIVE_HOME", "storage"), help="storage root (default: ./storage)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    serve = sub.add_parser("serve", help="run the server (put Caddy or similar in front for HTTPS)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8484)
    adduser = sub.add_parser("adduser", help="create a user and their folder (re-run to reset a password)")
    adduser.add_argument("username")
    rename = sub.add_parser("renameuser", help="rename an account; folder, photos, likes and faces follow")
    rename.add_argument("old")
    rename.add_argument("new")
    sub.add_parser("rescan", help="rebuild the photo index from the files on disk")
    sub.add_parser("scan-faces", help="detect + embed faces for photos not scanned yet")
    purge = sub.add_parser("purge-trash", help="permanently drop old trash")
    purge.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    match args.cmd:
        case "serve":
            uvicorn.run(create_app(args.home), host=args.host, port=args.port)
        case "adduser":
            core.configure(args.home)
            core.add_user(args.username, getpass.getpass("Password: "))
            print(f"user {args.username} ready — folder data/{args.username}")
        case "renameuser":
            core.configure(args.home)
            core.rename_user(args.old, args.new)
            print(f"{args.old} → {args.new}")
        case "rescan":
            core.configure(args.home)
            core.scan_all()
        case "scan-faces":
            core.configure(args.home)
            faces.scan_faces()
        case "purge-trash":
            core.configure(args.home)
            print(f"removed {core.purge_trash(args.days)} trash folder(s)")


if __name__ == "__main__":
    main()
