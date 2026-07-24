# ndrive — plan

## Goal

Temporary photo share for the family holiday; afterwards the pictures feed a
photo book. Lifespan: months, not years — every decision biased toward "small
enough to delete later".

## Decisions (as agreed)

- **Custom build, not Immich/Nextcloud.** Immich hard-requires Postgres; both
  miss at least one core need (mapped WebDAV drive, public likes,
  everyone-sees-all). This tool is ~1k lines on stdlib sqlite.
- **Ownership = first path segment.** `data/<user>/…`; you can write only under
  your own top folder, everyone (logged in) reads everything. One rule, enforced
  identically for WebDAV and the web UI. No ACL tables.
- **Auth = HTTP Basic everywhere** (browser and drive share credentials),
  scrypt-hashed (stdlib) in sqlite. HTTPS comes from Caddy in front — mandatory,
  Windows refuses Basic over plain HTTP.
- **No metadata sidecars.** Everything is derivable: owner = folder, datetime =
  EXIF (fallback mtime), phash/dimensions = pixels. One rebuildable sqlite index
  in `storage/cache/` (`python -m ndrive rescan` rebuilds from disk). If the
  book pipeline ever wants per-file YAML, that's an export command, not live
  state.
- **Soft delete.** Web and WebDAV deletes move into `storage/trash/<stamp>/…`;
  restore = move the file back; `purge-trash` drops old stamps (default 30 d).
- **Dedupe = warn, never block.** 64-bit perceptual hash, Hamming ≤ 6 counts as
  "close" (holiday bursts are legitimate near-dups). Web upload warns inline;
  WebDAV arrivals get a "dup?" badge in the gallery — a PUT has no dialog
  channel.
- **Formats: jpg / jpeg / png / heic / heif.** Browser always gets JPEG
  renditions (400 px thumbs, 2048 px view — HEIC becomes viewable this way);
  originals are never touched; zip download ships originals.

## Phase 1 (this repo)

- `core.py` — storage layout, users (scrypt), ownership rule, photo index
  (EXIF datetime, phash, dup detection), soft delete, likes, renditions.
- `app.py` — FastAPI: gallery (sort by date, filter by owner, multiselect →
  zip download / delete-with-"not yours"-skip, likes, lightbox), WsgiDAV
  mounted at `/dav` behind a small WSGI guard that enforces ownership,
  soft-deletes on DELETE, and indexes after PUT/MOVE/COPY.
- `__main__.py` — `serve`, `adduser`, `rescan`, `purge-trash`.
- Startup reconciles index with disk in a background thread.

## Phase 2 — faces (starts once pictures exist)

- Pretrained insightface embeddings (CPU, onnxruntime), background scan →
  sqlite rows (photo, bbox, 512-d vector, label NULL).
- **No model training.** The guess for an unlabeled face = cosine similarity to
  the per-person mean of labeled embeddings. The labeling page shows the face
  crop with the guessed name pre-selected in the person list; confirming stores
  the label, which immediately sharpens future guesses.
- Gallery grows a filter-by-person dropdown.
- Face labels are the one thing *not* derivable from files — back them up,
  unlike the rest of the cache.

## Skipped on purpose

Videos, quotas, sessions/logout, trash-restore UI, dedupe blocking, admin UI,
share links. Each gets added when someone actually asks, not before.
