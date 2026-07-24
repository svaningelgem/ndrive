# ndrive

Temporary family photo drive for the holiday: everyone uploads into their own
folder, everyone (logged in) sees and likes everything, only you can change or
delete your own pictures. WebDAV for drag & drop from Windows, a web gallery
for browsing, liking and picking pictures for the photo book.

See `PLAN.md` for the design and what's deliberately left out.

## Run

```bash
python -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python -m ndrive adduser steven      # once per family member
.venv/bin/python -m ndrive serve --port 8484
```

Put HTTPS in front (mandatory — Windows refuses Basic auth over plain HTTP),
e.g. Caddy:

```
photos.example.com {
    reverse_proxy 127.0.0.1:8484
}
```

## Windows drive

Explorer → This PC → Map network drive → `https://photos.example.com/dav`, or:

```
net use P: "https://photos.example.com/dav" /user:steven /persistent:yes
```

Notes:

- The built-in client caps files at 50 MB. Fix: registry key
  `HKLM\SYSTEM\CurrentControlSet\Services\WebClient\Parameters\FileSizeLimitInBytes`
  → `0xffffffff`, then restart the *WebClient* service.
- If the built-in client annoys you, rclone or RaiDrive mount the same URL
  better.

## Phones

The web gallery's upload button works fine on mobile. For automatic camera
upload, point the PhotoSync app (iOS/Android) at the WebDAV URL and your own
folder.

## Rules of the drive

- You can only add/change/delete inside your own folder (`data/<you>/`);
  everything is visible to every logged-in user. Selecting other people's
  pictures for deletion just skips them (the web UI tells you).
- Deleting (web or WebDAV) moves files to `storage/trash/<timestamp>/…`.
  Restore = move the file back. `python -m ndrive purge-trash --days 30`
  empties old trash.
- jpg / png / heic are indexed and shown in the gallery (HEIC is converted to
  JPEG for the browser; originals stay untouched). Videos (mp4 / mov / m4v /
  3gp / avi) are in the gallery too: frame-grab thumbnail with a ▶ badge,
  playback in the lightbox. There is no transcoding — if your browser can't
  play a codec (e.g. HEVC on some desktops), use the download link. Other file
  types are fine on the drive — they're just not media.
- A `dup?` badge or upload warning means a perceptual-hash near-match; nothing
  is ever blocked.

## Ops

- **Backup**: rsync/restic `storage/data` (add `storage/trash` if you care).
  `storage/cache` is disposable — `python -m ndrive rescan` rebuilds the index
  from disk.
- **Users**: `python -m ndrive adduser <name>` (re-running resets the
  password).
