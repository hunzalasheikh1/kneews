"""Resume extraction and verify each newly extracted member with ZIP CRC checks."""
import json
import os
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/rsna-knee-abnormality-detection"
STATUS = DEST / "extraction-status.json"


def main():
    start = time.time()
    with zipfile.ZipFile(DEST / "rsna-knee-abnormality-detection.zip") as archive:
        members = sorted(archive.infolist(), key=lambda i: (not i.filename.endswith('.csv'), i.filename))
        total_bytes = sum(i.file_size for i in members)
        done_bytes = 0
        for index, member in enumerate(members, 1):
            target = DEST / member.filename
            if not target.resolve().is_relative_to(DEST.resolve()):
                raise ValueError("Unsafe archive path")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not target.exists() or target.stat().st_size != member.file_size:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + '.partial')
                with archive.open(member) as source, temporary.open('wb') as sink:
                    while chunk := source.read(4 * 1024 * 1024):
                        sink.write(chunk)
                os.replace(temporary, target)
            done_bytes += member.file_size
            if index % 500 == 0 or index == len(members):
                status = dict(status='complete' if index == len(members) else 'running',
                              files=index, total_files=len(members), bytes=done_bytes,
                              total_bytes=total_bytes, elapsed_seconds=round(time.time()-start),
                              updated_at=time.time(), pid=os.getpid())
                temporary = STATUS.with_suffix('.tmp')
                temporary.write_text(json.dumps(status, indent=2))
                os.replace(temporary, STATUS)
                print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
