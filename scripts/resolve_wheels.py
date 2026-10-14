"""Build an offline wheelhouse using fetched PyPI metadata (development helper)."""
import json
from pathlib import Path
from pip._vendor.packaging.tags import sys_tags
from pip._vendor.packaging.utils import parse_wheel_filename, canonicalize_name
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.version import Version
from pip._vendor.packaging.specifiers import SpecifierSet

base = Path('.build')
tags = set(sys_tags())
queue = ['fastapi', 'uvicorn', 'sqlalchemy', 'psycopg[binary]', 'alembic',
         'pydantic-settings', 'python-multipart', 'pydicom', 'numpy', 'scipy',
         'scikit-learn', 'pillow', 'nibabel', 'pytest', 'httpx', 'pandas',
         'pylibjpeg', 'pylibjpeg-libjpeg', 'pylibjpeg-openjpeg', 'torch',
         'torchvision', 'transformers', 'sentencepiece', 'lingua-language-detector']
seen, metadata, downloads, errors = set(), {}, {}, []
while queue:
    req = Requirement(queue.pop(0))
    name = canonicalize_name(req.name)
    key = (name, tuple(sorted(req.extras)))
    if key in seen:
        continue
    seen.add(key)
    path = base/'pypi'/f'{name}.json'
    if not path.exists():
        metadata[name] = f'https://pypi.org/pypi/{name}/json'
        continue
    data = json.loads(path.read_text())
    compatible = []
    for item in data['urls']:
        if item['yanked'] or not item['filename'].endswith('.whl'):
            continue
        if item.get('requires_python') and Version('3.12.13') not in SpecifierSet(item['requires_python']):
            continue
        try:
            _, version, _, wt = parse_wheel_filename(item['filename'])
            if wt & tags and version in req.specifier:
                compatible.append(item)
        except ValueError:
            pass
    if not compatible:
        errors.append(f'No latest compatible wheel: {req}')
        continue
    item = sorted(compatible, key=lambda x:x['filename'])[0]
    if not (base/'wheels'/item['filename']).exists():
        downloads[item['filename']] = item['url']
    for dependency in data['info'].get('requires_dist') or []:
        dep = Requirement(dependency)
        if dep.marker is None or any(dep.marker.evaluate({'extra': extra}) for extra in ['', *req.extras]):
            queue.append(dependency)
print(json.dumps(dict(metadata=metadata, downloads=downloads, errors=errors)))
