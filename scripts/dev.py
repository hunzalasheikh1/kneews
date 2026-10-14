"""Start the local PostgreSQL-backed API, durable worker and Next.js interface."""
import os
import signal
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PYTHON=ROOT/'.venv312/bin/python'
env={**os.environ,'PYTHONPATH':str(ROOT/'backend')}
if len(sys.argv)>1 and sys.argv[1]=='ui-test':
    sys.exit(subprocess.call(['node',str(ROOT/'scripts/ui_smoke.mjs')],cwd=ROOT,env=env))
if len(sys.argv)>1 and sys.argv[1]=='test':
    env['KNEEWS_INTEGRATION']='1'
    sys.exit(subprocess.call([str(PYTHON),'-m','pytest','backend/tests',*sys.argv[2:]],cwd=ROOT,env=env))
if len(sys.argv)>1 and sys.argv[1]=='model-server':
    parallel=sys.argv[2] if len(sys.argv)>2 else '2'
    if parallel not in ('1','2','3','4'):raise SystemExit('Model parallel requests must be 1, 2, 3 or 4')
    model_env={**os.environ,'OLLAMA_HOST':'127.0.0.1:11435','OLLAMA_NO_CLOUD':'true','OLLAMA_NUM_PARALLEL':parallel}
    sys.exit(subprocess.call(['ollama','serve'],cwd=ROOT,env=model_env))
if len(sys.argv)>1 and sys.argv[1]=='research':
    sys.exit(subprocess.call([str(PYTHON),'-m','app.research.cli',*sys.argv[2:]],cwd=ROOT,env=env))
subprocess.run([str(PYTHON),'-m','app.bootstrap'],cwd=ROOT,env=env,check=True)
children=[]
def shutdown(*_):
    for child in children:
        if child.poll() is None: child.terminate()
    for child in children:
        try: child.wait(timeout=10)
        except subprocess.TimeoutExpired: child.kill()
    sys.exit(0)
signal.signal(signal.SIGTERM,shutdown)
signal.signal(signal.SIGINT,shutdown)
try:
    children.append(subprocess.Popen([str(PYTHON),'-m','uvicorn','app.main:app','--host','127.0.0.1','--port','8200'],cwd=ROOT,env=env))
    children.append(subprocess.Popen([str(PYTHON),'-m','app.jobs'],cwd=ROOT,env=env))
    children.append(subprocess.Popen(['npm','run','dev'],cwd=ROOT/'web',env=env))
    children[-1].wait()
finally: shutdown()
