"""Wait for fully validated inputs, then launch a fresh training process once."""
import json
import subprocess
import sys
import time
from sqlalchemy import select
from ..config import settings,ROOT
from ..db import SessionLocal
from ..models import Experiment
from .leases import stage_lease
from .readiness import inspect_inputs
from .training import ARMS


def inputs_complete(result):
    total=result.get('total',0)
    return bool(result.get('ready_for_training')) and total>0 and result.get('valid_features')==total and result.get('valid_labels')==total and not any(
        result.get(key) for key in ['missing_features','invalid_features','missing_labels','invalid_labels'])


def await_ready(inspect,launch,publish,sleep=time.sleep):
    while True:
        result=inspect()
        publish(dict(status='waiting',total=result['total'],valid_features=result['valid_features'],valid_labels=result['valid_labels'],
            invalid_features=len(result['invalid_features']),invalid_labels=len(result['invalid_labels'])))
        if inputs_complete(result):
            publish({'status':'launching'})
            return launch()
        sleep(60)


def wait_and_train(label_version,feature_signature,include_reserve=False,device='auto'):
    from .devices import resolve_device
    resolved=str(resolve_device(device))
    config=dict(label_version=label_version,feature_signature=feature_signature,include_reserve=include_reserve,device=resolved)
    root=settings.artifact_root/'training_wait';root.mkdir(parents=True,exist_ok=True)
    path=root/(str(time.time_ns())+'.json');status=dict(config=config,started_at=time.time())
    def publish(update):
        status.update(update);status['updated_at']=time.time()
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(status,indent=2));temporary.replace(path)
        print(json.dumps(update),flush=True)
    def inspect():
        with SessionLocal() as db:return inspect_inputs(db,label_version,feature_signature)
    def launch():
        with SessionLocal() as db:
            existing=db.scalars(select(Experiment).where(Experiment.status=='completed')).all()
            arms=list((*ARMS,'reserve') if include_reserve else ARMS)
            if any(e.config.get('label_version')==label_version and e.config.get('feature_signature')==feature_signature
                   and e.config.get('arms')==arms and e.config.get('seeds')==[2026,2027,2028] and e.config.get('epochs')==30 for e in existing):
                raise RuntimeError('Matching completed experiment already exists; review it before starting another explicit run')
        # Import code afresh after the potentially long wait. The manifest then hashes the code actually loaded for training.
        command=[sys.executable,'-m','app.research.cli','train','--label-version',label_version,'--feature-signature',feature_signature,'--device',resolved]
        if include_reserve:command.append('--include-reserve')
        publish({'status':'training','command':command})
        result=subprocess.run(command,cwd=ROOT)
        if result.returncode:raise RuntimeError(f'Training process exited with status {result.returncode}; inspect its saved experiment')
        publish({'status':'completed'});return result.returncode
    with stage_lease('training-waiter'):
        try:return await_ready(inspect,launch,publish)
        except BaseException as exc:
            publish(dict(status='interrupted' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'failed',error=str(exc)))
            raise
