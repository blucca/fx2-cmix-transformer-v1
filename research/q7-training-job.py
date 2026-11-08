"""Run one admitted full12 q7 method: read-only inputs, preflight, then 32 epochs."""
from pathlib import Path
import argparse, json, math, os, resource, subprocess, sys, time

CPU_SECONDS = 172800

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--job', type=Path, required=True)
    a = p.parse_args(); root, job = a.root.resolve(), a.job.resolve()
    state_path = job/'pipeline-state.json'
    assert not state_path.exists(), 'inspect the existing admitted pipeline'
    deadline = time.time()+7*86400
    state = dict(status='running', stage='waiting_for_inputs', method='q7-teacher-only-v1',
                 pipeline_pid=os.getpid(), cpu_limit_seconds=CPU_SECONDS, target_epochs=32,
                 deadline_utc=deadline, child_runs=[])
    def save(stage, **values):
        state.update(stage=stage, updated_at=time.time(), **values)
        q=state_path.with_suffix('.tmp');q.write_text(json.dumps(state,indent=2)+'\n');q.replace(state_path)
    def finish(status, stage, code, **values):
        save(stage, status=status, pipeline_exit_code=code, completed_at=time.time(), **values)
        (job/'pipeline.exit').write_text(str(code)+'\n')
        return code
    def run(stage, argv):
        left=math.floor(deadline-time.time());assert left>0, 'seven-day wall deadline'
        command=['timeout','-k','30s',str(left),*map(str,argv)]
        with (job/(stage+'.log')).open('w') as log:
            child=subprocess.Popen(command,cwd=job,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            save(stage,child_pid=child.pid,command=command)
            _,status,usage=os.wait4(child.pid,0);code=os.waitstatus_to_exitcode(status);child.returncode=code
        row=dict(stage=stage,exit_code=code,cpu_seconds=usage.ru_utime+usage.ru_stime,peak_rss_kib=usage.ru_maxrss)
        state['child_runs'].append(row);save(stage,child_pid=None)
        return code,row
    save('waiting_for_inputs')
    try:
        while True:
            ready_path=job/'input-transfer.json'
            if ready_path.is_file():
                ready=json.loads(ready_path.read_text())
                if ready['status']=='failed':
                    return finish('failed','input_transfer_failed',1,error=ready.get('error'))
                if ready['status']=='passed':break
            assert time.time()<deadline, 'input wait exceeded seven-day allowance'
            time.sleep(30)
        setup=json.loads((job/'setup-receipt.json').read_text())
        # Early setup/controller failures use one CPU second of carry; model child CPU is measured separately.
        input_cpu=(ready['cpu_seconds']+setup['source_clone_cpu_seconds']
                   +setup.get('source_overlay_cpu_seconds',0.)+setup.get('source_check_cpu_seconds',0.)
                   +setup.get('failed_mixed_wait_cpu_seconds',0.)+setup.get('failed_preflight_cpu_seconds',0.)+1.)
        assert 0<=input_cpu<420, 'preflight CPU room exhausted by input preparation'
        source=job/'source'; training=job/'training'
        files=dict(checkpoint=root/'round416/source/models/6m-q4-fp32.tch',kernel=job/'assets/cpu-kda.so',
                   data=job/'incoming/data-manifest.json',baseline=job/'incoming/baseline.json',
                   weights=root/'candidate/models/6m-q4-fp32.tfwc2',original_static15=job/'assets/original-static15.tfwc3',
                   teacher_cache=job/'teacher-cache',recode=job/'assets/recode',evaluator=job/'bin/activation-probe')
        assert all(path.exists() for path in files.values())
        argv=[sys.executable,'-B','-u',str(source/'research/integer-coordinate-qat.py'),'--recipe','q7-teacher-only-v1']
        for name,path in files.items():argv+=['--'+name.replace('_','-'),str(path)]
        argv+=['--output',str(training)]
        code,usage=run('preflight',argv+['--cpu-seconds','600','--prior-cpu-seconds',str(input_cpu),'--check-only'])
        if code:return finish('failed','preflight_failed',code,error='read preflight.log and training/state.json')
        check=json.loads((training/'preflight.json').read_text());initial=json.loads((training/'state.json').read_text())
        assert check['status']=='passed' and check['initialization_restored'] and check['optimizer_reset']
        assert check['ce_report_only'] and check['exact_round241_quant7_export'] and check['teacher_cache_misses']==0
        assert initial['status']=='ready' and initial['step']==initial['epoch']==0
        (job/'check').mkdir(exist_ok=True)
        (job/'check/preflight.json').write_text(json.dumps(check,indent=2)+'\n')
        # Resume the restored fresh initialization, preserving the one zero-step export and VJP receipt.
        actual_prep=input_cpu+usage['cpu_seconds']+time.process_time()
        correction=max(0.,actual_prep-initial['total_cpu_seconds'])
        ledger=dict(cpu_seconds=actual_prep,input_cpu_seconds=input_cpu,
                    input_failed_setup_cpu_upper_bound_seconds=1.,preflight_child_cpu_seconds=usage['cpu_seconds'],
                    controller_cpu_seconds=time.process_time(),resume_correction_seconds=correction,
                    reused_baseline_cpu_seconds=0.,teacher_resamples=0)
        (job/'prep-ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
        code,usage=run('training',argv+['--cpu-seconds',str(CPU_SECONDS),'--resume','--recovery-cpu-seconds',str(correction)])
        (job/'training.exit').write_text(str(code)+'\n')
        terminal=json.loads((training/'state.json').read_text())
        details=dict(training_exit_code=code,training_status=terminal['status'],training_epoch=terminal['epoch'],
                     trainer_total_cpu_seconds=terminal['total_cpu_seconds'])
        if code==0 and terminal['status'] in ('complete','cpu_budget','stopped'):
            return finish(terminal['status'],'training_terminal',code,**details)
        if code==124 and terminal['status']=='cpu_budget':
            return finish('cpu_budget','training_terminal',code,**details)
        return finish('failed','training_failed',code or 1,**details)
    except Exception as error:
        return finish('failed','failed',1,error=repr(error))

if __name__=='__main__':
    raise SystemExit(main())
