"""One live cap/accounting handoff; original capture and pipeline groups stay fixed."""
from pathlib import Path
import json, os, resource, signal, time

J = Path('/data/workspace/temp/hutter/round414')
D = Path('/data/workspace/temp/hutter/round423')
D.mkdir(exist_ok=True)
CAP, PIPE, START = 30420, 30417, 4678336
NEW, PREP = 18000, 34200
ID = 'capture-cap-capacity-overhead-round423'
CAPACITY_CPU = .005143194
assert 14400 + NEW + 1800 == PREP
state = dict(schema='hutter.capture.live-cap-handoff.v1', status='starting',
             capture_pid=CAP, pipeline_group=PIPE, new_cpu_limit_seconds=NEW,
             authorized_prep_cpu_limit_seconds=PREP, new_scans=0,
             source_changes=0, teacher_calls=0, paused_pipeline=False)

def save(stage, **items):
    state.update(stage=stage, updated_at=time.time(), **items)
    p = D/'capture-cap-handoff-state.json'
    t = p.with_suffix('.tmp'); t.write_text(json.dumps(state, indent=2)+'\n'); t.replace(p)

def reaccount(captured, overhead):
    assert captured['cpu_limit_seconds'] == 16200 and captured['prep_cpu_limit_seconds'] == 32400
    items = captured.setdefault('capacity_overhead_items', [])
    assert not any(item['id'] == ID for item in items)
    captured['launch_budget_snapshot'] = dict(cpu_limit_seconds=16200, prep_cpu_limit_seconds=32400)
    items.append(dict(id=ID, cpu_seconds=overhead,
                      scope='one capacity metadata worker plus terminal handoff self and waited child CPU'))
    captured['recovery_prepare_cpu_seconds'] += overhead
    captured.update(cpu_limit_seconds=NEW, prep_cpu_limit_seconds=PREP,
                    runtime_capacity_override=dict(receipt=str(D/'capture-cap-handoff-state.json'),
                                                   capture_pid=CAP, starttime_ticks=START))
    return captured

save('guard')
try:
    captured = json.loads((J/'capture-state.json').read_text())
    pipeline = json.loads((J/'pipeline-state.json').read_text())
    assert captured['status'] == 'running' and captured['capture_pid'] == CAP
    assert pipeline['status'] == 'running' and pipeline['stage'] == 'waiting_for_inputs'
    assert not (J/'training/state.json').exists() and not (J/'capture.exit').exists()
    stat = Path(f'/proc/{CAP}/stat').read_text()
    fields = stat[stat.rfind(')')+2:].split()
    assert stat.split(' (', 1)[1].split(')', 1)[0] == 'causal-ppmd'
    assert int(fields[19]) == START and os.getpgid(CAP) == 30416
    assert os.getpgid(PIPE) == PIPE
    before = resource.prlimit(CAP, resource.RLIMIT_CPU)
    assert before == (16200, 16200), before
    resource.prlimit(CAP, resource.RLIMIT_CPU, (NEW, NEW))
    assert resource.prlimit(CAP, resource.RLIMIT_CPU) == (NEW, NEW)
    os.killpg(PIPE, signal.SIGSTOP); state['paused_pipeline'] = True
    deadline = captured['started_at'] + 43200
    save('waiting_capture_terminal', status='waiting', before_cpu_limit=list(before),
         after_cpu_limit=[NEW, NEW], original_capture_start=captured['started_at'],
         deadline=deadline, original_pipeline_stage=pipeline['stage'])
    # Only the existing local terminal is watched. No capacity/network/model polling.
    while not (J/'capture.exit').exists():
        if time.time() >= deadline: raise TimeoutError('original_capture_wall_deadline')
        time.sleep(min(30, max(.1, deadline-time.time())))
    terminal = json.loads((J/'capture-state.json').read_text())
    assert terminal['status'] in ('passed', 'failed') and terminal['capture_pid'] == CAP
    used = resource.getrusage(resource.RUSAGE_CHILDREN)
    overhead = CAPACITY_CPU + time.process_time() + used.ru_utime + used.ru_stime
    terminal = reaccount(terminal, overhead)
    p = J/'capture-state.json'; t = p.with_suffix('.cap-round423.tmp')
    t.write_text(json.dumps(terminal, indent=2)+'\n'); t.replace(p)
    save('terminal_accounted', status='complete', capture_status=terminal['status'],
         capture_exit=int((J/'capture.exit').read_text()), accounted_overhead_cpu_seconds=overhead,
         recovery_prepare_cpu_seconds=terminal['recovery_prepare_cpu_seconds'])
except Exception as e:
    save('failed', status='failed', error=repr(e))
    raise
finally:
    if state['paused_pipeline']:
        os.killpg(PIPE, signal.SIGCONT)
        state['paused_pipeline'] = False
        save(state['stage'], pipeline_continued=True)
