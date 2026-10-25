"""Fixed 3,686-unit selection; both untouched second halves must pass."""
from pathlib import Path
import csv, json, math, sys

j = Path(sys.argv[1])
p = json.loads((j/'protocol.json').read_text())
samples = ['prefix-1m','article-q2-1m']
rows = {}
for sample in samples:
    with (j/(sample+'.csv')).open() as f:
        data = list(csv.DictReader(f))
    assert len(data) == p['unit_count']
    rows[sample] = []
    for index, row in enumerate(data):
        r = {key: float(value) if key == 'norm2' else int(value) for key,value in row.items()}
        assert (r['layer'],r['unit']) == divmod(index,768)
        assert math.isfinite(r['norm2']) and r['norm2'] >= 0
        for half in ['first','second']:
            assert r[half+'_hits'] <= r[half+'_q2'] <= r[half+'_hits']*127**2
        rows[sample].append(r)
for a,b in zip(*(rows[s] for s in samples)):
    assert a['norm2'] == b['norm2']
energies = {s:{h:[r[h+'_q2']*r['norm2'] for r in rows[s]]
                 for h in ['first','second']} for s in samples}
train = [sum(energies[s]['first'][i] for s in samples) for i in range(p['unit_count'])]
selected = sorted(range(p['unit_count']), key=lambda i: (train[i],i))[:p['selected_count']]
selected_set = set(selected)
summary = {'selected_count':len(selected),'selected_per_layer':[sum(i//768==l for i in selected) for l in range(12)],
           'samples':{},'protocol':p,'mask_or_pack_started':False}
passed = True
for s in samples:
    info = {}
    for h in ['first','second']:
        e = energies[s][h]
        total = math.fsum(e)
        assert total > 0
        share = math.fsum(e[i] for i in selected)/total
        per_layer = []
        for l in range(12):
            ids = range(l*768,(l+1)*768)
            den = math.fsum(e[i] for i in ids)
            per_layer.append(math.fsum(e[i] for i in ids if i in selected_set)/den if den else 0)
        info[h] = {'energy_share':share,'total_energy':total,'selected_energy':math.fsum(e[i] for i in selected),
                   'per_layer_energy_share':per_layer}
    passed = passed and info['second']['energy_share'] <= p['max_second_half_energy_share']
    info['zero_observed_units'] = sum(r['first_hits']+r['second_hits']==0 for r in rows[s])
    summary['samples'][s] = info
zeros = [i for i in range(p['unit_count']) if all(rows[s][i]['first_hits']+rows[s][i]['second_hits']==0 for s in samples)]
summary.update(joint_zero_observed_units=len(zeros),joint_zero_two_copy_raw_nibble_upper_bytes=len(zeros)*384,
               decision='pass-read-only-gate-decide-mask-separately' if passed else 'stop-fixed-near-zero-unit-pruning')
(j/'selection.json').write_text(json.dumps({'selected': [list(divmod(i,768)) for i in selected],
                                           'joint_zero_units':[list(divmod(i,768)) for i in zeros]},indent=2)+'\n')
(j/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
