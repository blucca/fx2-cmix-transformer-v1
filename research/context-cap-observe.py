# SPDX-License-Identifier: GPL-3.0-only
# Read-only diagnostic of data-defined external mixer keys from an existing trace.
from pathlib import Path
import struct, math, json, time, sys
start=time.monotonic()
root=Path(__file__).resolve().parents[1]
if len(sys.argv)!=2: raise SystemExit('usage: context-cap-observe.py TRACE.bin')
trace=Path(sys.argv[1])
dictbytes=(root/'dictionary/english.dic').read_bytes().replace(b'\n',b' ')
prev,prev2,prev3=dictbytes[-1],dictbytes[-2],dictbytes[-3]
words=0; old=0
for c in dictbytes[-16:]:
 words=((words*2)+int((97<=c<=122 or c>=128) and old!=12))&127
 if c==10: words=124
 elif c in (44,46): words|=124
 old=c
prefix=1; byte=0; bitno=0
names=('mx9','mx10=mx11','mx14','mx15','mx16','wordscxt')
state={n:{'kept':set(),'all':set(),'counts':{},'cap_at':None,'fallback_bits':0,'loss_bits':0.0,'fallback_loss_bits':0.0,'blocks':[]} for n in names}
block={n:[0,0.0,0.0] for n in names}
with trace.open('rb') as f:
 while buf:=f.read(128*1024):
  for p,flags,neural,fxcm,mixer in struct.iter_unpack('<HHfff',buf):
   bit=flags&1
   loss=-math.log2((p if bit else 65536-p)/65536)
   keys=((((prev2*16)+prev)&255)*256+prefix,prev*256+prefix,prev*256+prev2,prev2*256+prev3,prev2*256+prefix,words*256+prefix)
   for n,key in zip(names,keys):
    s=state[n]; kept=s['kept'];s['all'].add(key)
    if key not in kept and len(kept)<10000:
     kept.add(key)
     if len(kept)==10000:s['cap_at']=bitno
    miss=key not in kept
    s['fallback_bits']+=miss;s['loss_bits']+=loss;s['fallback_loss_bits']+=loss*miss
    block[n][0]+=miss;block[n][1]+=loss;block[n][2]+=loss*miss
    s['counts'][key]=s['counts'].get(key,0)+1
   prefix=prefix*2+bit;byte=byte*2+bit;bitno+=1
   if prefix>=256:
    c=byte
    words=((words*2)+int((97<=c<=122 or c>=128) and prev!=12))&127
    if c==10: words=124
    elif c in (44,46): words|=124
    prev3,prev2,prev=prev2,prev,c
    prefix=1;byte=0
   if bitno%(256*1024*8)==0:
    for n in names:
     z=block[n];state[n]['blocks'].append({'end_input_byte':bitno//8,'fallback_fraction':z[0]/(256*1024*8),'base_loss_bits':z[1],'fallback_loss_fraction':z[2]/z[1]})
    block={n:[0,0.0,0.0] for n in names}
res={'trace':str(trace),'input_bytes':bitno//8,'scope':'actual stream-defined keys, same dictionary pretraining and 10000 first-distinct admission; losses from the named probability trace; contexts other than named subset not measured','dictionary_does_not_allocate_external_mixer_keys':True,'models':{}}
for n,s in state.items():
 kept=s.pop('kept');alls=s.pop('all');counts=s.pop('counts')
 s['distinct_total']=len(alls);s['kept_keys']=len(kept);s['fallback_fraction']=s['fallback_bits']/bitno;s['fallback_loss_fraction']=s['fallback_loss_bits']/s['loss_bits']
 s['kept_keys_seen_under_32_bits']=sum(counts[k]<32 for k in kept)
 s['unkept_keys_seen_at_least_256_bits']=sum(counts[k]>=256 for k in alls-kept)
 res['models'][n]=s
res['elapsed_seconds']=time.monotonic()-start
print(json.dumps(res,indent=2))
