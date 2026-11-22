from pathlib import Path
from urllib.request import urlopen
import json
url='https://huggingface.co/Xenova/distilgpt2/resolve/a41c10485c18a64b6606729b6a082330cbd8f49e/tokenizer.json'
path=Path('tokenizer.json')
with urlopen(url,timeout=90) as response:
    assert response.status==200
    data=response.read()
x=json.loads(data);assert x['model']['type']=='BPE' and len(x['model']['vocab'])==50257
path.write_bytes(data)
Path('tokenizer-source.json').write_text(json.dumps({'source':url,'bytes':len(data),'type':'BPE','vocab':50257,'source_family':'same Apache-2.0 pinned Xenova DistilGPT2 companion'},indent=2))
print('TOKENIZER_SOURCE_COMPLETE',len(data))
