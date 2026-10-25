"""036：复用 sanpier/pubchem-neural025-lb-0-466（corrected 0.465 基线 + 神经重排 RRF），
主提交 weight=0.25（上游公榜 0.466 配方），同次运行额外导出 0.15/0.35/0.50 变体文件。手动运行。"""
import json, hashlib, sys
from pathlib import Path
src = Path(sys.argv[1] if len(sys.argv) > 1 else '/workspace/pub/sanpier466/pubchem-neural025-lb-0-466.ipynb')
out = Path(__file__).resolve().parents[2] / 'notebooks/036'
nb = json.loads(src.read_text())
cells = nb['cells']
cells[1]['source'] = ["NEURAL_WEIGHT = 0.25\n", "RRF_OFFSET = 3.0\n", "VARIANT_WEIGHTS = (0.15, 0.35, 0.50)\n",
                      "assert NEURAL_WEIGHT == 0.25 and RRF_OFFSET == 3.0\n"]
s = ''.join(cells[9]['source'])
s = s.replace("assert NEURAL_WEIGHT == 0.15 and RRF_OFFSET == 3.0\n",
  "assert NEURAL_WEIGHT == 0.25 and RRF_OFFSET == 3.0\n"
  "# 036：先导出变体，最后写主 submission.csv，审计文件以主权重为准\n"
  "for _w in VARIANT_WEIGHTS:\n"
  "    apply(ROOT/'neural_candidate_features.parquet',ROOT/'submission_baseline0465.csv',\n"
  "        ROOT/f'submission_w{int(round(_w*100)):03d}.csv',weight=_w,offset=RRF_OFFSET)\n"
  "    print('variant written', _w, flush=True)\n")
s = s.replace("experiment='corrected_pubchem_fixed_neural_015'", "experiment='enveda036_corrected_pubchem_neural_025'")
assert 'VARIANT_WEIGHTS' in s
cells[9]['source'] = s.splitlines(True)
cells.insert(0, {'cell_type': 'markdown', 'metadata': {}, 'source': [
  "# enveda 036：corrected PubChem 0.465 基线 + 神经重排\n\n",
  "来源：[sanpier/pubchem-neural025-lb-0-466](https://www.kaggle.com/code/sanpier/pubchem-neural025-lb-0-466)，上游源码与致谢原样内嵌。",
  "本轮只把神经权重设为上游已评分的 0.25，并在同次运行额外导出 0.15/0.35/0.50 变体，供后续对照提交。\n"]})
for c in cells:
    if c['cell_type'] == 'code':
        c['outputs'] = []; c['execution_count'] = None
out.mkdir(parents=True, exist_ok=True)
p = out / 'enveda-036-neural-rrf.ipynb'
p.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
meta = json.loads((src.parent / 'kernel-metadata.json').read_text())
meta.update(id='ccjjllt/enveda-036-neural-rrf', title='enveda-036-neural-rrf', code_file=p.name, is_private=True)
meta['dataset_sources'] = [d for d in meta['dataset_sources'] if d]
meta.pop('id_no', None); meta['docker_image']=meta['docker_image'].replace('kaggle-private-byod','kaggle-gpu-images')
(out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False))
print(p, hashlib.sha256(src.read_bytes()).hexdigest(), hashlib.sha256(p.read_bytes()).hexdigest())
