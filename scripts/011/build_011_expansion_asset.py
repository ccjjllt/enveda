#!/usr/bin/env python3
"""手动构建011私有离线扩库资产；只读固定公开结构索引，不读取query标签来选候选。

mb539为008固定来源版本名。逐条生产身份核对剔除1个空InChIKey/Markush组，
实际有效538条，原顺序保留。命名空间索引与canonical身份分开，避免base raw-key碰撞。
仅本地构建，不上传、不提交、不训练、不创建测试。
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,re,sys,time,zipfile
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem,RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Descriptors import ExactMolWt
import rdkit
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from chemistry import canonical_group_key
from candidate_expansion import fp_bits_for_smiles
RDLogger.DisableLog('rdApp.warning');RDLogger.DisableLog('rdApp.error')
PREFIX='mb539';DATASET='ccjjllt/casmi26-massbank539-expansion'
ALLOWED={'CC0','CC BY','CC BY-SA','MIT'}
EXPECTED_BITS_SHA='a61e372fc41091bf5c7467e08134081266262665077cb9e2c56d73e4887f3013'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def field(text,name):
 m=re.search(r'^'+re.escape(name)+r':\s*(.+)$',text,re.M);return m.group(1).strip() if m else None

def previous_window_effect(excluded):
 """Read completed window outputs only AFTER fixed structural exclusion; no selection uses these."""
 result={};badkeys={x['canonical_key'] for x in excluded}
 p=ROOT/'artifacts/runs/009-expansion/research_interference.json'
 if p.exists():
  for ds,doc in read(p).items():
   hits=[r['key'] for r in doc['rows'] if badkeys&set(r['added'])]
   result[ds]={'n_queries':len(doc['rows']),'excluded_structure_window_hits':hits,'previous_scores_unaffected_by_exclusion':not hits}
 p=ROOT/'artifacts/runs/009-expansion/r3_source_sensitivity.json'
 if p.exists():
  rows=read(p)['rows'];hits=[r['key'] for r in rows if badkeys&set(r['new_candidates'])]
  result['r3_50']={'n_queries':len(rows),'excluded_structure_window_hits':hits,'previous_scores_unaffected_by_exclusion':not hits}
 return result


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'artifacts/runs/011-expansion-asset');args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True);t=time.monotonic()
 indexdir=ROOT/'artifacts/runs/008-expansion';entries=[json.loads(line) for line in (indexdir/'candidate_index_records.jsonl').read_text().splitlines()];meta=read(indexdir/'candidate_index_meta.json');prov009=read(ROOT/'artifacts/runs/009-expansion/massbank_complete_provenance.json')
 maps=[];map_paths=[]
 for name in ['train','coco']:
  path=ROOT/f'artifacts/runs/005/structure_identity_{name}.parquet';tab=pq.read_table(path,columns=['raw_key','canonical_group_key']).to_pydict();maps.append(dict(zip(tab['raw_key'],tab['canonical_group_key'])));map_paths.append(path)
 basecanonical=set(maps[0].values())|set(maps[1].values());baseraw=set(maps[0])|set(maps[1]);original=[(i,e) for i,e in enumerate(entries) if e['canonical_key'] not in basecanonical]
 if len(original)!=539:raise ValueError('Frozen539 selection changed: '+str(len(original)))
 bits_path=ROOT/'artifacts/runs/005/coco/fp_bits.npy';bits=np.load(bits_path,allow_pickle=False)
 if sha(bits_path)!=EXPECTED_BITS_SHA or bits.shape!=(6930,):raise ValueError('FP selected-bit mapping changed')
 valid=[];excluded=[];index_map=[];collisions=[];canon_cache={}
 for original_i,(source_i,e) in enumerate(original):
  mol=Chem.MolFromSmiles(e['smiles_rep']);actual=canonical_group_key(e['smiles_rep']) if mol is not None else None;canon_cache[e['smiles_rep']]=actual;reasons=[]
  if mol is None:reasons.append('representative SMILES cannot be parsed')
  if not actual or not re.fullmatch('[A-Z]{14}',actual):reasons.append('canonical InChIKey14 is empty or invalid; no definite structure identity')
  if mol is not None and any(a.GetAtomicNum()==0 for a in mol.GetAtoms()):reasons.append('Markush/wildcard atom: not a fully specified molecular structure')
  if actual!=e['canonical_key']:reasons.append('representative canonical identity does not match frozen source index')
  if reasons:
   item={'original_new_index':original_i,'source_index_8831':source_i,**e,'reasons':reasons};excluded.append(item);index_map.append({'original_new_index':original_i,'source_index_8831':source_i,'usable_index':None,'canonical_key':e['canonical_key'],'excluded_reasons':reasons});continue
  mass=float(ExactMolWt(mol));formula=rdMolDescriptors.CalcMolFormula(mol)
  if abs(mass-e['mass_theo'])>1e-10 or formula!=e['formula_calc']:raise ValueError('Frozen mass/formula changed for '+actual)
  key='MB539_'+actual
  if key in baseraw or actual in basecanonical:raise ValueError('New key or canonical overlaps base')
  if actual in baseraw:collisions.append({'canonical_key':actual,'base_raw_key':actual,'base_canonical_values':sorted({mp[actual] for mp in maps if actual in mp}),'asset_key':key,'resolution':'namespaced asset index; derive canonical from representative SMILES or explicit manifest map'})
  item={'asset_index':len(valid),'asset_key':key,'original_new_index':original_i,'source_index_8831':source_i,**e};valid.append(item);index_map.append({'original_new_index':original_i,'source_index_8831':source_i,'usable_index':len(valid)-1,'asset_key':key,'canonical_key':actual})
 if len(valid)!=538 or len(excluded)!=1 or excluded[0]['canonical_key']!='':raise ValueError('Unexpected validity exclusions; inspect without silently changing fixed candidates')
 print('original539 -> usable538; excluded1 empty canonical/Markush group',flush=True)
 # Preserve the full per-record attribution of included and quarantined entries. No spectra are copied.
 desired={}
 for e in valid+excluded:
  key=e['canonical_key'];accessions=prov009['canonical'][key]['accessions']
  if len(accessions)!=e['n_accessions_total']:raise ValueError('Provenance count mismatch for '+key)
  for acc in accessions:
   if acc in desired and desired[acc]!=key:raise ValueError('Accession belongs to conflicting canonical groups')
   desired[acc]=key
 records={k:[] for k in {e['canonical_key'] for e in valid+excluded}};bulk=Path(meta['snapshot']['bulk_zip'])
 provenance_mode='bulk_records_reparsed' if bulk.exists() else 'restored_from_008_unique_license_sets_and_009_complete_accessions'
 if not bulk.exists():
  if meta['snapshot']['sha256']!=prov009['bulk_sha256']:raise ValueError('Preserved source snapshot hashes disagree')
  license_sets={e['canonical_key']:e['licenses_set'] for e in valid+excluded}
  if any(len(v)!=1 for v in license_sets.values()):raise ValueError('Missing bulk and ambiguous per-record licenses; cannot infer')
  for acc,key in desired.items():
   contributor=prov009['source_by_accession'][acc]
   records[key].append({'accession':acc,'source_contributor':contributor,'record_url':'https://massbank.eu/MassBank-api/records/'+acc,'source_zip_entry':meta['snapshot']['zip_top_prefix']+'/'+contributor+'/'+acc+'.txt','license_verbatim':license_sets[key][0],'license_evidence':'All original allowed records in this canonical group share this one LICENSE value in saved 008 index; complete accession membership is from saved 009 provenance.','canonical_key':key,'attribution_note':'Original record URL and contributor retained. Author/copyright/raw record content were not preserved in 008/009; no missing fields are invented.','provenance_mode':provenance_mode})
 else:
  if sha(bulk)!=meta['snapshot']['sha256'] or sha(bulk)!=prov009['bulk_sha256']:raise ValueError('Source bulk SHA changed')
  with zipfile.ZipFile(bulk) as z:
   for name in z.namelist():
    if not name.endswith('.txt') or Path(name).stem not in desired:continue
    text=z.read(name).decode('utf-8',errors='strict');acc=field(text,'ACCESSION');smi=field(text,'CH$SMILES');lic=field(text,'LICENSE');expected=desired[acc]
    if lic not in ALLOWED:raise ValueError('Unaccepted source license '+str(lic))
    actual=canon_cache.get(smi)
    if smi not in canon_cache:actual=canonical_group_key(smi);canon_cache[smi]=actual
    if actual!=expected:raise ValueError('Record identity mismatch '+acc)
    records[expected].append({'accession':acc,'source_contributor':name.split('/')[-2],'record_url':'https://massbank.eu/MassBank-api/records/'+acc,'source_zip_entry':name,'record_sha256':hashlib.sha256(text.encode('utf-8')).hexdigest(),'license_verbatim':lic,'authors':field(text,'AUTHORS'),'copyright':field(text,'COPYRIGHT'),'publication':field(text,'PUBLICATION'),'date_verbatim':field(text,'DATE'),'smiles_source':smi,'formula_source':field(text,'CH$FORMULA'),'canonical_key':actual})
 for k,rs in records.items():
  if {r['accession'] for r in rs}!={a for a,c in desired.items() if c==k}:raise ValueError('Missing provenance records '+k)
  rs.sort(key=lambda r:r['accession'])
 keys=np.asarray([e['asset_key'] for e in valid],dtype='<U20');masses=np.asarray([e['mass_theo'] for e in valid],dtype=np.float64);selected=[]
 for e in valid:
  fp,formula,mass=fp_bits_for_smiles(e['smiles_rep'],bits)
  if formula!=e['formula_calc'] or abs(mass-e['mass_theo'])>1e-10:raise ValueError('Fingerprint representation changed')
  selected.append(fp)
 selected=np.asarray(selected,dtype=np.uint8);packed=np.packbits(selected,axis=1,bitorder='big')
 if packed.shape!=(538,867) or selected.shape!=(538,6930):raise ValueError('Invalid fingerprint shape')
 if not np.array_equal(np.unpackbits(packed,axis=1,bitorder='big')[:,:6930],selected):raise ValueError('Pack/unpack identity failed')
 if np.any(packed[:,-1]&0b00111111):raise ValueError('Nonzero padding bits')
 if len(set(keys.tolist()))!=538 or len({e['canonical_key'] for e in valid})!=538:raise ValueError('Duplicate identity')
 np.save(out/(PREFIX+'_keys.npy'),keys,allow_pickle=False);np.save(out/(PREFIX+'_masses.npy'),masses,allow_pickle=False);np.save(out/(PREFIX+'_fp.npy'),packed,allow_pickle=False);(out/(PREFIX+'_smiles.txt')).write_text(''.join(e['smiles_rep']+'\n' for e in valid))
 provenance={'dataset_id':DATASET,'source_snapshot':meta['snapshot'],'provenance_mode':provenance_mode,'asset_changes':'Fixed canonical index subset; representative molecules unchanged; RDKit exact masses and v9-selected fingerprints derived; one empty canonical/Markush group excluded; opaque namespaced asset IDs introduced to avoid raw-key collisions. No spectra, competition query data, or model weights included.','license_policy':'Per-record licenses and attribution preserved below. Collection-level Kaggle license CC-BY-SA-4.0 is conservative for mixed CC BY, CC BY-SA, CC0 sources; it does not erase original source terms or invent versions missing from record LICENSE fields.','structures':[{**e,'records':records[e['canonical_key']]} for e in valid],'excluded_structure_groups':[{**e,'records':records[e['canonical_key']]} for e in excluded]}
 write(out/(PREFIX+'_provenance.json'),provenance)
 license_counts=Counter(e['license_rep'] for e in valid);record_licenses=Counter(r['license_verbatim'] for e in valid for r in records[e['canonical_key']]);source_counts=Counter(r['source_contributor'] for e in valid for r in records[e['canonical_key']])
 description=('Private offline CASMI26 research input: the frozen MassBank539 source version, with 538 valid structures after excluding one empty-canonical Markush group. Includes only public source structures, exact masses, derived packed fingerprints and attribution; no competition spectra or model weights. Sources: MassBank-data 2026.03, DOI https://doi.org/10.5281/zenodo.19073053 . Structure licenses: CC BY 479; CC BY-SA 49; CC0 10. Dataset-level collection license is CC BY-SA 4.0; individual source records retain their original LICENSE statements and source attribution as listed in mb539_provenance.json. LICENSE fields without a version are preserved verbatim. This package adds identifiers and derived fingerprints and does not claim all records are CC0. See LICENSES.md and mb539_manifest.json for derivation, exclusion and ordering details.')
 write(out/'dataset-metadata.json',{'id':DATASET,'title':'CASMI26 MassBank539 expansion - 538 valid','description':description,'licenses':[{'name':'CC-BY-SA-4.0'}]})
 (out/'LICENSES.md').write_text('# Source licenses and attribution\n\nThis private research collection uses the Kaggle dataset-level label **CC BY-SA 4.0**. It contains mixed source terms, preserved record by record in `mb539_provenance.json`: 479 structures with CC BY records, 49 with CC BY-SA records, and 10 with CC0 records. It is not an all-CC0 dataset.\n\nSource: [MassBank-data 2026.03](https://doi.org/10.5281/zenodo.19073053). The source release reports CC BY 4.0; individual record LICENSE statements are retained and remain relevant. The package does not infer a license version when a source record only says `CC BY`, `CC BY-SA`, or `CC0`. Contributor and original record URLs are included in the provenance file. Where available, author and copyright fields are retained; missing original fields are not invented. The temporary bulk archive used in 008/009 may no longer be present; provenance reconstruction mode is explicitly recorded.\n\nChanges: canonical deduplication inherited from the fixed 008 index; v9 fingerprint derivation and exact-mass calculation; namespaced indexing; removal of one undefined Markush/empty-InChIKey group. No spectrum peak arrays, competition query records or model weights are distributed.\n\nThe legacy `mb539` prefix identifies the fixed source version; the usable asset has **538 rows**. The excluded group and original-to-usable index mapping are documented rather than silently replaced.\n')
 (out/'README.md').write_text('# CASMI26 MassBank539 expansion asset\n\nPrivate dataset input; do not create with a public-upload flag. The legacy source version contains 539 entries, of which 538 pass definite-structure identity checks.\n\nLoad `mb539_keys.npy`, `mb539_masses.npy`, and `mb539_fp.npy` with `allow_pickle=False`. Read one representative SMILES per line from `mb539_smiles.txt`. All four files have matching row order, inherited from the 008 mass-sorted index without further sorting.\n\nKeys are opaque namespaced IDs `MB539_<canonical InChIKey14>`; never treat them as already-normalized chemical keys. Obtain canonical identity from the corresponding SMILES using the existing pipeline, or from `mb539_manifest.json` (`canonical_keys` / `raw_to_canonical`). A source canonical key collides with an existing base raw key, so directly appending unnamespaced keys is unsafe.\n\nMasses: float64 RDKit ExactMolWt. Fingerprints: uint8 [538,867], np.packbits(bitorder="big"), 6930 selected bits from v9 10407-bit concatenation; six padding bits in the final byte are zero. Unpack using np.unpackbits(axis=1,bitorder="big")[:,:6930].\n\nSource licensing and attribution are documented in LICENSES.md and mb539_provenance.json. No query labels were used to select candidates. No model weights or competition spectra are included.\n')
 file_names=[PREFIX+'_keys.npy',PREFIX+'_masses.npy',PREFIX+'_fp.npy',PREFIX+'_smiles.txt',PREFIX+'_provenance.json','dataset-metadata.json','LICENSES.md','README.md']
 manifest={'version':'011-massbank539-valid538-v1','dataset_id':DATASET,'visibility_required':'private','upload_note':'Local build only. Kaggle create must remain private (do not use --public). Dataset metadata alone is not a visibility guarantee.','counts':{'original_new':539,'usable':538,'excluded':1},'original_new_entries':539,'usable_entries':538,'excluded_entries':1,'excluded_original_indices':[e['original_new_index'] for e in excluded],'original_to_usable':index_map,'canonical_keys':[e['canonical_key'] for e in valid],'raw_to_canonical':{e['asset_key']:e['canonical_key'] for e in valid},'raw_key_collisions_avoided':collisions,'shapes':{'keys':[538],'masses':[538],'fp':[538,867]},'dtypes':{'keys':str(keys.dtype),'masses':str(masses.dtype),'fp':str(packed.dtype)},'fingerprint_contract':{'full_bits':10407,'selected_bits':6930,'packed_bytes':867,'bitorder':'big','padding_zero_bits':6,'fp_bits_sha256':EXPECTED_BITS_SHA,'definition':'Morgan radius2 4096 + Morgan radius3 4096 + RDKitFP maxPath6 2048 + MACCS167; index using unchanged v9 fp_bits.npy'},'identity_rule':'canonical_group_key: remove stereochemistry -> RDKit TautomerEnumerator.Canonicalize -> InChIKey14; required 14 uppercase letters and no wildcard atoms','ordering':'Filter original 008 index by base canonical absence, then remove invalid identity entries; preserve remaining order. No query labels or score-based selection.','mass_rule':'RDKit ExactMolWt float64 of unchanged representative SMILES; no neutralization/salt removal','license_counts_structures':dict(license_counts),'license_counts_records':dict(record_licenses),'source_counts_records':dict(source_counts),'included_source_records':sum(record_licenses.values()),'excluded_source_records':sum(len(records[e['canonical_key']]) for e in excluded),'exclusion_effect_on_previous_research':previous_window_effect(excluded),'manual_checks':{'canonical_identity_mismatches':0,'mass_formula_mismatches':0,'source_record_canonical_mapping_consistent':True,'source_record_content_reparsed':bulk.exists(),'keys_unique':True,'new_vs_base_canonical_overlap':0,'namespaced_keys_vs_base_raw_overlap':0,'pack_unpack_bit_mismatches':0,'nonzero_padding_bits':0},'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [indexdir/'candidate_index_records.jsonl',ROOT/'artifacts/runs/009-expansion/massbank_complete_provenance.json',bits_path,*map_paths]},'source_bulk_sha256':meta['snapshot']['sha256'],'source_provenance_mode':provenance_mode,'files':{name:{'bytes':(out/name).stat().st_size,'sha256':sha(out/name)} for name in file_names},'environment':{'python':sys.version.split()[0],'numpy':np.__version__,'rdkit':rdkit.__version__},'created_utc':datetime.now(timezone.utc).isoformat(),'seconds':time.monotonic()-t,'builder_sha256':sha(Path(__file__))}
 write(out/(PREFIX+'_manifest.json'),manifest)
 print(json.dumps({'output':str(out),'original':539,'usable':538,'fingerprint_shape':list(packed.shape),'manifest_sha256':sha(out/(PREFIX+'_manifest.json')),'raw_collisions_avoided':collisions,'license_counts_structures':dict(license_counts),'license_counts_records':dict(record_licenses),'included_records':manifest['included_source_records'],'excluded_records':manifest['excluded_source_records'],'exclusion_effect':manifest['exclusion_effect_on_previous_research'],'seconds':manifest['seconds']},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
