"""Reassemble completed CPU checkpoints with v1 source priority, schema, sorting and legacy comparison."""
from collections import Counter
import json
from pathlib import Path
import time

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import rdBase

import rebuild_parents as v1


def merge(index, shard_dirs, legacy, out):
    start=time.monotonic()
    config=json.loads(index.read_bytes())
    index_sha=v1.checksum(index)
    source_sha=v1.checksum(Path(v1.__file__))
    if config['version']!=2 or config['status']!='complete' or len(shard_dirs)!=config['lanes'] or rdBase.rdkitVersion!=v1.RDKit_VERSION:
        raise ValueError('Expected a complete index and one finished shard per lane')
    ids={}
    stats={source:Counter(c) for source,c in config['raw_source_counts'].items()}
    inputs={}
    for directory in shard_dirs:
        rpt_path=directory/'parent_shard_receipt.json'
        receipt=json.loads(rpt_path.read_bytes())
        lane=receipt['lane']
        target=f'raw_{lane:02}.parquet'
        if (receipt['version']!=2 or receipt['status']!='complete' or receipt['source']!='raw_parent_shard'
                or receipt['index_sha256']!=index_sha or receipt['chem_source_sha256']!=source_sha
                or receipt['lanes']!=config['lanes'] or lane in inputs or target not in config['files']
                or receipt['input_file_sha256']!=config['files'][target]['sha256']
                or receipt['input_rows']!=config['files'][target]['rows'] or receipt['block_size']!=512):
            raise ValueError(f'Incomplete, duplicated, or mismatched lane {lane}')
        size=receipt['block_size']
        n=receipt['input_rows']
        blocks=receipt['blocks']
        if set(blocks)!={str(i) for i in range((n+size-1)//size)}:
            raise ValueError(f'Missing or duplicate block in lane {lane}')
        for i in range(len(blocks)):
            info=blocks[str(i)]
            rel=f'blocks/block_{i:05}.parquet'
            path=directory/rel
            if (info['path']!=rel or info['input_rows']!=min(size,n-i*size)
                    or v1.checksum(path)!=info['sha256'] or pq.ParquetFile(path).metadata.num_rows!=info['valid_rows']):
                raise ValueError(f'Checkpoint block changed: {lane}/{i}')
            table=pq.read_table(path)
            if not table.schema.equals(v1.PARENT_SCHEMA,check_metadata=False):raise ValueError('Wrong checkpoint schema')
            for name in ('train','coconut'):stats[name].update(info['counts'][name])
            for row in zip(*(col.to_pylist() for col in table.columns)):
                bits=row[-1]; rec=row[:-1]; identity=(rec[0],rec[5])
                if bits not in (1,2,3):raise ValueError('Unknown provenance bit')
                previous=ids.get(identity)
                if previous is None:
                    ids[identity]=(bits,rec)
                else:
                    prior_bits,prior=previous
                    # This handles an earlier COCONUT-only shard even when its SMILES sorts first.
                    prefer_new=(not bool(bits&v1.TRAIN),rec[1]) < (not bool(prior_bits&v1.TRAIN),prior[1])
                    ids[identity]=(bits|prior_bits,rec if prefer_new else prior)
        inputs[lane]=dict(receipt_sha256=v1.checksum(rpt_path),blocks=len(blocks),input_rows=n)
    if set(inputs)!=set(range(config['lanes'])):raise ValueError('Missing lanes')
    for name,bit in [('train',v1.TRAIN),('coconut',v1.COCONUT)]:
        unique=sum(bool(bits&bit) for bits,_ in ids.values())
        stats[name]['duplicate_normalized_identity']=stats[name]['valid_structures']-unique
        admitted=(config['distinct_raw_smiles']['train'] if bit==v1.TRAIN
                  else stats[name]['source_rows']-stats[name]['missing_smiles'])
        if (stats[name]['duplicate_normalized_identity']<0 or admitted!=stats[name]['valid_structures']+stats[name]['invalid_structures']):
            raise ValueError(f'Unreconciled normalized source totals for {name}')
    overlap=v1.compare(ids,legacy)
    ordered=sorted(ids.values(),key=lambda entry:(entry[1][2],entry[1][0],entry[1][5]))
    counts=Counter(bits for bits,_ in ordered)
    out.mkdir(parents=True,exist_ok=True)
    target=out/'rebuilt_parent_meta.parquet'
    with pq.ParquetWriter(target,v1.PARENT_SCHEMA,compression='zstd') as writer:
        for offset in range(0,len(ordered),32768):
            rows=[dict(zip(v1.PARENT_SCHEMA.names,row+(bits,))) for bits,row in ordered[offset:offset+32768]]
            writer.write_table(pa.Table.from_pylist(rows,schema=v1.PARENT_SCHEMA))
    train_ids=counts[1]+counts[3]
    doc=dict(version=2,status='complete',rdkit=rdBase.rdkitVersion,coconut_release=config['coconut_source'],
             train_sha256=config['train_sha256'],source_input_counts={k:dict(v) for k,v in stats.items()},
             structure_counts=dict(counts),identities=len(ordered),legacy_overlap=overlap,output_sha256=v1.checksum(target),
             index_sha256=index_sha,chem_source_sha256=source_sha,lane_receipts=inputs,
             v1_train_count_check=dict(previous=274195,rebuilt=train_ids,matches=train_ids==274195),
             legacy_pool_sha256=v1.LEGACY_SHA256,inventory='Independent structure-only inventory',
             source_bits={'1':'competition train','2':'COCONUT','3':'both'},
             normalization=['v1 largest-fragment / unchanged charge, without numeric cutoffs',
                            'de-stereochemized parent SMILES, canonical tautomer InChIKey14 + heavy-element signature identity',
                            'lexicographically first training representative, otherwise lexicographically first COCONUT representative'],
             seconds=time.monotonic()-start)
    (out/'rebuilt_parent_receipt.json').write_text(json.dumps(doc,indent=2)+'\n')
    print('COMPLETED INDEPENDENT PARENT INVENTORY',len(ordered),'train',train_ids,flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',required=True,type=Path)
    p.add_argument('--shard-dirs',required=True,nargs='+',type=Path)
    p.add_argument('--legacy',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    merge(**vars(p.parse_args()))
