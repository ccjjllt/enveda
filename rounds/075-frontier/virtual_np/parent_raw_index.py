"""Index exact raw SMILES once, preserving COCONUT multiplicities and train deduplication."""
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

from rebuild_parents import checksum, coconut_smiles, training_smiles, verified_release


RAW_SCHEMA=pa.schema([('raw_smiles',pa.string()),('train',pa.bool_()),('coconut_n',pa.int64())])


def build(train, archive, source_receipt, out, lanes):
    if lanes<1:raise ValueError('Choose at least one independent CPU lane')
    started=time.monotonic()
    release=verified_release(archive,source_receipt)
    csv.field_size_limit(sys.maxsize)
    train_counts,coconut_counts=Counter(),Counter()
    # v1 ran each distinct training SMILES once; its COCONUT counts instead include every valid source record.
    training=set(training_smiles(train,train_counts))
    coconut=Counter(coconut_smiles(archive,release,coconut_counts))
    text_by_lane=[[] for _ in range(lanes)]
    for text in training|coconut.keys():
        n=int.from_bytes(sha256(text.encode()).digest()[:8],'big')%lanes
        text_by_lane[n].append(text)
    out.mkdir(parents=True,exist_ok=True)
    files={}
    for i,items in enumerate(text_by_lane):
        target=out/f'raw_{i:02}.parquet'
        rows=sorted(items)
        with pq.ParquetWriter(target,RAW_SCHEMA,compression='zstd') as writer:
            for pos in range(0,len(rows),32768):
                block=rows[pos:pos+32768]
                table=pa.Table.from_arrays([pa.array(block),pa.array([s in training for s in block]),
                                            pa.array([coconut.get(s,0) for s in block],type=pa.int64())],schema=RAW_SCHEMA)
                writer.write_table(table)
        files[target.name]=dict(sha256=checksum(target),rows=len(rows))
    total=sum(v['rows'] for v in files.values())
    report=dict(version=2,status='complete',source='structure_only_train_and_coconut_raw_index',lanes=lanes,
                partition='First 8 bytes of SHA256(UTF-8 original raw SMILES) modulo lanes; lexicographically sorted within each lane.',
                train_sha256=checksum(train),coconut_source=release,coconut_source_receipt_sha256=checksum(source_receipt),
                raw_source_counts={'train':dict(train_counts),'coconut':dict(coconut_counts)},
                distinct_raw_smiles={'train':len(training),'coconut':len(coconut),'union':total,
                                     'shared':len(training)+len(coconut)-total},
                files=files,seconds=time.monotonic()-started)
    if (train_counts['source_rows']-train_counts['missing_smiles']-train_counts['duplicate_raw_smiles']!=len(training)
            or coconut_counts['source_rows']-coconut_counts['missing_smiles']!=sum(coconut.values())):
        raise ValueError('Raw source multiplicities do not reconcile')
    (out/'parent_raw_index.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PARENT RAW INDEX',report['distinct_raw_smiles'],flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('train','archive','source-receipt','out'):p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--lanes',required=True,type=int)
    build(**vars(p.parse_args()))
