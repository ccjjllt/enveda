"""Normalize one indexed lane with the unchanged parent chemistry and publish independently resumable blocks."""
import argparse
from collections import deque
from hashlib import sha256
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
import time


SOURCE = 'raw_parent_shard'
VERSION = 2
BLOCK_SIZE = 512
CHEM_SHA = '62e8464079bad7c4f7bd8b22c35c7d618445bfa224f2bafb40423328d82084eb'
RDKIT_VERSION = '2026.03.3'
RAW_FIELDS = ('raw_smiles','train','coconut_n')
COUNT_FIELDS = ('multicomponent_reduced','invalid_structures','valid_structures','valid_net_charged')
SOURCES = ('train','coconut')


def file_sha(path):
    h = sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20), b''):
            h.update(block)
    return h.hexdigest()


def write_receipt(out, receipt):
    tmp = out/'parent_shard_receipt.json.tmp'
    tmp.write_text(json.dumps(receipt,separators=(',',':'),allow_nan=False)+'\n')
    tmp.replace(out/'parent_shard_receipt.json')


def block_name(i):
    return f'blocks/block_{i:05d}.parquet'


def zero_counts():
    return {s:dict.fromkeys(COUNT_FIELDS,0) for s in SOURCES}


def describe_block(records, chemistry):
    texts, train, coconut = (records.column(k).to_pylist() for k in RAW_FIELDS)
    result = chemistry.describe_batch(texts)
    if len(result)!=len(texts):
        raise ValueError('Original parent normalization returned a different number of entries')
    counts, valid = zero_counts(), []
    for (row,salt),in_train,occurrences in zip(result,train,coconut):
        for name,weight in (('train',int(in_train)),('coconut',occurrences)):
            current = counts[name]
            current['multicomponent_reduced'] += int(salt)*weight
            if row is None:
                current['invalid_structures'] += weight
            else:
                current['valid_structures'] += weight
                current['valid_net_charged'] += int(row[-1]!=0)*weight
        if row is not None:
            src = (chemistry.TRAIN if in_train else 0) | (chemistry.COCONUT if occurrences else 0)
            valid.append(dict(zip(chemistry.PARENT_SCHEMA.names,(*row,src))))
    return valid,counts


def child(connection, lane_path, stage):
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    current = None
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        import rebuild_parents as chemistry
        if (chemistry.rdBase.rdkitVersion != RDKIT_VERSION
                or file_sha(Path(chemistry.__file__))!=CHEM_SHA):
            raise ValueError('Unexpected original chemistry implementation or RDKit version in worker')
        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
        chemistry.initialize()
        raw = pq.read_table(lane_path,columns=list(RAW_FIELDS),use_threads=False,pre_buffer=False)
        while True:
            current = connection.recv()
            if current is None:
                return
            records = raw.slice(BLOCK_SIZE*current,BLOCK_SIZE)
            valid,counts = describe_block(records,chemistry)
            name = block_name(current)
            target = stage/Path(name).name
            table = pa.Table.from_pylist(valid, schema=chemistry.PARENT_SCHEMA)
            pq.write_table(table,target,compression='zstd')
            metadata = dict(path=name,sha256=file_sha(target),input_rows=len(records),counts=counts,valid_rows=len(valid))
            connection.send(('done',current,metadata))
    except (EOFError,BrokenPipeError):
        return
    except Exception as e:
        try:
            connection.send(('failed',current,f'{type(e).__name__}: {e}'))
        except (EOFError,BrokenPipeError):
            pass
    finally:
        connection.close()


def lane_stats(path, original):
    """Validate sorted raw membership and compute independent per-block source multiplicities."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    if ([schema.field(k).type for k in RAW_FIELDS]!=[pa.string(),pa.bool_(),pa.int64()]
            or pf.metadata.num_rows!=original['rows']):
        raise ValueError('Raw lane schema or declared row count differs from the index')
    previous, stats = None, []
    for batch in pf.iter_batches(columns=list(RAW_FIELDS),batch_size=4096,use_threads=False):
        for text,train,n in zip(*(c.to_pylist() for c in batch.columns)):
            if (not isinstance(text,str) or not text.strip() or (previous is not None and text<=previous)
                    or type(train) is not bool or type(n) is not int or n<0 or not (train or n)):
                raise ValueError('Raw structure membership is unsorted, duplicated, missing, or invalid')
            previous = text
            if not stats or stats[-1]['input_rows']==BLOCK_SIZE:
                stats.append(dict(input_rows=0,train=0,coconut=0))
            s = stats[-1]
            s['input_rows']+=1
            s['train']+=int(train)
            s['coconut']+=n
    return stats


def check_block(i, entry, raw, path, parent_schema):
    import pyarrow.parquet as pq
    if (entry['path']!=block_name(i) or entry['input_rows']!=raw['input_rows']
            or type(entry['valid_rows']) is not int or not 0<=entry['valid_rows']<=raw['input_rows']
            or set(entry['counts'])!=set(SOURCES) or file_sha(path)!=entry['sha256']):
        raise ValueError(f'Invalid normalized block metadata or SHA256 for block {i}')
    pf = pq.ParquetFile(path)
    if not pf.schema_arrow.equals(parent_schema,check_metadata=False) or pf.metadata.num_rows!=entry['valid_rows']:
        raise ValueError(f'Invalid normalized block schema or row count for block {i}')
    for source in SOURCES:
        c = entry['counts'][source]
        if (set(c)!=set(COUNT_FIELDS) or any(type(v) is not int or v<0 for v in c.values())
                or c['valid_structures']+c['invalid_structures']!=raw[source]
                or c['valid_net_charged']>c['valid_structures'] or c['multicomponent_reduced']>raw[source]):
            raise ValueError(f'Invalid weighted {source} counts for block {i}')


def recover(previous, out, receipt, stats, parent_schema):
    old_path = previous/'parent_shard_receipt.json'
    old = json.loads(old_path.read_bytes())
    for key in ('source','version','index_sha256','lane','lanes','input_file','input_file_sha256',
                'input_rows','block_size','chem_source_sha256','rdkit'):
        if old[key]!=receipt[key]:
            raise ValueError(f'Previous checkpoint has different {key}')
    if old['status'] not in ('complete','partial') or not isinstance(old['blocks'],dict):
        raise ValueError('Previous checkpoint is not a completed or partial parent shard')
    if old['status']=='complete' and len(old['blocks'])!=len(stats):
        raise ValueError('Previous complete checkpoint omits normalized blocks')
    for key,entry in old['blocks'].items():
        i = int(key)
        if str(i)!=key or not 0<=i<len(stats) or entry['path']!=block_name(i):
            raise ValueError('Previous checkpoint has noncanonical block numbers or paths')
        dest = out/block_name(i)
        shutil.copyfile(previous/block_name(i),dest)
        check_block(i,entry,stats[i],dest,parent_schema)
        receipt['blocks'][key] = entry
    receipt['previous_receipt_sha256'] = file_sha(old_path)
    receipt['recovered_blocks'] = len(receipt['blocks'])


def supervise(path, out, receipt, stats, schema, workers, deadline):
    remaining = deque(i for i in range(len(stats)) if str(i) not in receipt['blocks'])
    if not remaining:
        return 'all_blocks_completed'
    if time.monotonic()>=deadline:
        return 'soft_deadline'
    context = mp.get_context('spawn')
    children, assigned = {}, {}
    with TemporaryDirectory(prefix='.parent-workers-',dir=out) as temp:
        stage = Path(temp)
        try:
            for _ in range(min(workers,len(remaining))):
                if time.monotonic()>=deadline:
                    break
                inbox, endpoint = context.Pipe()
                process = context.Process(target=child,args=(endpoint,path,stage))
                process.start()
                endpoint.close()
                children[inbox] = process
                i = remaining.popleft()
                inbox.send(i)
                assigned[inbox] = i
            while assigned:
                remaining_time = deadline-time.monotonic()
                ready = wait(list(assigned),timeout=max(0,remaining_time))
                if not ready:
                    return 'soft_deadline'
                for inbox in ready:
                    expected = assigned.pop(inbox)
                    try:
                        state,i,entry = inbox.recv()
                    except (EOFError,OSError) as err:
                        state,i,entry = 'failed',expected,f'Worker exited: {err}'
                    if state!='done' or i!=expected:
                        receipt.setdefault('worker_errors',{})[str(expected)] = str(entry)[:1000]
                        # Other processes can complete independent blocks; retain every already published result.
                        continue
                    src = stage/Path(block_name(i)).name
                    check_block(i,entry,stats[i],src,schema)
                    src.replace(out/block_name(i))
                    receipt['blocks'][str(i)] = entry
                    write_receipt(out,receipt)
                    if remaining and time.monotonic()<deadline:
                        next_id = remaining.popleft()
                        try:
                            inbox.send(next_id)
                        except (BrokenPipeError,OSError) as err:
                            receipt.setdefault('worker_errors',{})[str(next_id)] = f'Worker exited: {err}'
                        else:
                            assigned[inbox] = next_id
                # At the cutoff, poll once more to admit other already-finished results before stopping.
            if receipt.get('worker_errors'):
                return 'worker_error'
            return 'soft_deadline' if remaining else 'all_blocks_completed'
        finally:
            # Never join an unfinished chemistry call at the deadline: RDKit can be inside native code.
            for inbox,process in children.items():
                if process.is_alive():
                    process.kill()
                inbox.close()
            for process in children.values():
                process.join(timeout=2)
                if not process.is_alive():
                    process.close()


def run(index, lane, out, workers=4, soft_seconds=6900, previous=None):
    start = time.monotonic()
    if not 1<=workers<=4 or not 0<soft_seconds<float('inf'):
        raise ValueError('Expected 1–4 workers and a positive finite soft deadline')
    soft = min(float(soft_seconds),3*3600-90)
    chem_path = Path(__file__).resolve().with_name('rebuild_parents.py')
    if file_sha(chem_path)!=CHEM_SHA:
        raise ValueError('Expected the unchanged original rebuild_parents.py')
    import rebuild_parents as chemistry
    if (Path(chemistry.__file__).resolve()!=chem_path or chemistry.rdBase.rdkitVersion!=RDKIT_VERSION
            or chemistry.BATCH_SIZE!=BLOCK_SIZE):
        raise ValueError('Expected original RDKit and batch size')
    index = Path(index)
    manifest_path = index/'parent_raw_index.json' if index.is_dir() else index
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    lanes = manifest['lanes']
    if manifest['version']!=VERSION or type(lanes) is not int or not 0<=lane<lanes:
        raise ValueError('Wrong parent raw-index version or lane')
    names = {}
    for name in manifest['files']:
        match = re.fullmatch(r'raw_(\d+)\.parquet',name)
        if not match:
            raise ValueError(f'Unexpected raw-index lane filename: {name}')
        number = int(match.group(1))
        if number in names:
            raise ValueError('Duplicate raw lane in index')
        names[number] = name
    if set(names)!=set(range(lanes)):
        raise ValueError('Incomplete raw lane inventory')
    name = names[lane]
    source = manifest['files'][name]
    path = manifest_path.parent/name
    if file_sha(path)!=source['sha256']:
        raise ValueError('Input raw lane differs from the frozen raw inventory')
    stats = lane_stats(path,source)
    out = Path(out)
    if previous is not None and Path(previous).resolve()==out.resolve():
        raise ValueError('The previous checkpoint and fresh output must be separate')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'parent_shard_receipt.json').exists():
        raise FileExistsError('An existing shard receipt is present; resume it into a fresh output directory')
    (out/'blocks').mkdir()
    receipt = dict(source=SOURCE,version=VERSION,index_source=manifest.get('source'),
                   index_sha256=sha256(manifest_bytes).hexdigest(),lane=lane,lanes=lanes,input_file=name,
                   input_file_sha256=source['sha256'],input_rows=source['rows'],rdkit=RDKIT_VERSION,
                   block_size=BLOCK_SIZE,chem_source_sha256=CHEM_SHA,total_blocks=len(stats),
                   workers=workers,soft_seconds=soft,recovered_blocks=0,blocks={},status='partial')
    if previous is not None:
        recover(Path(previous),out,receipt,stats,chemistry.PARENT_SCHEMA)
    write_receipt(out,receipt)
    reason = supervise(path,out,receipt,stats,chemistry.PARENT_SCHEMA,workers,start+soft)
    receipt['status'] = 'complete' if len(receipt['blocks'])==len(stats) else 'partial'
    receipt['stop_reason'] = reason
    receipt['elapsed_seconds'] = round(time.monotonic()-start,3)
    write_receipt(out,receipt)
    print(f"Parent lane {lane}/{lanes}: {receipt['status']} ({len(receipt['blocks'])}/{len(stats)} blocks)",flush=True)
    return receipt


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=Path,required=True)
    parser.add_argument('--lane',type=int,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--soft-seconds',type=float,default=6900)
    parser.add_argument('--previous',type=Path)
    run(**vars(parser.parse_args()))
