#!/bin/bash
# 063 watcher（源自 k062a/run3.sh，审计改用系统 python3）：等 kernel 完成 → 单文件拉取 → 审计 → 提交
. /workspace/kenv.sh
K=${K:?}
V=${V:?}
D=${D:?}
O=${O:?}
MSG=${MSG:?}
LOG=$D/watch.log
END=$(( $(date +%s) + ${HOURS:-12}*3600 ))
echo "watch start $(date '+%F %T %Z') $K v$V" | tee -a "$LOG"
saw=0
while [ $(date +%s) -lt $END ]; do
  s=$(timeout 120 kaggle kernels status $K 2>&1 | tail -1)
  echo "$(date '+%m-%d %H:%M') $s" >> "$LOG"
  case "$s" in
    *RUNNING*|*QUEUED*|*PENDING*) saw=1 ;;
    *COMPLETE*) [ "$saw" = 1 ] && break ;;
    *ERROR*|*CANCEL*) echo FAIL | tee -a "$LOG"; echo FAIL > $D/NOT_SUBMITTED; exit 1 ;;
  esac
  sleep 120
done
mkdir -p "$O"
sleep 600  # 完成后先等 10 分钟再拉，避开 429 限流
for f in submission.csv apex_extlib_audit.json submission_neural.log; do
  for t in 1 2 3 4 5 6 7 8 9 10; do
    timeout 240 kaggle kernels output $K -p "$O" --file-pattern "^${f//./\\.}\$" >>"$LOG" 2>&1 || true
    [ -s "$O/$f" ] && break; sleep $((120*t))
  done
done
[ -s "$O/submission.csv" ] || { echo NO_OUTPUT | tee -a "$LOG"; echo NO_OUTPUT > $D/NOT_SUBMITTED; exit 2; }
export O; check=$(python3 - <<'PY'
import os,json,pandas as pd
O=os.environ['O']
a=pd.read_csv(f'{O}/submission.csv',dtype=str,keep_default_na=False); b=pd.read_csv('/workspace/kout_v36/submission.csv',dtype=str,keep_default_na=False)
ok=True; why=[]
if len(a)!=400: ok=False; why.append(f'rows={len(a)}')
L=a.smiles.str.split(';')
if L.str[0].nunique()<12: ok=False; why.append("placeholder")  # smoke: 12 visible + 1 placeholder = 13
if not L.map(lambda x: 1<=len(x)<=25 and len(set(x))==len(x) and all(s.strip() for s in x)).all(): ok=False; why.append('badlist')
if set(a.molecule_id)!=set(b.molecule_id): ok=False; why.append('ids')
m=a.merge(b,on='molecule_id',suffixes=('_n','_o'))
nt=int((m.smiles_n.str.split(';').str[0]!=m.smiles_o.str.split(';').str[0]).sum()); nf=int((m.smiles_n!=m.smiles_o).sum())
au={}
try: au=json.load(open(f'{O}/apex_extlib_audit.json'))
except Exception: why.append('no_audit')
print('DECISION','SUBMIT' if ok else 'SKIP','TOP1_VS_056',nt,'FULL_VS_056',nf,'extlib',json.dumps(au).replace(' ',''),'why',','.join(why) or '-')
PY
)
echo "$check" | tee -a "$LOG"; echo "$check" > $D/check.txt
echo "$check" | grep -q 'DECISION SUBMIT' || { echo SKIP > $D/NOT_SUBMITTED; exit 3; }
if [ -n "$NEURAL_CHECK" ]; then
  if [ ! -s "$O/submission_neural.log" ] || grep -qiE 'Traceback|Error' "$O/submission_neural.log"; then echo "NEURAL_FAIL" | tee -a "$LOG"; echo NEURAL_FAIL > $D/NOT_SUBMITTED; exit 4; fi
fi
if [ -n "$SKIP_IF" ] && [ -e "$SKIP_IF" ]; then echo "SKIP_IF $SKIP_IF exists -> hold" | tee -a "$LOG"; echo "HOLD sibling submitted" > $D/NOT_SUBMITTED; exit 0; fi
[ -n "$NO_SUBMIT" ] && { echo "NO_SUBMIT set" | tee -a "$LOG"; echo HOLD > $D/NOT_SUBMITTED; exit 0; }
for attempt in 1 2; do
  if kaggle competitions submit -c enveda-CASMI26-molecule-id-mass-spectra -k $K -v $V -f submission.csv -m "$MSG" >>"$LOG" 2>&1; then
    echo SUBMITTED | tee -a "$LOG"; echo "submitted $(date '+%F %T %Z') v$V" > $D/SUBMITTED; exit 0
  fi
  echo "submit failed $attempt" | tee -a "$LOG"
  target=$(date -d 'today 08:05' +%s); [ $(date +%s) -ge $target ] && target=$(date -d 'tomorrow 08:05' +%s)
  while [ $(date +%s) -lt $target ]; do sleep 240; done
done
echo fail > $D/NOT_SUBMITTED; exit 1
