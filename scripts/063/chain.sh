#!/bin/bash
# 063 chain：循环推 kernel（GPU 会话上限时每 5 分钟重试）→ watch.sh（等完成→单文件拉取→审计→-k/-v 提交）
. /workspace/kenv.sh
NB=${NB:?}; K=${K:?}; D=${D:?}; O=${O:?}; MSG=${MSG:?}
END=$(( $(date +%s) + ${PUSH_HOURS:-24}*3600 ))
while [ $(date +%s) -lt $END ]; do
  if [ -n "$WAIT_FOR" ] && [ ! -e "$WAIT_FOR" ]; then sleep 300; continue; fi
  out=$(cd /workspace/enveda && timeout 300 kaggle kernels push -p notebooks/$NB 2>&1)
  echo "$(date '+%m-%d %H:%M') $out" >> $D/chain.log
  v=$(echo "$out" | grep -oP 'Kernel version \K[0-9]+')
  if [ -n "$v" ]; then echo $v > $D/version
    K=$K D=$D O=$O V=$v MSG="$MSG" HOURS=${HOURS:-30} exec /workspace/enveda/scripts/063/watch.sh > $D/watch.out 2>&1
  fi
  sleep 300
done
echo PUSH_TIMEOUT > $D/NOT_SUBMITTED
