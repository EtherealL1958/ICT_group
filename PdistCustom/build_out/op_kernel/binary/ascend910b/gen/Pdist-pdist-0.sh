#!/bin/bash
echo "[Ascend910B1] Generating Pdist_16996420756afb19140baf25c1330ba0 ..."
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=1

while true; do
  case "$1" in
    --kernel-src=*)
      export BUILD_KERNEL_SRC=$(echo "$1" | cut -d"=" -f2-)
      shift
      ;;
    -*)
      shift
      ;;
    *)
      break
      ;;
  esac
done
res=$(opc $1 --main_func=pdist --input_param=/home/ICT/PdistCustom/build_out/op_kernel/binary/ascend910b/gen/Pdist_16996420756afb19140baf25c1330ba0_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/Pdist_16996420756afb19140baf25c1330ba0.json ; then
  echo "$2/Pdist_16996420756afb19140baf25c1330ba0.json not generated!"
  exit 1
fi

if ! test -f $2/Pdist_16996420756afb19140baf25c1330ba0.o ; then
  echo "$2/Pdist_16996420756afb19140baf25c1330ba0.o not generated!"
  exit 1
fi
echo "[Ascend910B1] Generating Pdist_16996420756afb19140baf25c1330ba0 Done"
