#!/bin/bash
echo "[Ascend910B1] Generating Pdist_bc0ddcf31cab9c23dc21a02108acf75a ..."
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
res=$(opc $1 --main_func=pdist --input_param=/home/ICT/PdistCustom/build_out/op_kernel/binary/ascend910b/gen/Pdist_bc0ddcf31cab9c23dc21a02108acf75a_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/Pdist_bc0ddcf31cab9c23dc21a02108acf75a.json ; then
  echo "$2/Pdist_bc0ddcf31cab9c23dc21a02108acf75a.json not generated!"
  exit 1
fi

if ! test -f $2/Pdist_bc0ddcf31cab9c23dc21a02108acf75a.o ; then
  echo "$2/Pdist_bc0ddcf31cab9c23dc21a02108acf75a.o not generated!"
  exit 1
fi
echo "[Ascend910B1] Generating Pdist_bc0ddcf31cab9c23dc21a02108acf75a Done"
