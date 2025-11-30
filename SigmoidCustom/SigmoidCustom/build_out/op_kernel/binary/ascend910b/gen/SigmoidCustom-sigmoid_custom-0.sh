#!/bin/bash
echo "[Ascend910B1] Generating SigmoidCustom_a3c9eb1f1b227778957282b95ed93786 ..."
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
res=$(opc $1 --main_func=sigmoid_custom --input_param=/home/ICT/SigmoidCustom/SigmoidCustom/build_out/op_kernel/binary/ascend910b/gen/SigmoidCustom_a3c9eb1f1b227778957282b95ed93786_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/SigmoidCustom_a3c9eb1f1b227778957282b95ed93786.json ; then
  echo "$2/SigmoidCustom_a3c9eb1f1b227778957282b95ed93786.json not generated!"
  exit 1
fi

if ! test -f $2/SigmoidCustom_a3c9eb1f1b227778957282b95ed93786.o ; then
  echo "$2/SigmoidCustom_a3c9eb1f1b227778957282b95ed93786.o not generated!"
  exit 1
fi
echo "[Ascend910B1] Generating SigmoidCustom_a3c9eb1f1b227778957282b95ed93786 Done"
