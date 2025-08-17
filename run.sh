#!/bin/bash
#SBATCH --job-name=solver_itac
#SBATCH --output=itac.out
#SBATCH --error=itac.err
#SBATCH --partition=M7
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --time=00:09:00
#SBATCH --hint=nomultithread

set -euo pipefail

######## 0) 环境 ########
source /pxe/opt/spack/share/spack/setup-env.sh
spack load intel-oneapi-mpi
spack load intel-oneapi-itac

# OpenMP 线程与绑定
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Intel MPI 进程绑定
export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN=omp
unset I_MPI_PMI_LIBRARY
export I_MPI_HYDRA_BOOTSTRAP=slurm

######## 1) 路径解析 ########
WORKDIR="${SLURM_SUBMIT_DIR:-$PWD}"   # 你 sbatch 的目录（建议为 src/lab4）
BIN_REL="./build/bicgstab"
INPUT_REL="${INPUT_FILE:-data/case_6001.bin}"

# 转成绝对路径
[[ "${BIN_REL}"   = /* ]] && BIN_ABS="${BIN_REL}"     || BIN_ABS="${WORKDIR}/${BIN_REL}"
[[ "${INPUT_REL}" = /* ]] && INPUT_ABS="${INPUT_REL}" || INPUT_ABS="${WORKDIR}/${INPUT_REL}"

[[ -x "${BIN_ABS}"  ]] || { echo "ERROR: 程序不存在或不可执行：${BIN_ABS}"; exit 1; }
[[ -f "${INPUT_ABS}" ]] || { echo "ERROR: 输入文件不存在：${INPUT_ABS}"; exit 1; }

######## 2) 结果目录（你手动指定；必须是绝对路径） ########
# 用法示例（提交前在命令行指定 SAVE_DIR）：
#   SAVE_DIR="$PWD/itac_res_$(date +%Y%m%d_%H%M%S)" sbatch run.sh
#   # 或者指定到共享盘：
#   # SAVE_DIR="/work/hpc101/$USER/itac_res_$(date +%Y%m%d_%H%M%S)" sbatch run.sh
: "${SAVE_DIR:?请用 SAVE_DIR=/绝对/路径 指定输出目录，例如：SAVE_DIR=\"$PWD/itac_res_$(date +%Y%m%d_%H%M%S)\" sbatch run.sh}"

OUT_DIR="${SAVE_DIR%/}/trace"
[[ "${OUT_DIR}" = /* ]] || { echo "ERROR: SAVE_DIR 必须是绝对路径：${OUT_DIR}"; exit 1; }
mkdir -p "${OUT_DIR}"

# ITAC：单文件 .stf，GUI 直接打开
export VT_LOGFILE_PREFIX="${OUT_DIR}"
export VT_LOGFILE_FORMAT=STFSINGLE

echo "== 工作目录      : ${WORKDIR}"
echo "== 可执行文件    : ${BIN_ABS}"
echo "== 输入文件      : ${INPUT_ABS}"
echo "== ITAC 输出目录 : ${OUT_DIR}"

######## 3) 运行 ########
RANKS=$(( SLURM_JOB_NUM_NODES * SLURM_NTASKS_PER_NODE ))
echo "MPI ranks=${RANKS}, OMP threads/rank=${OMP_NUM_THREADS}"

mpirun -n "${RANKS}" \
  -genv VT_LOGFILE_PREFIX "${VT_LOGFILE_PREFIX}" \
  -genv VT_LOGFILE_FORMAT "${VT_LOGFILE_FORMAT}" \
  -trace "${BIN_ABS}" "${INPUT_ABS}"

######## 4) 收尾：打印 .stf 路径 ########
echo "== 运行完成，列出输出目录 =="
ls -lh "${OUT_DIR}" || true

STF="$(find "${OUT_DIR}" -maxdepth 1 -name '*.stf' -print -quit || true)"
if [[ -n "${STF}" ]]; then
  echo "ITAC trace: ${STF}"
  echo "可用命令：traceanalyzer \"${STF}\""
else
  echo "WARN: 未找到 .stf 文件。若文件很小/打不开，多半是计算节点无法写入 ${OUT_DIR}。"
fi
