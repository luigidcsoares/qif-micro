script_dir=$(dirname "$0")

sh "${script_dir}/run_generic.sh"
echo ""
sh "${script_dir}/run_count_sum.sh"
