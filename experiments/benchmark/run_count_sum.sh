echo "+----------------------------------------------------+"
echo "| Benchmarking count-sum aggregation query           |"
echo "+----------------------------------------------------+"
echo ""

script_dir=$(dirname "$0")

uv run python -m benchmark.count_sum.run \
  --load-from "${script_dir}/count_sum/scenarios" \
  --scenarios \
  100x2_5e3.yaml \
  100x2_5e4.yaml \
  200x4_5e3.yaml \
  200x4_5e4.yaml \
  400x8_5e3.yaml \
  400x8_5e4.yaml \
  800x16_5e3.yaml \
  800x16_5e4.yaml \
  1600x32_5e3.yaml \
  1600x32_5e4.yaml

uv run python -m benchmark.count_sum.plot \
  --load-from "${script_dir}/results" \
  --scenarios \
  100x2_5e3 \
  100x2_5e4 \
  200x4_5e3 \
  200x4_5e4 \
  400x8_5e3 \
  400x8_5e4 \
  800x16_5e3 \
  800x16_5e4 \
  1600x32_5e3 \
  1600x32_5e4
