echo "+----------------------------------------------------+"
echo "| Benchmarking random response + geometric noise     |"
echo "+----------------------------------------------------+"
echo ""

script_dir=$(dirname "$0")

uv run python -m benchmark.generic.run \
  --load-from "${script_dir}/generic/scenarios" \
  --scenarios \
  100x2_5e3.yaml \
  100x2_5e4.yaml \
  100x2_5e5.yaml \
  100x2_5e6.yaml \
  200x4_5e3.yaml \
  200x4_5e4.yaml \
  200x4_5e5.yaml \
  200x4_5e6.yaml \
  400x8_5e3.yaml \
  400x8_5e4.yaml \
  400x8_5e5.yaml \
  400x8_5e6.yaml 

uv run python -m benchmark.generic.plot \
  --load-from "${script_dir}/results" \
  --scenarios \
  100x2_5e3 \
  100x2_5e4 \
  100x2_5e5 \
  100x2_5e6 \
  200x4_5e3 \
  200x4_5e4 \
  200x4_5e5 \
  200x4_5e6 \
  400x8_5e3 \
  400x8_5e4 \
  400x8_5e5 \
  400x8_5e6
