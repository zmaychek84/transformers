# Clone the repository
git clone https://gitenterprise.xilinx.com/zepingl/spec_dec_simplest.git

cd Spec-dec-simplest

# Install dependencies
git clone https://gitenterprise.xilinx.com/zepingl/human-eval.git

pip install -e ./human-eval

# Run the evaluation script
sh evaluate_humaneval.sh
