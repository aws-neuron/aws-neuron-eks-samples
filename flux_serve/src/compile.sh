#!/bin/bash
find . -type d \( -name "compiled_model" -o -name "__pycache__" -o -name "compiler_workdir" \) -exec rm -rf {} +

myloc="$(dirname "$0")"

python "$myloc/text_encoder_1/compile.py"
python "$myloc/text_encoder_2/compile.py" -m 32
python "$myloc/transformer/compile.py" -hh 512 -w 512 -m 32
python "$myloc/decoder/compile.py" -hh 512 -w 512
echo "done!"
