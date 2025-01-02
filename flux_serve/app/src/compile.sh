#!/bin/bash

find $COMPILER_WORKDIR_ROOT -type d \( -name "compiled_model" -o -name "__pycache__" -o -name "compiler_workdir" \) -exec rm -rf {} +

myloc=$COMPILER_WORKDIR_ROOT
python "$myloc/text_encoder_1/compile.py"
python "$myloc/text_encoder_2/compile.py" -m 32
python "$myloc/transformer/compile.py" -hh $HEIGHT -w $WIDTH -m 32
python "$myloc/decoder/compile.py" -hh $HEIGHT -w $WIDTH
echo "done compiling; cleaning compiler_workdir"
find $COMPILER_WORKDIR_ROOT -type d \( -name "__pycache__" -o -name "compiler_workdir" \) -exec rm -rf {} +
