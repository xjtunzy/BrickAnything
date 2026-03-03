#!/usr/bin/env bash
cd cpp/build
make -j4
cd ../..
echo 'Begin LEGO construct!!'
python legolazition_cpp.py --name cabinet --res 64