#!/bin/bash

scenes=(
scene0568_00 scene0568_01 scene0568_02
scene0304_00
scene0488_00 scene0488_01
scene0412_00 scene0412_01
scene0217_00
scene0019_00 scene0019_01
scene0414_00
scene0575_00 scene0575_01 scene0575_02
scene0426_00 scene0426_01 scene0426_02 scene0426_03
scene0549_00 scene0549_01
scene0578_00 scene0578_01 scene0578_02
scene0665_00 scene0665_01
scene0050_00 scene0050_01 scene0050_02
scene0257_00
scene0025_00 scene0025_01 scene0025_02
scene0583_00 scene0583_01 scene0583_02
scene0701_00 scene0701_01 scene0701_02
scene0580_00 scene0580_01
scene0565_00
scene0169_00 scene0169_01
scene0655_00 scene0655_01 scene0655_02
scene0063_00
scene0221_00 scene0221_01
)

for scene in "${scenes[@]}"; do
    echo "Downloading $scene..."
    python download_wrapper.py "$scene"
done