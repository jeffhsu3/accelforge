#!/bin/bash

# Check if these are installed. If they are, skip
output=$(pip3 list)
if ! echo "$output" | grep -q "hwcomponents"; then
    pip3 install -e /.dependencies/hwcomponents --break-system-packages
    pip3 install -e /.dependencies/hwcomponents/models/* --break-system-packages
fi
if ! echo "$output" | grep -q "combinatorics"; then
    pip3 install -e /.dependencies/combinatorics --break-system-packages
fi
if ! echo "$output" | grep -q "fastfusion"; then
    pip3 install -e /.dependencies/fastfusion --break-system-packages
fi
