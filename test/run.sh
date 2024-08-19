#!/bin/bash

adb wait-for-device
adb root
adb shell rm -r /data/local/tmp/wl || true
adb shell mkdir -p /data/local/tmp/wl
adb push /home/lixiang/code/gemm/test/build/build/main.x /data/local/tmp/wl/
adb shell chmod +x /data/local/tmp/wl/main.x
#adb shell "echo \`cat /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq\` > /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq"
adb shell "cd /data/local/tmp/wl && taskset 1 /data/local/tmp/wl/main.x >> /data/local/tmp/wl/output_gemm.m"
adb shell taskset 1 /data/local/tmp/wl/main.x
#adb pull /data/local/tmp/wl/output_gemm.m ${OUTPUT_DIR}/output_gemm.m