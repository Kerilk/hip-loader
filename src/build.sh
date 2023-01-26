ruby extract_hip.rb
ruby gen_hip_dipatch_table.rb | indent -bfda -l160 > hip_dispatch.h
gcc -c hip_loader.c -I include -D__HIP_PLATFORM_AMD__ -Wall
