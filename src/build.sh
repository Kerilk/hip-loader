ruby extract_hip.rb
ruby gen_hip_dipatch_table.rb | indent -bfda -l160 > hip_dispatch.h
ruby gen_hip_dispatch_stubs.rb | indent -bfda -l160 > hip_dispatch_stubs.h
gcc -c hip_loader.c -I include -D__HIP_PLATFORM_AMD__ -Wall
