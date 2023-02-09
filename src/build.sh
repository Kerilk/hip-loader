ruby extract_hip.rb
ruby gen_hip_dipatch_table.rb | indent -bfda -l160 > hip_dispatch.h
ruby gen_hip_dispatch_stubs.rb | indent -bfda -l160 > hip_dispatch_stubs.h
gcc -fPIC -c hip_loader.c -I include -D__HIP_PLATFORM_AMD__ -Wall
gcc -fPIC -shared -Wl,--version-script -Wl,hip.map hip_loader.o -Wl,-soname -Wl,libHIPLoader.so.5 -o libHIPLoader.so.5.4.0
