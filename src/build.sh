rm -rf include-modified
cp -r ../include include-modified
cat headers.patch | patch -i - -d include-modified/ -s -p1
ruby extract_hip.rb
ruby gen_hip_dipatch_table.rb > hip_dispatch.h
ruby gen_hip_dispatch_stubs.rb > hip_dispatch_stubs.h
gcc -fPIC -g -c hip_loader.c -I ../include -D__HIP_PLATFORM_AMD__ -Wall
gcc -fPIC -g -shared -Wl,--version-script -Wl,hip.map hip_loader.o -Wl,-soname -Wl,libHIPLoader.so.5 -lpthread -ldl -o libHIPLoader.so.5.4.0
ln -f -s libHIPLoader.so.5.4.0 libHIPLoader.so
ln -f -s libHIPLoader.so.5.4.0 libHIPLoader.so.5
