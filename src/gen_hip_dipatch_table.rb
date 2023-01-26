require 'yaml'
require_relative '../utils/yaml_ast'

$spec = YAMLCAst.from_yaml_ast(YAML.load_file("hip_api.yaml"))

def declare(*args)
  args.each { |n|
    t = $spec["typedefs"].find { |t| t.name == n }
    t.instance_variable_set(:@storage, "typedef")
    puts t.to_s + ";"
  }
end

declare("activity_domain_t", "hipRegisterTracerCallback_callback_t")

$spec["functions"].each { |f|
  puts YAMLCAst::Declaration.new(name: f.name + "_t", type: f.type, storage: "typedef").to_s + ";"
}

puts YAMLCAst::Struct.new(name: "_hip_dipatch_s", members: $spec["functions"].map { |f|
  YAMLCAst::Declaration.new(name: f.name, type: YAMLCAst::Pointer.new(type: YAMLCAst::CustomType.new(name: f.name + "_t")))
}).to_s + ";"

puts <<EOF

struct _hip_driver_s;
struct _hip_device_s;
struct _hip_driver_s {
	void                  *pLibrary;
	int                    deviceCount;
	struct _hip_device_s  *pDevices;
	hipGetDeviceCount_t   *hipGetDeviceCount;
	hipDeviceGet_t        *hipDeviceGet;
//	hipGetFunc_t          *hipGetFunc; //should be one of the only exposed
	struct _hip_dipatch_s  dispatch;
	struct _hip_driver_s  *pNext;
};

static hipError_t
_fillDriverDispatch(struct _hip_driver_s *pDriver) {
EOF

$spec["functions"].each { |f|
  puts "  pDriver->dispatch.#{f.name} = (#{f.name}_t *)(intptr_t)dlsym(pDriver->pLibrary, \"#{f.name}\");"
}

puts <<EOF
	return hipSuccess;
}
EOF
