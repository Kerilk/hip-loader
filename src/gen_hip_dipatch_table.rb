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

$spec["functions"].each { |f|
  next if ["hipGetStreamDeviceId", "hipCreateChannelDesc"].include?(f.name)
  puts YAMLCAst::Declaration.new(name: f.name + "_unimp", type: f.type, storage: "static").to_s + "{"
  ret = ""
  if f.type.type.is_a?(YAMLCAst::Pointer)
    ret = "NULL";
  elsif f.type.type.is_a?(YAMLCAst::CustomType)
    if f.type.type.name == "hipError_t"
      ret = "hipErrorNotSupported"
    elsif f.type.type.name == "hiprtcResult"
      ret = "HIPRTC_ERROR_INTERNAL_ERROR"
    end
  end
  puts "	return #{ret};"
  puts "}"
}

puts <<EOF
static int
hipGetStreamDeviceId_unimp (
  hipStream_t stream)
{
	return -1;
}
static hipChannelFormatDesc
hipCreateChannelDesc_unimp (
  int x,
  int y,
  int z,
  int w,
  enum hipChannelFormatKind f)
{
	hipChannelFormatDesc desc = {x, y, z, w, f};
	return desc;
}
EOF

puts YAMLCAst::Struct.new(name: "_hip_dipatch_s", members: $spec["functions"].map { |f|
  YAMLCAst::Declaration.new(name: f.name, type: YAMLCAst::Pointer.new(type: YAMLCAst::CustomType.new(name: f.name + "_t")))
}).to_s + ";"

puts <<EOF

struct _hip_device_s;
struct _hip_driver_s;
// Instance layers can be added later
struct _multiplex_s {
	struct _hip_dipatch_s  dispatch;
	struct _hip_device_s  *pDevice;
	struct _hip_driver_s  *pDriver;
};

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
  puts <<EOF
	pDriver->dispatch.#{f.name} = (#{f.name}_t *)(intptr_t)dlsym(pDriver->pLibrary, \"#{f.name}\");
	if (!pDriver->dispatch.#{f.name})
		pDriver->dispatch.#{f.name} = &#{f.name}_unimp;
EOF
}

puts <<EOF
	return hipSuccess;
}
EOF
