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
    elsif f.type.type.name == "size_t"
      ret = 0
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

typedef void * hipGetFunc_t(const char *fName);

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
	hipGetFunc_t          *hipGetFunc; //should be one of the only exposed
	struct _hip_dipatch_s  dispatch;
	struct _hip_driver_s  *pNext;
};

static inline intptr_t
_hipld_driver_get_function(struct _hip_driver_s *pDriver, const char *fName) {
	if (pDriver->hipGetFunc)
		return (intptr_t)pDriver->hipGetFunc(fName);
	else
		return (intptr_t)dlsym(pDriver->pLibrary, fName);
}

static hipError_t
_fillDriverDispatch(struct _hip_driver_s *pDriver) {
EOF

load_block = lambda { |name|
  puts <<EOF
	pDriver->dispatch.#{name} = (#{name}_t *)_hipld_driver_get_function(pDriver, \"#{name}\");
	if (!pDriver->dispatch.#{name})
		pDriver->dispatch.#{name} = &#{name}_unimp;
EOF
}

$spec["functions"].each { |f|
  load_block.call(f.name)
}

puts <<EOF
	return hipSuccess;
}
EOF
