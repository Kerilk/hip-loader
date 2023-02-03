require 'yaml'
require_relative '../utils/yaml_ast'

$spec = YAMLCAst.from_yaml_ast(YAML.load_file("hip_api.yaml"))

$dispatch = YAML.load_file("hip_dispatch.yaml")

$spec["functions"].each { |f|
  dispatch = $dispatch[f.name]
  next unless dispatch
  puts YAMLCAst::Declaration.new(name: f.name, type: f.type).to_s
  puts <<EOF
{
	_initOnce();
EOF
  dispatch_params = [f.name]
  dispatch_params += f.type.params.map { |p| p.name } if f.type.params
  check = "_HIPLD_RETURN"
  check = "_HIPLD_CHECK_ERR" if dispatch["create"]
  case(dispatch["type"])
  when :current_context
    multiplex = "_hip_context"
    puts <<EOF
	hipCtx_t _hip_context = _ctxStackTop();
	_HIPLD_CHECK_CTX(_hip_context);
EOF
  when :current_device
    multiplex = "_hip_device"
    puts <<EOF
	struct _hip_device_s *_hip_device = _ctxDeviceGet();
EOF
  when :device_id
    multiplex = "_hip_device"
    puts <<EOF
	_HIPLD_CHECK_DEVICEID(#{dispatch["name"]});
	struct _hip_device_s * _hip_device = _deviceArray[#{dispatch["name"]}];
	#{dispatch["name"]} = _hip_device->driverIndex;
EOF
  when "hipDevice_t"
    multiplex = "_hip_device"
    puts <<EOF
	int index = _handleToIndex(#{dispatch["name"]});
	_HIPLD_CHECK_DEVICEID(index);
        struct _hip_device_s * _hip_device = _deviceArray[index];
	#{dispatch["name"]} = _hip_device->driverHandle;
EOF
  when "hipEvent_t", "hipCtx_t", "hipModule_t", "hipFunction_t"
    multiplex = dispatch["name"]
  when "hipStream_t"
    multiplex = "_hip_device"
    puts <<EOF
	struct _hip_device_s *_hip_device;
	if (#{dispatch["name"]})
		_hip_device = #{dispatch["name"]}->multiplex->pDevice;
	else
		_hip_device = _ctxDeviceGet();
EOF
  when :all_drivers
    puts <<EOF
	_initOnce();
	if (!modules)
		return;
	void *** res = (void ***)modules;
	struct _hip_driver_s *driver = _driverList;
	for (int i = 0; i < _hipDriverCount; i++)
		if (res[i]) {
			modules = res[i];
			driver->dispatch.#{dispatch_params.first}(#{dispatch_params[1..-1].join(", ")});
		}
}
EOF
    next
  else
    raise "unsupported dispatch"
  end
  puts <<EOF
	#{check}(_HIPLD_DISPATCH(#{multiplex}, #{dispatch_params.join(", ")}));
EOF
  if dispatch["create"]
    puts <<EOF
	(*#{dispatch["create"]})->multiplex = #{multiplex}->multiplex;
	_HIPLD_RETURN(hipSuccess);
EOF
  end
  puts <<EOF
}
EOF
}
