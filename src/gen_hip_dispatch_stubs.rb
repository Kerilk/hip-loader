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
  create = dispatch["create"]
  dispatch_params = [f.name]
  dispatch_params += f.type.params.map { |p| p.name } if f.type.params
  check = "_HIPLD_RETURN"
  check = "_HIPLD_CHECK_ERR" if create && !dispatch["create_on_error"]
  check = "hipError_t _create_err = " if dispatch["create_on_error"]
  check = "_RETURN" if (!f.type.type.is_a?(YAMLCAst::CustomType) || f.type.type.name != "hipError_t") && !(create || dispatch["create_on_error"])
  ret = "_HIPLD_RETURN"
  ret = "_HIPLDRTC_RETURN" if f.type.type.is_a?(YAMLCAst::CustomType) && f.type.type.name == "hiprtcResult"
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
  when "textureReference"
    multiplex = "#{dispatch["name"]}->textureObject"
    puts <<EOF
	_HIPLD_CHECK_PTR(#{dispatch["name"]});
	_HIPLD_CHECK_PTR(#{multiplex});
EOF
  when "hipDevice_t"
    multiplex = "_hip_device"
    puts <<EOF
	int index = _handleToIndex(#{dispatch["name"]});
	_HIPLD_CHECK_DEVICEID(index);
        struct _hip_device_s * _hip_device = _deviceArray[index];
	#{dispatch["name"]} = _hip_device->driverHandle;
EOF
  when "hipEvent_t", "hipCtx_t", "hipModule_t", "hipFunction_t", "hiprtcProgram", "hiprtcLinkState", "hipTextureObject_t", "hipGraph_t", "hipGraphNode_t", "hipGraphExec_t", "hipGraphicsResource_t", "hipMemPool_t", "hipMemGenericAllocationHandle_t", "hipSurfaceObject_t", "hipUserObject_t"
    multiplex = dispatch["name"]
  when "hipStream_t"
    multiplex = "_hip_device"
    puts <<EOF
	struct _hip_device_s *_hip_device = _get_device_for_stream(#{dispatch["name"]});
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
  l = lambda { |c|
    puts <<EOF
	if (#{c} && *#{c})
		(*#{c})->multiplex = #{multiplex}->multiplex;
EOF
  }
  if dispatch["create_on_error"] && dispatch["create"]
    puts <<EOF
	if (_create_err == hipSuccess) {
EOF
  end
  if dispatch["create"]
    if dispatch["create"].kind_of?(Array)
      dispatch["create"].each { |e| l.call(e) }
    elsif dispatch["create"].kind_of?(Hash)
      puts <<EOF
	if (#{dispatch["create"]["names"].first})
		for (size_t i = 0; i < #{dispatch["create"]["size"]}; i++) {
EOF
      dispatch["create"]["names"].each { |e|
        puts <<EOF
			#{e}[i]->multiplex = #{multiplex}->multiplex;
EOF
      }
      puts <<EOF
		}
EOF
    else
      l.call(dispatch["create"])
    end
  end
  if dispatch["create_on_error"] && dispatch["create"]
    puts <<EOF
	}
EOF
  end
  if dispatch["create_on_error"]
    l.call(dispatch["create_on_error"])
    puts <<EOF
	_HIPLD_RETURN(_create_err);
EOF
  elsif dispatch["create"]
    puts <<EOF
	_HIPLD_RETURN(hipSuccess);
EOF
  end
  puts <<EOF
}
EOF
}
