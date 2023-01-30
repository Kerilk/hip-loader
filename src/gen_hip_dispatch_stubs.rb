require 'yaml'
require_relative '../utils/yaml_ast'

$spec = YAMLCAst.from_yaml_ast(YAML.load_file("hip_api.yaml"))

$dispatch = YAML.load_file("hip_dispatch.yaml")

$type_map = {
  "hipCtx_t" => "struct _hip_context_s *",
  "hipEvent_t" => "struct _hip_event_s *",
  "hipStream_t" => "struct _hip_stream_s *",
}

$unwrap = /hipCtx_t|hipEvent_t|hipStream_t/

$spec["functions"].each { |f|
  next unless $dispatch[f.name]
  puts YAMLCAst::Declaration.new(name: f.name, type: f.type).to_s
  puts <<EOF
{
	_initOnce();
EOF
  dispatch_params = [f.name] + f.type.params.map { |p|
    case p.type.to_s
    when $unwrap
      "_HIPLD_UNWRAP((#{$type_map[p.type.to_s]})#{p.name})"
    else
      p.name
    end
  }
  case($dispatch[f.name]["type"])
  when :current_context
    puts <<EOF
	struct _hip_context_s * _hip_context = _ctxStackTop();
	_HIPLD_CHECK_CTX(_hip_context);
	_HIPLD_RETURN(_HIPLD_DISPATCH(_hip_context, #{dispatch_params.join(", ")}));
EOF
  when :current_device
    puts <<EOF
	struct _hip_device_s *_hip_device = _ctxDeviceGet();
	_HIPLD_RETURN(_HIPLD_DISPATCH(_hip_device, #{dispatch_params.join(", ")}));
EOF
  when :device_id
    puts <<EOF
	_HIPLD_CHECK_DEVICEID(#{$dispatch[f.name]["name"]});
	struct _hip_device_s * _hip_device = _deviceArray[#{$dispatch[f.name]["name"]}];
	#{$dispatch[f.name]["name"]} = _hip_device->driverIndex;
	_HIPLD_RETURN(_HIPLD_DISPATCH(_hip_device, #{dispatch_params.join(", ")}));
EOF
  when "hipDevice_t"
    puts <<EOF
	int index = _handleToIndex(#{$dispatch[f.name]["name"]});
	_HIPLD_CHECK_DEVICEID(index);
        struct _hip_device_s * _hip_device = _deviceArray[index];
	#{$dispatch[f.name]["name"]} = _hip_device->driverHandle;
	_HIPLD_RETURN(_HIPLD_DISPATCH(_hip_device, #{dispatch_params.join(", ")}));
EOF
  when "hipEvent_t"
    puts <<EOF
	struct _hip_event_s *_hip_event = (struct _hip_event_s *)#{$dispatch[f.name]["name"]};
	_HIPLD_RETURN(_HIPLD_DISPATCH(_hip_event, #{dispatch_params.join(", ")}));
EOF
  when "hipCtx_t"
    puts <<EOF
	struct _hip_context_s * _hip_context = (struct _hip_context_s *)#{$dispatch[f.name]["name"]};
	_HIPLD_RETURN(_HIPLD_DISPATCH(_hip_context, #{dispatch_params.join(", ")}));
EOF
  else
    raise "unsupported dispatch"
  end
  puts <<EOF
}
EOF
}
