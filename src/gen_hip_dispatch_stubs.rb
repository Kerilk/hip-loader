require 'yaml'
require_relative '../utils/yaml_ast'

$spec = YAMLCAst.from_yaml_ast(YAML.load_file("hip_api.yaml"))

$dispatch = YAML.load_file("hip_dispatch.yaml")

$type_map = {
  "hipCtx_t" => "struct _hip_context_s *"
}

$unwrap = /hipCtx_t/

$spec["functions"].each { |f|
  next unless $dispatch[f.name]
  puts YAMLCAst::Declaration.new(name: f.name, type: f.type).to_s
  puts <<EOF
{
	_initOnce();
EOF
  params = f.type.params.map { |p|
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
	_HIPLD_CHECK_ERR(_hip_context->multiplex->dispatch.#{f.name}(#{params.join(", ")}));
EOF
  when "hipDevice_t"
    puts <<EOF
	int index = _handleToIndex(#{$dispatch[f.name]["name"]});
	_HIPLD_CHECK_DEVICEID(index);
        struct _hip_device_s * _hip_device = _deviceList + index;
	#{$dispatch[f.name]["name"]} = _hip_device->driverHandle;
	_HIPLD_CHECK_ERR(_hip_device->multiplex->dispatch.#{f.name}(#{params.join(", ")}));
EOF
  else
    raise "unsupported dispatch"
  end
  puts <<EOF
	_HIPLD_RETURN(hipSuccess);
}
EOF
}
