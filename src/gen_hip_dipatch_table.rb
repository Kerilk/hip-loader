require 'yaml'
require_relative '../utils/yaml_ast'

spec = YAMLCAst.from_yaml_ast(YAML.load_file("hip_api.yaml"))

spec["functions"].each { |f|
  puts YAMLCAst::Declaration.new(name: f.name + "_t", type: f.type, storage: "typedef").to_s + ";"
}

puts YAMLCAst::Struct.new(name: "hip_dipatch", members: spec["functions"].map { |f|
  YAMLCAst::Declaration.new(name: f.name, type: YAMLCAst::Pointer.new(type: YAMLCAst::CustomType.new(name: f.name + "_t")))
}).to_s + ";"
