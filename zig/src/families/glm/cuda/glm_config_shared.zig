//! `glm_config` in the native server: families/glm/config.zig as the NVFP4 family's module (glm_cuda) holds it,
//! since a file belongs to one module. tf-glm, which has no glm_cuda, imports config.zig as its own module.
const c = @import("glm_cuda").config;
pub const Config = c.Config;
pub const Kind = c.Kind;
pub const parse = c.parse;
pub const max_layers = c.max_layers;
