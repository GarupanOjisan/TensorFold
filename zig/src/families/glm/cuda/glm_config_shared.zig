//! glm_config in the native server: families/glm/config.zig through glm_cuda, as a file belongs to one module.
const c = @import("glm_cuda").config;
pub const Config = c.Config;
pub const Kind = c.Kind;
pub const parse = c.parse;
pub const max_layers = c.max_layers;
