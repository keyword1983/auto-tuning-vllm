import re

import transformers.integrations.heterogeneity.configuration_utils as h

file_path = h.__file__
with open(file_path, 'r', encoding='utf-8') as f:
    code = f.read()

# Match the line raising AmbiguousGlobalPerLayerAttributeError and capture its
# leading whitespace, so the injected replacement lines reuse the *actual*
# indentation of this call site instead of a hardcoded guess. A hardcoded
# indent silently breaks if it doesn't match the enclosing `if` block's
# nesting level: Python then raises IndentationError on the next statement
# after the (multi-line) raise call, which crashes transformers' import for
# every model, not just heterogeneous-config ones.
pattern = re.compile(
    r'^([ \t]*)raise AmbiguousGlobalPerLayerAttributeError\(', re.MULTILINE
)
match = pattern.search(code)

if match:
    indent = match.group(1)
    replacement = (
        f"{indent}return getattr(self.per_layer_config[0], name) if hasattr(self, 'per_layer_config') and self.per_layer_config else None\n"
        f"{indent}if False: raise AmbiguousGlobalPerLayerAttributeError("
    )
    code = pattern.sub(replacement, code, count=1)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(code)
    print("✅ transformers patched successfully!")
else:
    print("ℹ️ Already patched or target not found.")
