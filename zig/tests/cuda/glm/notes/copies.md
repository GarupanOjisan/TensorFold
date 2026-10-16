# copies

Check that each kernel copy in zig/kernels/cuda matches the Python source lines its first line names.

A copy's first line reads "// Device code of src/tensorfold/<path> (lines A-B, ...)", or, for a file of the v1.8 tree,
"// Device code of src/tensorfold/<path> [v1.8] (lines A-B, ...)". Lines A-B of that file, without
comment-only lines and ATen/c10/torch includes, with ``namespace {`` named as the copy names it, must be the copy's lines
after the first, compared the same way; what follows them (the namespace close, explicit instantiations) is the copy's own.
The Python checkout is $TF_PYTHON_REPO (default ~/tf-native/py066, the v0.6.6 tree with src/tensorfold); a [v1.8] copy
is checked against $TF_PYTHON_REPO_V18 (default ~/tf-native/py18, v0.6.0 with the 82 patches of v1.8).
