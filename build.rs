fn main() {
    // If you want to auto-find Metal/CUDA features at build time, do it here.
    // For now we just print helpful notes.
    println!("cargo:rerun-if-env-changed=MODEL_PATH");
}
