fn main() {
    let root = ".cuda";
    let lib = ".cuda/lib";

    println!("cargo::rustc-link-search=native={lib}");
    println!("cargo::rustc-env=CUDA_PATH={root}");
    println!("cargo::rustc-env=LD_LIBRARY_PATH={lib}");
}
