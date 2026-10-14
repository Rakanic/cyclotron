// `--features gemmini`: compile libgemmini's Radiance build (functional + timing MX-Gemmini behind the MMIO registers,
// libgemmini/gemmini_radiance.h) into the crate. Without the feature there is nothing to build.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=GEMMINI_ROCC_TESTS");
    if env::var_os("CARGO_FEATURE_GEMMINI").is_none() {
        return;
    }
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let lib = root.join("libgemmini");
    let rocc_tests = env::var_os("GEMMINI_ROCC_TESTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("../../gemmini/software/gemmini-rocc-tests"));
    let vpu_ref = rocc_tests.join("include/vpu_ref.h");
    if !vpu_ref.exists() {
        panic!(
            "--features gemmini needs {} (set GEMMINI_ROCC_TESTS to a gemmini-rocc-tests checkout)",
            vpu_ref.display()
        );
    }

    let mut sources = vec![lib.join("gemmini.cc"), lib.join("gemmini_radiance.cc")];
    let mut dirs = vec![lib.join("perf")];
    while let Some(dir) = dirs.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                dirs.push(path);
            } else if path.extension().map_or(false, |e| e == "cc") {
                sources.push(path);
            }
        }
    }
    sources.sort();

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .opt_level(3)
        .define("GEMMINI_NO_SPIKE", None)
        .define("GEMMINI_RADIANCE", None)
        .include(&lib)
        .include(rocc_tests.join("include"))
        .warnings(false)
        .files(&sources)
        .compile("gemmini_radiance");

    println!("cargo:rerun-if-changed={}", lib.display());
    println!("cargo:rerun-if-changed={}", vpu_ref.display());
}
