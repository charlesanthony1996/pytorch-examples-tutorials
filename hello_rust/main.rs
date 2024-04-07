use std::process::Command;

fn main() {
    // Get Rust version
    let rust_version_output = Command::new("rustc").arg("--version").output().expect("Failed to execute rustc");
    let rust_version = String::from_utf8_lossy(&rust_version_output.stdout);
    println!("Rust version: {}", rust_version);

    // Get system information
    let uname_output = Command::new("uname").arg("-a").output().expect("Failed to execute uname");
    let system_info = String::from_utf8_lossy(&uname_output.stdout);
    println!("System information: {}", system_info);

    // Get system architecture
    let arch_output = Command::new("uname").arg("-m").output().expect("Failed to execute uname");
    let system_arch = String::from_utf8_lossy(&arch_output.stdout);
    println!("System architecture: {}", system_arch);
}
