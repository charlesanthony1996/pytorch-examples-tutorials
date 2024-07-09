def calculate_cpu_tflops(num_cores, clock_speed_ghz, flops_per_cycle):
    """
    Calculate CPU TFLOPS
    :param num_cores: Number of CPU cores
    :param clock_speed_ghz: Clock speed in GHz
    :param flops_per_cycle: Floating-point operations per cycle per core
    :return: TFLOPS
    """
    return num_cores * clock_speed_ghz * flops_per_cycle / 1000

def calculate_gpu_tflops(num_cores, clock_speed_ghz, flops_per_cycle):
    """
    Calculate GPU TFLOPS
    :param num_cores: Number of GPU cores
    :param clock_speed_ghz: Clock speed in GHz
    :param flops_per_cycle: Floating-point operations per cycle per core
    :return: TFLOPS
    """
    return num_cores * clock_speed_ghz * flops_per_cycle / 1000

# Example values for M3 Pro MacBook Pro (these are assumed values, you need to replace them with actual values)
cpu_cores = 12  # Example value
cpu_clock_speed_ghz = 3.2  # Example value
cpu_flops_per_cycle = 16  # Common value for modern CPUs

gpu_cores = 16  # Example value
gpu_clock_speed_ghz = 1.5  # Example value
gpu_flops_per_cycle = 2  # Common value for GPUs

# Calculate TFLOPS
cpu_tflops = calculate_cpu_tflops(cpu_cores, cpu_clock_speed_ghz, cpu_flops_per_cycle)
gpu_tflops = calculate_gpu_tflops(gpu_cores, gpu_clock_speed_ghz, gpu_flops_per_cycle)

print(f"CPU TFLOPS: {cpu_tflops:.2f}")
print(f"GPU TFLOPS: {gpu_tflops:.2f}")
