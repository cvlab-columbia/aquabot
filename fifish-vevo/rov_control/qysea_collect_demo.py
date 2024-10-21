import subprocess
import psutil

def set_cpu_affinity(process, cpu_list):
    try:
        p = psutil.Process(process.pid)
        p.cpu_affinity(cpu_list)
    except Exception as e:
        print(f"Error setting CPU affinity: {e}")

def launch_script(script_path, cpu_list):
    process = subprocess.Popen(["python", script_path])
    set_cpu_affinity(process, cpu_list)
    return process

if __name__ == "__main__":
    # Define CPU threads for each script
    qysea_xbox_cpus = list(range(32))
    all_cams_cpus = list(range(32, 60))
    qysea_status_cpus = list(range(60, 64))

    # Paths to the scripts
    qysea_xbox_path = "qysea_xbox.py"
    all_cams_path = "all_cams.py"
    qysea_status_path = "qysea_status.py"

    # Launch scripts with specified CPU threads
    qysea_xbox_process = launch_script(qysea_xbox_path, qysea_xbox_cpus)
    all_cams_process = launch_script(all_cams_path, all_cams_cpus)
    qysea_status_process = launch_script(qysea_status_path, qysea_status_cpus)

    # Optionally, you can wait for the processes to complete
    qysea_xbox_process.wait()
    all_cams_process.wait()
