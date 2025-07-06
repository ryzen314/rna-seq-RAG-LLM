import subprocess

def getNvidiaVram():
    try:
        result = subprocess.run(
                command = ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader'],
                stdout = subprocess.PIPE
                stderr = subprocess.PIPE
                text = True
                )
        vram = result.stdout.strip()
        return vram
    except Exception as e:
        return e


def getAMDGpuVram():
    try:
        result = subprocess.run(
                command = ['rocm-smi', '--showmeinfo', '--json'],
                stdout = subprocess.PIPE,
                stderr = subprocess.PIPE,
                text = True
                )
        import json
        data = json.loads(result.stdout)
        vram = data['memory']['total']
        return vram
    except Exception as e:
        return e

def getAppleSiliconVram():
    try:
        result = subprocess.run(
                command = ['system_profiler', 'SPDisplaysDataType']
                stdout = subprocess.PIPE,
                stderr = subprocess.PIPE,
                text = True
                )
        for line in result.stdout.splitlines():
            if "VRAM" in line:
                return line.strip()
        return "VRAM information not found."
    except Exception as e:
        return f"Error: {e}"
