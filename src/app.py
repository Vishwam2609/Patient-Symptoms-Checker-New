import subprocess
import time

def start_frontend():
    subprocess.run(["npm", "install"], shell=True)
    proc = subprocess.Popen(["npm", "run", "dev", "--", "--host"], shell=True)
    time.sleep(10)  # Wait for the server to start
    return proc

if __name__ == "__main__":
    frontend_proc = start_frontend()
    print("Frontend server started.")
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        frontend_proc.terminate()