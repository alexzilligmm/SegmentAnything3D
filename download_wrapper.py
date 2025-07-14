import subprocess
import sys

scene_id = sys.argv[1]
cmd = f"python download.py -o data --id {scene_id}"

process = subprocess.Popen(cmd, stdin=subprocess.PIPE, shell=True)
process.stdin.write(b"t\n")  # accetta i termini
process.stdin.write(b"r\n")  # non scaricare .sens
process.stdin.close()
process.wait()