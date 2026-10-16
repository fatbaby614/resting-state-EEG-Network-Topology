import os
import paramiko

HOST = "192.168.140.9"
USER = "tanhuang"
KEY_PATH = os.path.join(os.environ["USERPROFILE"], ".ssh", "id_ed25519_hbn4")
RESULT_PATH = os.path.join(os.environ["TEMP"], "hbn4_lobe_check.txt")

BATCH = (
    "date '+%Y-%m-%d %H:%M:%S'; "
    "echo '--- HEAD3'; head -3 /tmp/lobe_run_v2.log; "
    "echo '--- TAIL15'; tail -15 /tmp/lobe_run_v2.log; "
    "echo '--- PS'; ps -eo pid,etime,cmd | grep run_lobe_allometry | grep -v grep; "
    "echo '--- PSWC'; ps aux | grep run_lobe_allometry | grep -v grep | wc -l; "
    "echo '--- CKPT'; ls -la /mnt/data1/home/tanhuang/code/EEG_code/HBN4/results/tables/lobe_features_checkpoint.pkl; "
    "echo '--- CSV'; ls -la /mnt/data1/home/tanhuang/code/EEG_code/HBN4/results/tables/tae_lobe_allometry.csv; "
    "echo '--- END'"
)

pkey = paramiko.Ed25519Key.from_private_key_file(KEY_PATH)
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=22, username=USER, pkey=pkey, timeout=10,
          banner_timeout=10, auth_timeout=10, allow_agent=False, look_for_keys=False)
stdin, stdout, stderr = c.exec_command(BATCH, timeout=25)
stdout.channel.settimeout(25)
out = stdout.read().decode(errors="replace").strip()
err = stderr.read().decode(errors="replace").strip()
c.close()

with open(RESULT_PATH, "w", encoding="utf-8") as f:
    f.write(out + ("\nSTDERR: " + err if err else ""))
print("DONE, result:", RESULT_PATH)
