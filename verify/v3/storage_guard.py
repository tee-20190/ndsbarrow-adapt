"""Preflight and live disk guard for new project jobs (GB means 10**9 bytes).

Use inside WSL for WSL jobs; the output path must be on the actual C: volume,
e.g. /mnt/c/.../outputs. Completed frozen launchers are not rewritten.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

GB = 10**9

def existing_ancestor(path):
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    return path

def assess(path, peak_gb, reserve_gb=20.0):
    if not math.isfinite(peak_gb) or peak_gb <= 0:
        raise ValueError('Declare a finite positive peak additional allocation')
    if not math.isfinite(reserve_gb) or reserve_gb < 20:
        raise ValueError('The reserve must be at least 20 GB')
    probe = existing_ancestor(path)
    free = shutil.disk_usage(probe).free
    needed = math.ceil((peak_gb+reserve_gb)*GB)
    return {'path':str(probe),'free_bytes':free,'peak_additional_bytes':math.ceil(peak_gb*GB),
            'reserve_bytes':math.ceil(reserve_gb*GB),'required_bytes':needed,
            'status':'PASS' if free >= needed else 'BLOCKED_LOW_DISK'}

def interrupt_group(process):
    if process.poll() is not None:
        return
    if os.name == 'nt':
        process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        os.killpg(process.pid, signal.SIGINT)

def stop_group(process):
    if process.poll() is not None:
        return
    if os.name == 'nt':
        process.terminate()
    else:
        os.killpg(process.pid, signal.SIGTERM)

def run_guarded(path, peak_gb, command, reserve_gb=20.0, interval=15.0):
    if not command:
        raise ValueError('Missing command after --')
    if os.name == 'nt' and Path(command[0]).stem.lower() == 'wsl':
        raise ValueError('Run this guard INSIDE WSL, around the actual Linux job')
    receipt = assess(path,peak_gb,reserve_gb)
    print(json.dumps(receipt),flush=True)
    if receipt['status'] != 'PASS':
        return 73
    options = {'creationflags':subprocess.CREATE_NEW_PROCESS_GROUP} if os.name=='nt' else {'start_new_session':True}
    process = subprocess.Popen(command,**options)
    interrupted_at = None
    try:
        while process.poll() is None:
            free = shutil.disk_usage(existing_ancestor(path)).free
            if free < reserve_gb*GB and interrupted_at is None:
                print(json.dumps({'status':'LOW_DISK_INTERRUPT','free_bytes':free,'pid':process.pid}),flush=True)
                interrupt_group(process)
                interrupted_at = time.monotonic()
            if interrupted_at is not None and time.monotonic()-interrupted_at >= 30:
                stop_group(process)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if os.name=='nt': process.kill()
                    else: os.killpg(process.pid,signal.SIGKILL)
                break
            time.sleep(interval)
        result = process.wait()
        return 73 if interrupted_at is not None else result
    except KeyboardInterrupt:
        interrupt_group(process)
        try: process.wait(timeout=30)
        except subprocess.TimeoutExpired: stop_group(process)
        return 130

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['check','run'])
    parser.add_argument('--path',required=True)
    parser.add_argument('--peak-gb',type=float,required=True,
                        help='Peak NEW bytes, including sources, checkpoints, optimizer, temp saves, cache and concurrent writers')
    parser.add_argument('--reserve-gb',type=float,default=20)
    # Split explicitly so options for this guard are parsed before the child.
    argv=sys.argv[1:]
    split=argv.index('--') if '--' in argv else len(argv)
    args=parser.parse_args(argv[:split])
    command=argv[split+1:]
    if args.mode=='check':
        rec=assess(args.path,args.peak_gb,args.reserve_gb)
        print(json.dumps(rec))
        return 0 if rec['status']=='PASS' else 73
    return run_guarded(args.path,args.peak_gb,command,args.reserve_gb)

if __name__=='__main__':
    raise SystemExit(main())
