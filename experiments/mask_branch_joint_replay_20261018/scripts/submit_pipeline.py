"""Submit exactly one finite Slurm dependency chain after successful preparation."""
from pathlib import Path
import argparse
import datetime
import json
import os
import subprocess

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-job', default=None)
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    study=Path(config['study_root'])
    receipt=study/'SLURM_SUBMISSION.json'
    with receipt.open('x') as out:
        state={'submitted_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'data_job':args.data_job,'kind':'one_shot_finite_dependency_chain',
               'no_automation_or_goal':True,'jobs':[]}
        json.dump(state,out,indent=2)
    previous=args.data_job
    for stage,hours in [('gate',8),('G_budget',8),('C_native',8),('PC_native',8),('evaluation',12)]:
        command=['sbatch','--parsable','--partition=gpu','--account=lab','--qos=normal',
                 '--nodes=1','--ntasks=1','--gres=gpu:1','--cpus-per-task=8','--mem=64G',
                 f'--time={hours}:00:00', '--kill-on-invalid-dep=yes',
                 '--job-name=pc-joint-'+stage,
                 f'--output=/srv/gpu-lab/users/a25023104/logs/pc-replay-{stage}-%j.log',
                 str(study/'scripts/server_stage.sbatch'),stage]
        if previous:
            command.insert(2, f'--dependency=afterok:{previous}')
        result=subprocess.run(command,text=True,capture_output=True)
        record={'stage':stage,'command':command,'returncode':result.returncode,'stderr':result.stderr}
        if result.returncode==0:
            job=result.stdout.strip().split(';')[0]
            if not job.isdigit(): raise RuntimeError('ambiguous sbatch result: '+result.stdout)
            record.update(job_id=job,depends_on=previous,execution_status='submitted_not_started')
            previous=job
        state['jobs'].append(record)
        temporary=receipt.with_suffix('.tmp')
        temporary.write_text(json.dumps(state,indent=2)+'\n')
        os.replace(temporary,receipt)
        if result.returncode: raise RuntimeError('Slurm submission failed; earlier submitted jobs preserved: '+result.stderr)
    print(json.dumps(state,indent=2))

if __name__=='__main__':main()
