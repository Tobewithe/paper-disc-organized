"""Pinned-runtime and input audit; no base model or optimizer instantiated."""
from rfc_runtime import *
from rfc_contract import check_native_args


def main():
    with Run('PREFLIGHT',CACHE,600) as run:
        data=inspect_inputs(); resources=initial_resources()
        check_native_args(read_json(BASE/'trace/NATIVE_ARGS.json'),read_json(CACHE/'trace/NATIVE_ARGS.json'))
        run.freeze(dict(environment=data['environment'],resources=resources,
                        design_sha256=sha256(STUDY/'DESIGN_SPEC.json'),
                        authorization_sha256=sha256(STUDY/'AUTHORIZATION.json')))
        write_new(run.path/'ENVIRONMENT.json',data['environment'])
        audit(run,data,existing_source_hashes=33,original_native_configuration_match=True)
        run.finish(dict(environment=data['environment'],resources=resources,
                        base_model_forward_images=0,optimizer_steps=0),scientific='execution_preflight_only')
    set_pointer('PREFLIGHT',run)
    write_new(STUDY/'EXECUTION_FREEZE.json',dict(status='frozen',configuration_validation='passed',
              preflight_run=run.run_id,source_manifest_sha256=sha256(run.path/'SOURCE_MANIFEST.json'),
              design_sha256=sha256(STUDY/'DESIGN_SPEC.json'),authorization_sha256=sha256(STUDY/'AUTHORIZATION.json'),
              requires_real_smoke_before_formal=True,formal_A_Q_steps=3000))

if __name__=='__main__': main()
