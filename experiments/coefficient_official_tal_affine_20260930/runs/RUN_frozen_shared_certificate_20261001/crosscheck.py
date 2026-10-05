import numpy as np,json,pathlib,math
p=pathlib.Path('/root/official_tal_affine_20260930/runs/RUN_frozen_shared_certificate_20261001')
f=np.load(p/'FROZEN_DESIGN.npz'); g=np.load(p/'GRADIENTS.npz'); c=json.loads((p/'CERTIFICATE.json').read_text())
N=len(f['H']); terms=[]; checks=[]
for l in range(3):
 m=f['levels']==l; H=f['H'][m]; G=g['G'][m]; Q,R=np.linalg.qr(H,mode='reduced'); n=float(np.linalg.norm(Q.T@G,'fro')); terms.append(n*n/(2*.003*N))
 checks.append({'level':l,'qr_gradient_norm':n,'svd_gradient_norm':c['projections'][l]['projected_unaveraged_gradient_norm'],'absolute_difference':abs(n-c['projections'][l]['projected_unaveraged_gradient_norm']),'qr_orthogonality':float(np.linalg.norm(Q.T@Q-np.eye(65),2))})
r={'kind':'remote CPU QR recomputation from frozen arrays; no model or gradient reevaluation','N':N,'qr_epsilon_raw':math.fsum(terms),'svd_epsilon_raw':c['epsilon_S_raw'],'absolute_difference':abs(math.fsum(terms)-c['epsilon_S_raw']),'levels':checks}
(p/'CERTIFICATE_CROSSCHECK.json').write_text(json.dumps(r,indent=2),encoding='utf-8'); print(json.dumps(r,indent=2))
