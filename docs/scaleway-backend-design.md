# Scaleway Backend — Design Proposal

**Statut** : RFC — en attente d'avis équipe  
**Auteur** : Gireg Roussel  
**Date** : 2026-04-24

---

## Motivation

retrain dispose aujourd'hui de trois backends : `local` (GPU local), `tinker` (service propriétaire) et `prime_rl` (stack PRIME-RL externe). Aucun d'eux ne répond à la contrainte de **souveraineté** : données et calcul hébergés en France/UE, sans dépendance à un cloud américain.

L'objectif est d'ajouter un backend `scaleway` qui utilise l'infrastructure GPU de Scaleway (H100, L40S, A100…) pour faire tourner inférence et entraînement, avec provisionnement automatique via Terraform.

---

## Vision d'ensemble

```
retrain (CPU local)
    │
    ├─ terraform apply ──────────────────→  Instance GPU Scaleway
    │   (cloud-init au boot)                    ├─ Inference engine  :8000  (vLLM par défaut, ou SGLang)
    │                                           ├─ Training API      :8001
    │                                           └─ LoRA weights      (RAM GPU)
    │
    ├─ sample()     ────────────────────→  Inference engine  /v1/chat/completions
    ├─ train_step() ────────────────────→  Training API  /train_step
    ├─ checkpoint() ────────────────────→  Training API  /checkpoint  →  reload LoRA sur inference engine
    └─ fin du run   ────────────────────→  terraform destroy
```

**Pas de GPU local requis.** Tout tourne sur l'instance Scaleway. retrain ne fait que piloter le cycle de vie et envoyer les données.

---

## Expérience utilisateur cible

```toml
[backend]
backend = "scaleway"

[backend.options]
gpu_type        = "l40s"                              # ou "h100", "a100", "L40S-1-48G" (type exact)
zone            = "fr-par-2"
model           = "meta-llama/Llama-3.1-8B-Instruct"
lora_rank       = 32
inference_engine = "vllm"                            # ou "sglang" (voir section dédiée)

[algorithm]
advantage_mode = "maxrl"
transform_mode = "gtpo_sepa"

[training]
max_steps  = 200
batch_size = 4
group_size = 8
```

```bash
export SCW_SECRET_KEY=...
export SCW_DEFAULT_PROJECT_ID=...
retrain
```

retrain provisionne l'instance, attend qu'elle soit prête, entraîne, puis la détruit. L'adaptateur LoRA final est rapatrié localement avant teardown.

---

## Résolution du type de GPU

On supporte deux formes dans `gpu_type` :

| Valeur TOML | Instance Scaleway résolue |
|-------------|--------------------------|
| `"h100"` | `H100-1-80G` |
| `"l40s"` | `L40S-1-48G` |
| `"a100"` | `GPU-A100-S` |
| `"l4"` | `L4-1-24G` |
| `"L40S-1-48G"` | passthrough (type exact) |

Si la valeur n'est pas dans la table, elle est passée telle quelle au Terraform — ce qui permet d'utiliser tout nouveau type Scaleway sans mise à jour de retrain.

---

## Choix du moteur d'inférence

Le backend supporte **vLLM** (défaut) et **SGLang**, configurables via `inference_engine`.
TensorRT-LLM est exclu du MVP : il requiert de compiler un engine TRT au boot (~10–30 min) et le hot-swap LoRA y est significativement plus complexe.

### Comparaison pour le cas d'usage retrain

| Critère | vLLM | SGLang |
|---------|------|--------|
| LoRA hot-swap fiable | ✅ mature (`/v1/load_lora_adapter`) | ⚠️ récent (`/add_lora`) |
| KV cache sur préfixes partagés | ❌ limité | ✅ **RadixAttention** |
| Throughput sampling | 🟡 moyen | ✅ 20–40% supérieur |
| Setup cloud-init | ✅ simple | ✅ simple |
| API OpenAI-compat | ✅ | ✅ |

**Point clé — RadixAttention** : en RL, retrain échantillonne `group_size` completions depuis le **même prompt** (8, 16…). SGLang réutilise le KV cache du préfixe partagé entre toutes les completions d'un groupe, ce qui réduit massivement la mémoire et le temps de sampling. C'est un avantage structurel pour ce workload.

**Recommandation** : démarrer avec **vLLM** pour la maturité du hot-swap LoRA. SGLang devient le choix naturel dès que son API de rechargement LoRA est validée stable — l'architecture le supporte sans changement côté training.

### Rechargement LoRA à chaque `checkpoint()`

```
Training server                Inference engine
      │                               │
      │  sauvegarde poids LoRA        │
      │──────────────────────────────→│ POST /v1/load_lora_adapter  (vLLM)
      │                               │ POST /add_lora               (SGLang)
      │                               │  → moteur recharge les poids en RAM GPU
```

L'endpoint varie selon `inference_engine` mais le protocole côté retrain est identique.

---

## Composants

### 1. Terraform (`retrain/scaleway/terraform/`)

```
main.tf          # scaleway_instance_server + règles réseau (ports 8000/8001)
variables.tf     # instance_type, zone, project_id, image_id, model, lora_rank, inference_engine
outputs.tf       # inference_url, training_url, instance_ip
cloud-init.yaml  # bootstrap : installe le moteur d'inférence + training server au démarrage
```

Le `cloud-init` installe les dépendances et lance, selon `inference_engine` :
- `vllm serve <model> --port 8000 --enable-lora` **(défaut)**
- ou `python -m sglang.launch_server --model <model> --port 8000`
- `retrain-training-server --port 8001 --model <model> --lora-rank <rank>`

### 2. Training Server (`retrain/scaleway/training_server.py`)

Serveur FastAPI minimal qui tourne sur l'instance GPU et expose `TrainHelper` via HTTP. Il instancie `LocalTrainHelper` en interne (PyTorch/PEFT sur le GPU Scaleway).

| Endpoint | Corps | Réponse |
|----------|-------|---------|
| `POST /train_step` | tokens, logprobs, advantages, lr, wd | `{loss: float}` |
| `POST /checkpoint` | name | `{}` — sync poids training → vLLM |
| `POST /save_adapter` | path, name | poids LoRA (bytes) |
| `POST /load_state` | name | `{}` |
| `GET  /health` | — | `{status: "ok"}` |

Packagé comme entrypoint `retrain-training-server` dans `pyproject.toml`.

### 3. `ScalewayTrainHelper` (`retrain/scaleway_backend.py`)

Client Python côté retrain :

1. `__init__` → `TerraformRunner.apply()` (bloque ~2–5 min, log la progression)
2. Attend `/health` sur les deux services
3. Délègue `sample()` vers le moteur d'inférence (OpenAI-compat, identique pour vLLM et SGLang)
4. Délègue `train_step()` / `checkpoint()` vers le Training Server
5. `checkpoint()` déclenche le rechargement LoRA sur le moteur d'inférence via l'endpoint adapté :
   - vLLM : `POST /v1/load_lora_adapter`
   - SGLang : `POST /add_lora`
6. `__del__` + `try/finally` dans la boucle → `TerraformRunner.destroy()`

### 4. `TerraformRunner` (`retrain/scaleway/terraform_runner.py`)

- Résout `gpu_type` → type Scaleway exact
- Lookup de l'image GPU marketplace via API Scaleway (drivers NVIDIA pré-installés)
- `terraform init && terraform apply -auto-approve -json`
- Parse les outputs
- `terraform destroy -auto-approve` au teardown

---

## Capabilities

Identiques au backend `tinker` (remote, synchrone) :

| Capability | `scaleway` | `tinker` | `prime_rl` |
|------------|-----------|---------|------------|
| `reports_sync_loss` | `true` | `true` | `false` |
| `preserves_token_advantages` | `true` | `true` | `false` |
| `supports_checkpoint_resume` | `true` | `true` | `true` |
| `resume_runtime_dependent` | `true` | `true` | `false` |

`resume_runtime_dependent = true` : l'instance est recréée à chaque run, donc les poids en mémoire sont perdus. Le resume passe par un checkpoint sauvegardé localement (ou sur Scaleway Object Storage).

---

## Dépendances

```toml
# pyproject.toml
[project.optional-dependencies]
scaleway = ["fastapi", "uvicorn", "httpx", "scaleway"]
```

```bash
pip install retrain[scaleway]
# + terraform CLI (>= 1.5) dans le PATH
```

---

## Questions ouvertes pour l'équipe

1. **Moteur d'inférence** : vLLM (défaut) ou SGLang ? SGLang offre un gain significatif via RadixAttention sur les workloads RL multi-completions, mais son API de hot-swap LoRA est plus récente. Faut-il valider SGLang sur un run de smoke test avant d'en faire le défaut ?

2. **Sécurité réseau** : les ports 8000/8001 ne doivent pas être exposés publiquement. Quelle approche préférez-vous ?
   - VPC Scaleway (IP privée uniquement) + VPN/bastion pour retrain → plus sûr, plus complexe
   - Tunnel SSH automatique depuis `TerraformRunner` → simple, léger overhead
   - Security group filtré sur l'IP publique de la machine retrain → compromis

3. **État Terraform** : par défaut le state est local (`.terraform/terraform.tfstate`). Pour des équipes avec plusieurs runners simultanés, il faudra un backend Terraform partagé. Scaleway Object Storage est une option naturelle et souveraine — à activer ?

4. **Catalogue GPU** : la table de résolution `gpu_type → instance_type` doit être maintenue. Préférez-vous qu'elle soit hardcodée dans retrain, ou chargée depuis un endpoint API Scaleway au runtime ?

5. **Durée de provisionnement** : ~2–5 min de `terraform apply` avant le premier sample. Acceptable pour les campagnes longues, pénalisant pour les tests courts. Faut-il prévoir un mode "instance persistante" (pas de destroy automatique) ?

6. **Checkpoints entre runs** : avec `resume_runtime_dependent = true`, reprendre un run nécessite que l'adaptateur LoRA ait été sauvegardé localement avant le destroy. Faut-il forcer un `save_adapter` automatique au teardown ?

---

## Fichiers à créer / modifier (résumé)

| Fichier | Nature |
|---------|--------|
| `retrain/scaleway_backend.py` | nouveau |
| `retrain/scaleway/training_server.py` | nouveau |
| `retrain/scaleway/terraform_runner.py` | nouveau |
| `retrain/scaleway/terraform/main.tf` | nouveau |
| `retrain/scaleway/terraform/variables.tf` | nouveau |
| `retrain/scaleway/terraform/outputs.tf` | nouveau |
| `retrain/scaleway/terraform/cloud-init.yaml` | nouveau |
| `retrain/backend_definitions.py` | modifier — ajouter `"scaleway"` |
| `docs/backends.md` | modifier — ajouter section Scaleway |
| `pyproject.toml` | modifier — extra `[scaleway]` + entrypoint |
