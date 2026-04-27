# Scaleway Backend — Implementation Status

## Steps

| # | Description | Status | Fichier(s) |
|---|-------------|--------|------------|
| 1 | `TerraformRunner` | ✅ Done | `retrain/scaleway/terraform_runner.py` |
| 2 | Fichiers Terraform | ✅ Done | `retrain/scaleway/terraform/main.tf`, `variables.tf`, `outputs.tf`, `cloud-init.yaml` |
| 3 | Training Server (FastAPI) | ✅ Done | `retrain/scaleway/training_server.py` |
| 4 | `ScalewayTrainHelper` | ✅ Done | `retrain/scaleway_backend.py` |
| 5 | Enregistrement backend | ✅ Done | `retrain/backend_definitions.py` |
| 6 | `pyproject.toml` | ✅ Done | `pyproject.toml` |
| 7 | Test contract | ✅ Done (non exécuté) | `tests/test_backend_contract.py` |

## À valider

- [ ] `pytest tests/test_backend_contract.py::test_scaleway_backend_contract` passe
- [ ] `terraform validate` sur `retrain/scaleway/terraform/`
- [ ] Run end-to-end sur Scaleway avec `campaigns/sroie.toml`

## Questions ouvertes (du design doc)

Les questions du RFC restent ouvertes et n'ont pas été tranchées dans l'implémentation :

1. **Moteur d'inférence par défaut** — vLLM actuellement, SGLang à valider via smoke test
2. **Sécurité réseau** — ports 8000/8001 ouverts publiquement pour le MVP, à restreindre (VPC / SSH tunnel / security group IP source)
3. **État Terraform** — local par défaut, Scaleway Object Storage à activer pour multi-runners
4. **Catalogue GPU** — table hardcodée dans `TerraformRunner.GPU_TYPE_MAP`, API Scaleway non consultée
5. **Instance persistante** — pas de mode "no destroy" pour les tests courts
6. **Save adapter au teardown** — pas de `save_adapter` automatique avant `terraform destroy`
