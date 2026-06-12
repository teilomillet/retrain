.SILENT:
.DEFAULT_GOAL := help

help:
	echo "Please use \`make \033[36m<target>\033[0m\`"
	echo "\t where \033[36m<target>\033[0m is one of"
	grep -E '^\.PHONY: [a-zA-Z_-]+ .*?## .*$$' $(MAKEFILE_LIST) \
		| sort | awk 'BEGIN {FS = "(: |##)"}; {printf "• \033[36m%-30s\033[0m %s\n", $$2, $$3}'

# ==============================================================================
# scw CLI
# ==============================================================================

.PHONY: scw-server-list ## 📋 List Scaleway servers
scw-server-list:
	scw instance server list

.PHONY: scw-server-start ## 🚀 Start the Scaleway server (make scw-server-start ID=<server_id>)
scw-server-start:
	scw instance server start $(ID)

.PHONY: scw-server-stop ## 🛑 Stop the Scaleway server (make scw-server-stop ID=<server_id>)
scw-server-stop:
	scw instance server stop $(ID)

# ==============================================================================
# sops
# ==============================================================================

.PHONY: sops-decrypt ## 🔐 Decrypt terraform.tfvars.sops to terraform.tfvars
sops-decrypt:
	sops -d --input-type binary --output-type binary terraform.tfvars.sops > terraform.tfvars

.PHONY: sops-encrypt ## 🔐 Encrypt terraform.tfvars to terraform.tfvars.sops
sops-encrypt:
	sops -e --input-type binary --output-type binary terraform.tfvars > terraform.tfvars.sops
