#!/bin/bash
# Run this with: bash infra/setup_infra.sh

echo "Setting health check..."
az webapp config set \
    --resource-group RG_LYMPHOID \
    --name LymphoidWebApp \
    --health-check-path "/_stcore/health"

echo "Creating autoscale rules..."
az monitor autoscale create \
    --resource-group RG_LYMPHOID \
    --resource LymphoidWebApp \
    --resource-type Microsoft.Web/sites \
    --name "cpu-autoscale" \
    --min-count 1 --max-count 3 --count 1

az monitor autoscale rule create \
    -g RG_LYMPHOID --autoscale-name cpu-autoscale \
    --condition "Percentage CPU > 70 avg 10m" \
    --scale out 1

echo "Creating health alert..."
az monitor metrics alert create \
    --resource-group RG_LYMPHOID \
    --name "WebApp-Health-Alert" \
    --scopes $(az webapp show -g RG_LYMPHOID -n LymphoidWebApp --query id -o tsv) \
    --condition "count Http2xx == 0" \
    --evaluation-frequency 1m \
    --window-size 2m \
    --action-group "email-ops"
