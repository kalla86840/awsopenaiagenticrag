param(
    [Parameter(Mandatory = $true)]
    [string]$OpenAIApiKey,
    [string]$SecretName = "openai/api-key",
    [string]$Region = "us-west-1"
)

aws secretsmanager create-secret `
    --region $Region `
    --name $SecretName `
    --secret-string $OpenAIApiKey
