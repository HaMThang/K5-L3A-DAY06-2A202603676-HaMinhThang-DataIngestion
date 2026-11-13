param(
    [string]$Python = 'python',
    [string]$DockerContainer = '',
    [string]$RunPrefix = 'openai',
    [string]$LlmModel = 'gpt-5.6-terra'
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    foreach ($variant in @('small', 'large')) {
        $runId = $RunPrefix + '_' + $variant
        $arguments = @('--run-id', $runId, '--provider', 'openai', '--model', ('text-embedding-3-' + $variant), '--generate-answers', '--llm-model', $LlmModel)
        foreach ($stage in @('ingest', 'dev', 'select', 'test')) {
            if ($DockerContainer) {
                & docker exec $DockerContainer python -m src.run --stage $stage @arguments
            } else {
                & $Python -m src.run --stage $stage @arguments
            }
            if ($LASTEXITCODE -ne 0) { throw "Pipeline failed at $runId / $stage; inspect results/$runId. No fallback used." }
        }
    }
    if ($DockerContainer) {
        & docker exec $DockerContainer python -m src.run --stage compare --run-id ($RunPrefix + '_comparison') --compare-runs ($RunPrefix + '_small') ($RunPrefix + '_large')
    } else {
        & $Python -m src.run --stage compare --run-id ($RunPrefix + '_comparison') --compare-runs ($RunPrefix + '_small') ($RunPrefix + '_large')
    }
    if ($LASTEXITCODE -ne 0) { throw 'Comparison failed' }
} finally {
    Pop-Location
}
