param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Status', 'Download', 'Stop', 'Start')]
    [string]$Action,
    [Parameter(Mandatory = $true)][string]$RunInfo,
    [string]$Destination = 'checkpoints'
)

$ErrorActionPreference = 'Stop'
$command = Get-Command az -ErrorAction SilentlyContinue
$az = if ($command) { $command.Source } else { 'C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd' }
if (-not (Test-Path -LiteralPath $az)) { throw 'Azure CLI is required.' }
$run = Get-Content -LiteralPath $RunInfo -Raw | ConvertFrom-Json
if ($run.run_id -notmatch '^[A-Za-z0-9][A-Za-z0-9-]+$') { throw 'Invalid run identifier.' }

function Invoke-Azure {
    param([string[]]$Arguments)
    $value = & $az @Arguments --only-show-errors
    if ($LASTEXITCODE -ne 0) { throw "Azure CLI failed: $($Arguments[0..1] -join ' ')" }
    return $value
}

$resource = @('--subscription', $run.subscription_id, '--resource-group', $run.resource_group, '--name', $run.container_name)
if ($Action -eq 'Stop') {
    Invoke-Azure -Arguments (@('container', 'stop') + $resource + @('-o', 'none')) | Out-Null
    Write-Output 'Compute stopped. Azure checkpoints and results are retained.'
    exit 0
}
if ($Action -eq 'Start') {
    if ([DateTimeOffset]::UtcNow -ge [DateTimeOffset]::Parse($run.deadline_utc)) {
        throw 'The original job deadline has passed. Create a new bounded run rather than extending this one silently.'
    }
    Invoke-Azure -Arguments (@('container', 'start') + $resource + @('-o', 'none')) | Out-Null
    Write-Output 'Container started. It restores training or continues final evaluation within the original deadline.'
    exit 0
}

Invoke-Azure -Arguments (@('container', 'show') + $resource + @(
    '--query', '{name:name,state:containers[0].instanceView.currentState,provisioning:provisioningState}', '-o', 'json'
))
$temporary = Join-Path ([IO.Path]::GetTempPath()) ('papplu-download-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $temporary | Out-Null
$previousKey = $env:AZURE_STORAGE_KEY
try {
    $env:AZURE_STORAGE_KEY = Invoke-Azure -Arguments @(
        'storage', 'account', 'keys', 'list', '--subscription', $run.subscription_id,
        '--resource-group', $run.resource_group, '--account-name', $run.storage_account,
        '--query', '[0].value', '-o', 'tsv'
    )
    $storage = @('--account-name', $run.storage_account, '--container-name', $run.blob_container)
    foreach ($name in @('heartbeat.json', 'latest.json')) {
        $exists = Invoke-Azure -Arguments (@('storage', 'blob', 'exists') + $storage + @('--name', $name, '--query', 'exists', '-o', 'tsv'))
        if ($exists -eq 'true') {
            Invoke-Azure -Arguments (@('storage', 'blob', 'download') + $storage + @('--name', $name, '--file', (Join-Path $temporary $name), '-o', 'none')) | Out-Null
        }
    }
    $heartbeat = Join-Path $temporary 'heartbeat.json'
    if (Test-Path -LiteralPath $heartbeat) { Get-Content -LiteralPath $heartbeat -Raw }
    $pointerPath = Join-Path $temporary 'latest.json'
    if (-not (Test-Path -LiteralPath $pointerPath)) {
        if ($Action -eq 'Download') { throw 'No cloud checkpoint has been committed yet.' }
        Write-Output 'No cloud checkpoint has been committed yet.'
        exit 0
    }
    $pointer = Get-Content -LiteralPath $pointerPath -Raw | ConvertFrom-Json
    if ($pointer.run_id -ne $run.run_id) { throw 'Remote checkpoint belongs to a different run.' }
    if ($Action -eq 'Status') {
        $pointer.training_status | ConvertTo-Json -Depth 12
        exit 0
    }
    foreach ($entry in $pointer.files.PSObject.Properties) {
        $name = $entry.Name
        if ($name -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') { throw 'Invalid checkpoint filename.' }
        $path = Join-Path $temporary $name
        Invoke-Azure -Arguments (@('storage', 'blob', 'download') + $storage + @('--name', $entry.Value.blob, '--file', $path, '-o', 'none')) | Out-Null
        if ((Get-Item -LiteralPath $path).Length -ne $entry.Value.bytes) { throw "Checkpoint size mismatch: $name" }
        if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value.sha256) {
            throw "Checkpoint changed or failed verification: $name. Retry the download."
        }
    }
    $target = Join-Path $Destination $run.run_id
    New-Item -ItemType Directory -Force -Path $target | Out-Null
    Get-ChildItem -LiteralPath $temporary -File | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $target $_.Name)
    }
    foreach ($name in @('metrics.jsonl', 'final/summary.json', 'final/best.json', 'final/baseline.json')) {
        $exists = Invoke-Azure -Arguments (@('storage', 'blob', 'exists') + $storage + @('--name', $name, '--query', 'exists', '-o', 'tsv'))
        if ($exists -eq 'true') {
            $file = Join-Path $target ($name.Replace('/', '-'))
            Invoke-Azure -Arguments (@('storage', 'blob', 'download') + $storage + @('--name', $name, '--file', $file, '--overwrite', 'true', '-o', 'none')) | Out-Null
        }
    }
    Write-Output "Verified checkpoints downloaded to $target"
} finally {
    if ($null -eq $previousKey) {
        Remove-Item Env:\AZURE_STORAGE_KEY -ErrorAction SilentlyContinue
    } else {
        $env:AZURE_STORAGE_KEY = $previousKey
    }
    Remove-Item -LiteralPath $temporary -Recurse -Force
}
