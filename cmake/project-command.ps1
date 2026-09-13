#requires -Version 7.4
# dd project command backing script for nnc.
#
# Ported from the pre-dd bespoke driver. Downloads the GGUF weight files nnc can
# actually load: BF16/F16/F32 straight from the mmap, or Q4_K/Q5_K/Q6_K decoded at
# load time. Q4_0/Q8_0/IQ* files are rejected by the loader and are not listed.
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

$request = [Console]::In.ReadLine() | ConvertFrom-Json -AsHashtable
$root = Split-Path $PSScriptRoot
if ($request.schema -ne 1 -or $request.projectRoot -ne $root -or $request.parameters -isnot [hashtable] -or
    $request.dryRun -isnot [bool] -or $request.command -notin @('download')) {
    throw 'Invalid project-command request.'
}
Set-Location $root

$models = [ordered]@{
    'gemma-3-1b-bf16' = @{
        File = 'gemma-3-1b-it-BF16.gguf'
        Url  = 'https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-BF16.gguf'
        Size = '1.9 GB'
        Note = 'smallest end-to-end test; exercises the BF16 -> Q8_0 path'
    }
    'gemma-3-1b-q6k' = @{
        File = 'gemma-3-1b-it-Q6_K.gguf'
        Url  = 'https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q6_K.gguf'
        Size = '1.0 GB'
        Note = 'exercises the Q6_K -> Q8_0 load-time decode'
    }
    'gemma-3-4b-q4km' = @{
        File = 'gemma-3-4b-it-Q4_K_M.gguf'
        Url  = 'https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf'
        Size = '2.5 GB'
        Note = 'exercises the Q4_K -> Q8_0 load-time decode'
    }
}

$requested = if ($request.parameters.name) { [string]$request.parameters.name } else { '' }

$inventory = foreach ($key in $models.Keys) {
    $model = $models[$key]
    $path = Join-Path 'models' $model.File
    @{ name = $key; file = $model.File; size = $model.Size; note = $model.Note
        present = [bool](Test-Path -LiteralPath $path); path = $path }
}

# No name lists the catalogue rather than fetching gigabytes by accident.
if (-not $requested) {
    @{ schema = 1; data = @{ action = 'list'; models = @($inventory) }; files = @() } |
        ConvertTo-Json -Depth 10 -Compress | Write-Output
    exit 0
}

$names = if ($requested -eq 'all') { @($models.Keys) } else { @($requested) }
foreach ($name in $names) {
    if (-not $models.Contains($name)) { throw "Unknown model: $name. Run dd download with no --name to list them." }
}

$planned = foreach ($name in $names) {
    $model = $models[$name]
    $path = Join-Path 'models' $model.File
    @{ name = $name; file = $model.File; size = $model.Size; url = $model.Url
        path = $path; present = [bool](Test-Path -LiteralPath $path) }
}

if ($request.dryRun) {
    @{ schema = 1; data = @{ action = 'download'; status = 'planned'; models = @($planned) }; files = @() } |
        ConvertTo-Json -Depth 10 -Compress | Write-Output
    exit 0
}

$results = @()
$written = @()
foreach ($entry in $planned) {
    if ($entry.present) {
        $mb = [math]::Round((Get-Item -LiteralPath $entry.path).Length / 1MB, 1)
        $results += @{ name = $entry.name; status = 'present'; path = $entry.path; megabytes = $mb }
        continue
    }
    New-Item -ItemType Directory -Force -Path 'models' | Out-Null
    $temporary = "$($entry.path).part"
    $previous = $ProgressPreference
    $ProgressPreference = 'SilentlyContinue'
    try {
        Invoke-WebRequest -Uri $entry.url -OutFile $temporary
        Move-Item -LiteralPath $temporary -Destination $entry.path
    }
    finally {
        $ProgressPreference = $previous
        if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary -Force }
    }
    $mb = [math]::Round((Get-Item -LiteralPath $entry.path).Length / 1MB, 1)
    $results += @{ name = $entry.name; status = 'downloaded'; path = $entry.path; megabytes = $mb }
    $written += $entry.path
}

@{ schema = 1; data = @{ action = 'download'; status = 'complete'; models = @($results) }; files = @($written) } |
    ConvertTo-Json -Depth 10 -Compress | Write-Output
