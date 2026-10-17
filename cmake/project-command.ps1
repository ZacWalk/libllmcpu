#requires -Version 7.4
# dd project command backing script for nnc.
#
# Downloads the same catalogue as the app, defined once in
# src/appnnc/model_catalog.def.
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

$request = [Console]::In.ReadLine() | ConvertFrom-Json -AsHashtable
$root = Split-Path $PSScriptRoot
if ($request.schema -ne 1 -or $request.projectRoot -ne $root -or $request.parameters -isnot [hashtable] -or
    $request.dryRun -isnot [bool] -or $request.command -notin @('download')) {
    throw 'Invalid project-command request.'
}
Set-Location $root

$models = [ordered]@{}
$catalogFile = Join-Path $root 'src\appnnc\model_catalog.def'
$catalogLines = Get-Content -LiteralPath $catalogFile | Where-Object { $_ -match '^\s*NNC_MODEL\(' }
if (-not $catalogLines) { throw "No model entries found in '$catalogFile'." }
foreach ($line in $catalogLines) {
    $match = [regex]::Match($line, '^NNC_MODEL\("([^"]+)", "([^"]+)", "([^"]+)", ([0-9]+), "([^"]+)"\)$')
    if (-not $match.Success) { throw "Cannot parse model catalogue entry: $line" }
    $bytes = [uint64]::Parse($match.Groups[4].Value, [Globalization.CultureInfo]::InvariantCulture)
    $models[$match.Groups[1].Value] = @{
        File = $match.Groups[2].Value
        Url = $match.Groups[3].Value
        Bytes = $bytes
        Size = ('{0:N1} GB' -f ($bytes / 1GB))
        Note = $match.Groups[5].Value
    }
}

$requested = if ($request.parameters.name) { [string]$request.parameters.name } else { '' }

$inventory = foreach ($key in $models.Keys) {
    $model = $models[$key]
    $path = Join-Path 'models' $model.File
    $present = (Test-Path -LiteralPath $path -PathType Leaf) -and
        (Get-Item -LiteralPath $path).Length -eq $model.Bytes
    @{ name = $key; file = $model.File; size = $model.Size; note = $model.Note
        present = [bool]$present; path = $path }
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
    $present = (Test-Path -LiteralPath $path -PathType Leaf) -and
        (Get-Item -LiteralPath $path).Length -eq $model.Bytes
    @{ name = $name; file = $model.File; size = $model.Size; url = $model.Url
        path = $path; present = [bool]$present; expected_bytes = $model.Bytes }
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
        if ((Get-Item -LiteralPath $temporary).Length -ne $entry.expected_bytes) {
            throw "Downloaded '$($entry.file)' has an unexpected size."
        }
        $stream = [IO.File]::OpenRead($temporary)
        try {
            $magic = [byte[]]::new(4)
            if ($stream.Read($magic, 0, 4) -ne 4 -or [Text.Encoding]::ASCII.GetString($magic) -ne 'GGUF') {
                throw "Downloaded '$($entry.file)' is not a GGUF file."
            }
        }
        finally {
            $stream.Dispose()
        }
        Move-Item -LiteralPath $temporary -Destination $entry.path -Force
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
