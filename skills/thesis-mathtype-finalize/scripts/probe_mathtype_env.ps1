param(
    [string]$TemplatePath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Resolve-TemplatePath {
    param([string]$CandidatePath)

    if ($CandidatePath) {
        if (-not (Test-Path -LiteralPath $CandidatePath)) {
            throw "MathType template not found: $CandidatePath"
        }
        return (Resolve-Path -LiteralPath $CandidatePath).Path
    }

    $candidates = @(@(
        (Join-Path $env:ProgramFiles 'Microsoft Office\root\Office16\STARTUP\MathType Commands 2016.dotm'),
        (Join-Path $env:ProgramFiles 'Microsoft Office\root\Office16\STARTUP\MathType Commands for Word.dotm'),
        (Join-Path ${env:ProgramFiles(x86)} 'Microsoft Office\Office16\STARTUP\MathType Commands 2016.dotm'),
        (Join-Path $env:APPDATA 'Microsoft\Word\STARTUP\MathType Commands 2016.dotm')
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) })

    if (-not $candidates) {
        throw 'No MathType Word template was found in the standard startup paths.'
    }

    return $candidates[0]
}

function Get-MacroNamesFromTemplate {
    param([string]$Path)

    Add-Type -AssemblyName System.IO.Compression.FileSystem

    $tempZip = Join-Path $env:TEMP ("mathtype-probe-{0}.zip" -f ([guid]::NewGuid().ToString('N')))
    $tempDir = Join-Path $env:TEMP ("mathtype-probe-{0}" -f ([guid]::NewGuid().ToString('N')))

    try {
        Copy-Item -LiteralPath $Path -Destination $tempZip -Force
        [System.IO.Compression.ZipFile]::ExtractToDirectory($tempZip, $tempDir)

        $vbaDataPath = Join-Path $tempDir 'word\vbaData.xml'
        if (-not (Test-Path -LiteralPath $vbaDataPath)) {
            return @()
        }

        [xml]$xml = Get-Content -LiteralPath $vbaDataPath
        $all = @($xml.SelectNodes("//*[local-name()='mcd']") | ForEach-Object {
            @($_.Attributes) | Where-Object { $_.LocalName -in @('name', 'macroName') } | ForEach-Object { $_.Value }
        }) | Where-Object { $_ }

        return $all | Where-Object {
            $_ -match 'MTConvertEquations|MTCommand_ConvertEqns|ConvertEqn'
        } | Select-Object -Unique
    }
    finally {
        if (Test-Path -LiteralPath $tempZip) {
            Remove-Item -LiteralPath $tempZip -Force
        }
        if (Test-Path -LiteralPath $tempDir) {
            Remove-Item -LiteralPath $tempDir -Recurse -Force
        }
    }
}

$result = [ordered]@{
    template_found = $false
    template_path = $null
    word_com_ok = $false
    word_version = $null
    word_error = $null
    macro_candidates = @()
}

try {
    $resolvedTemplate = Resolve-TemplatePath -CandidatePath $TemplatePath
    $result.template_found = $true
    $result.template_path = $resolvedTemplate
    $result.macro_candidates = @(Get-MacroNamesFromTemplate -Path $resolvedTemplate)
}
catch {
    $result.word_error = $_.Exception.Message
}

$word = $null
try {
    $word = New-Object -ComObject Word.Application
    $result.word_com_ok = $true
    $result.word_version = $word.Version
}
catch {
    if (-not $result.word_error) {
        $result.word_error = $_.Exception.Message
    }
}
finally {
    if ($word) {
        $word.Quit()
    }
}

[pscustomobject]$result | ConvertTo-Json -Depth 4
