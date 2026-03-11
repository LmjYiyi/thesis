param(
    [Parameter(Mandatory = $true)]
    [string]$InputDocx,

    [string]$OutputDocx,

    [string]$TemplatePath,

    [switch]$KeepWordOpen,

    [switch]$ManualSave
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-TemplateCandidates {
    $candidates = @(
        (Join-Path $env:ProgramFiles 'Microsoft Office\root\Office16\STARTUP\MathType Commands 2016.dotm'),
        (Join-Path $env:ProgramFiles 'Microsoft Office\root\Office16\STARTUP\MathType Commands for Word.dotm'),
        (Join-Path $env:ProgramFiles 'Microsoft Office\root\Office16\STARTUP\MathType Commands 6 For Word 2013.dotm'),
        (Join-Path ${env:ProgramFiles(x86)} 'Microsoft Office\root\Office16\STARTUP\MathType Commands 2016.dotm'),
        (Join-Path ${env:ProgramFiles(x86)} 'Microsoft Office\root\Office16\STARTUP\MathType Commands for Word.dotm'),
        (Join-Path ${env:ProgramFiles(x86)} 'Microsoft Office\root\Office16\STARTUP\MathType Commands 6 For Word 2013.dotm'),
        (Join-Path ${env:ProgramFiles(x86)} 'Microsoft Office\Office16\STARTUP\MathType Commands 2016.dotm'),
        (Join-Path ${env:ProgramFiles(x86)} 'Microsoft Office\Office16\STARTUP\MathType Commands for Word.dotm'),
        (Join-Path ${env:ProgramFiles(x86)} 'Microsoft Office\Office16\STARTUP\MathType Commands 6 For Word 2013.dotm'),
        (Join-Path $env:APPDATA 'Microsoft\Word\STARTUP\MathType Commands 2016.dotm'),
        (Join-Path $env:APPDATA 'Microsoft\Word\STARTUP\MathType Commands for Word.dotm'),
        (Join-Path $env:APPDATA 'Microsoft\Word\STARTUP\MathType Commands 6 For Word 2013.dotm')
    ) | Where-Object { $_ }

    return $candidates | Select-Object -Unique
}

function Resolve-TemplatePath {
    param([string]$CandidatePath)

    if ($CandidatePath) {
        if (-not (Test-Path -LiteralPath $CandidatePath)) {
            throw "MathType template not found: $CandidatePath"
        }
        return (Resolve-Path -LiteralPath $CandidatePath).Path
    }

    $candidates = @(Get-TemplateCandidates | Where-Object { Test-Path -LiteralPath $_ })

    if (-not $candidates) {
        throw 'No MathType Word template was found in the known startup paths. Pass -TemplatePath to override the auto-detection result.'
    }

    return $candidates[0]
}

function Get-DefaultOutputPath {
    param([string]$InputPath)

    $directory = Split-Path -Parent $InputPath
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($InputPath)
    return Join-Path $directory ($stem + '_mt.docx')
}

function Get-OmmlEquationCount {
    param([string]$DocxPath)

    Add-Type -AssemblyName System.IO.Compression.FileSystem

    $zip = [System.IO.Compression.ZipFile]::OpenRead($DocxPath)
    try {
        $entry = $zip.GetEntry('word/document.xml')
        if (-not $entry) {
            return 0
        }

        $reader = New-Object System.IO.StreamReader($entry.Open())
        try {
            $xml = $reader.ReadToEnd()
        }
        finally {
            $reader.Dispose()
        }

        return ([regex]::Matches($xml, '<m:oMath\b')).Count
    }
    finally {
        $zip.Dispose()
    }
}

function New-WordAutomationVbs {
    param(
        [string]$TemplateFullPath,
        [string]$DocxFullPath,
        [bool]$LeaveWordOpen,
        [bool]$UseManualSave
    )

    return @"
Option Explicit

Dim templatePath, docPath, keepWordOpen, manualSave
templatePath = WScript.Arguments(0)
docPath = WScript.Arguments(1)
keepWordOpen = (LCase(WScript.Arguments(2)) = "true")
manualSave = (LCase(WScript.Arguments(3)) = "true")

Dim wordApp, doc, addinObj, addins, macroUsed, macroErrors
Set wordApp = CreateObject("Word.Application")
If Err.Number <> 0 Then
  WScript.StdErr.WriteLine "CreateObject failed: " & Err.Description
  WScript.Quit 20
End If

On Error Resume Next
wordApp.Visible = True
wordApp.DisplayAlerts = 0

Set addins = wordApp.AddIns
If Err.Number <> 0 Then
  WScript.StdErr.WriteLine "Access AddIns failed: " & Err.Description
  WScript.Quit 21
End If

Dim found
found = False
For Each addinObj In addins
  If LCase(addinObj.Name) = LCase(CreateObject("Scripting.FileSystemObject").GetFileName(templatePath)) Then
    addinObj.Installed = True
    found = True
    Exit For
  End If
Next

If Not found Then
  Set addinObj = addins.Add(templatePath, True)
  If Err.Number <> 0 Then
    WScript.StdErr.WriteLine "AddIn load failed: " & Err.Description
    WScript.Quit 22
  End If
  addinObj.Installed = True
End If

Set doc = wordApp.Documents.Open(docPath, False, False, False, "", "", False, "", "", 0, 0, True, True)
If Err.Number <> 0 Then
  WScript.StdErr.WriteLine "Open document failed: " & Err.Description
  WScript.Quit 23
End If

doc.Activate
doc.Range.Select

macroUsed = ""
macroErrors = ""

Err.Clear
wordApp.Run "MathTypeCommands.UILib.MTCommand_ConvertEqns"
If Err.Number = 0 Then
  macroUsed = "MathTypeCommands.UILib.MTCommand_ConvertEqns"
Else
  macroErrors = macroErrors & "MathTypeCommands.UILib.MTCommand_ConvertEqns => " & Err.Number & ": " & Err.Description & vbCrLf
End If

If macroUsed = "" Then
  Err.Clear
  wordApp.Run "MathTypeCommands.MTConvertEquations.DlgMain"
  If Err.Number = 0 Then
    macroUsed = "MathTypeCommands.MTConvertEquations.DlgMain"
  Else
    macroErrors = macroErrors & "MathTypeCommands.MTConvertEquations.DlgMain => " & Err.Number & ": " & Err.Description & vbCrLf
  End If
End If

If macroUsed = "" Then
  Err.Clear
  wordApp.Run "MathTypeCommands.MTCommandsDispatchCls.MTCommandsMain_MTConvertEquations_DlgMain"
  If Err.Number = 0 Then
    macroUsed = "MathTypeCommands.MTCommandsDispatchCls.MTCommandsMain_MTConvertEquations_DlgMain"
  Else
    macroErrors = macroErrors & "MathTypeCommands.MTCommandsDispatchCls.MTCommandsMain_MTConvertEquations_DlgMain => " & Err.Number & ": " & Err.Description & vbCrLf
  End If
End If

If macroUsed = "" Then
  WScript.StdErr.WriteLine "MathType conversion macro failed."
  WScript.StdErr.WriteLine macroErrors
  WScript.Quit 24
End If

If Not manualSave Then
  doc.Save
  If Err.Number <> 0 Then
    WScript.StdErr.WriteLine "Save failed: " & Err.Description
    WScript.Quit 25
  End If
End If

If Not keepWordOpen And Not manualSave Then
  doc.Close False
  wordApp.Quit
End If

WScript.StdOut.WriteLine "macro_used=" & macroUsed
If manualSave Then
  WScript.StdOut.WriteLine "working_doc=" & docPath
End If
WScript.Quit 0
"@
}

$resolvedInput = (Resolve-Path -LiteralPath $InputDocx).Path
if ([System.IO.Path]::GetExtension($resolvedInput).ToLowerInvariant() -ne '.docx') {
    throw 'Only .docx files are supported.'
}

$resolvedTemplate = Resolve-TemplatePath -CandidatePath $TemplatePath

if (-not $OutputDocx) {
    $OutputDocx = Get-DefaultOutputPath -InputPath $resolvedInput
}

if ([System.IO.Path]::GetExtension($OutputDocx).ToLowerInvariant() -ne '.docx') {
    throw 'The output path must end with .docx.'
}

$resolvedOutput = [System.IO.Path]::GetFullPath($OutputDocx)
$tempWorkingDocx = Join-Path $env:TEMP ("mathtype-working-{0}.docx" -f ([guid]::NewGuid().ToString('N')))
Copy-Item -LiteralPath $resolvedInput -Destination $tempWorkingDocx -Force

$beforeCount = Get-OmmlEquationCount -DocxPath $tempWorkingDocx

$tempVbsPath = Join-Path $env:TEMP ("mathtype-convert-{0}.vbs" -f ([guid]::NewGuid().ToString('N')))
$macroUsed = $null
$workingDocPath = $tempWorkingDocx

try {
    $vbsContent = New-WordAutomationVbs -TemplateFullPath $resolvedTemplate -DocxFullPath $tempWorkingDocx -LeaveWordOpen:$KeepWordOpen -UseManualSave:$ManualSave
    [System.IO.File]::WriteAllText($tempVbsPath, $vbsContent, [System.Text.Encoding]::ASCII)

    Write-Host 'Word is opening the MathType conversion dialog.'
    Write-Host 'In Word, choose: Whole document -> OMML equations -> MathType equations.'
    if ($ManualSave) {
        Write-Host "After conversion, save manually from Word. Suggested output path: $resolvedOutput"
    }

    $rawOutput = & cscript.exe //nologo $tempVbsPath $resolvedTemplate $tempWorkingDocx ($KeepWordOpen.IsPresent.ToString()) ($ManualSave.IsPresent.ToString()) 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw (($rawOutput | ForEach-Object { $_.ToString() }) -join [Environment]::NewLine)
    }

    $macroLine = @($rawOutput | Where-Object { $_ -like 'macro_used=*' }) | Select-Object -First 1
    if ($macroLine) {
        $macroUsed = $macroLine.Substring('macro_used='.Length)
    }
    $workingLine = @($rawOutput | Where-Object { $_ -like 'working_doc=*' }) | Select-Object -First 1
    if ($workingLine) {
        $workingDocPath = $workingLine.Substring('working_doc='.Length)
    }
}
finally {
    if (Test-Path -LiteralPath $tempVbsPath) {
        Remove-Item -LiteralPath $tempVbsPath -Force
    }
}

$afterCount = Get-OmmlEquationCount -DocxPath $tempWorkingDocx
if ((-not $KeepWordOpen) -and (-not $ManualSave)) {
    Copy-Item -LiteralPath $tempWorkingDocx -Destination $resolvedOutput -Force
    if (Test-Path -LiteralPath $tempWorkingDocx) {
        Remove-Item -LiteralPath $tempWorkingDocx -Force -ErrorAction SilentlyContinue
    }
}

[pscustomobject]@{
    input = $resolvedInput
    output = $resolvedOutput
    template = $resolvedTemplate
    macro_used = $macroUsed
    manual_save = $ManualSave.IsPresent
    working_doc = $workingDocPath
    omml_before = $beforeCount
    omml_after = $afterCount
    likely_converted = ($beforeCount -gt 0 -and $afterCount -eq 0)
} | ConvertTo-Json -Depth 3
