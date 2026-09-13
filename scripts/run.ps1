param([string]$Python='python')
$ErrorActionPreference='Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
& $Python -m careseq.cli --root $projectRoot run
if ($LASTEXITCODE -ne 0) { throw "Échec de l'expérience : code $LASTEXITCODE" }
