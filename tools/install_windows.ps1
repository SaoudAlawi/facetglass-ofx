# Installs the built FacetGlass.ofx.bundle into the system OFX plugin folder.
# Needs admin rights: run from an elevated PowerShell, or let it re-launch itself elevated.
#   powershell -ExecutionPolicy Bypass -File tools\install_windows.ps1 [-Build build]
param([string]$Build = "build")

$ErrorActionPreference = "Stop"
$src = Join-Path (Resolve-Path $Build) "FacetGlass.ofx.bundle"
$dstRoot = "C:\Program Files\Common Files\OFX\Plugins"
$dst = Join-Path $dstRoot "FacetGlass.ofx.bundle"

if (-not (Test-Path (Join-Path $src "Contents\Win64\FacetGlass.ofx"))) {
    throw "No built plugin at $src\Contents\Win64\FacetGlass.ofx - build first."
}

$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) {
    Write-Host "Requesting admin rights to write to $dstRoot ..."
    $p = Start-Process powershell -Verb RunAs -Wait -PassThru -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$PSCommandPath`"", "-Build", "`"$((Resolve-Path $Build).Path)`"")
    exit $p.ExitCode
}

if (Get-Process -Name Resolve -ErrorAction SilentlyContinue) {
    Write-Warning "DaVinci Resolve is running: it keeps the old plugin loaded. Quit and restart it after installing."
}

# Replace only our own bundle folder.
if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
Copy-Item -Recurse $src $dst
$f = Get-Item (Join-Path $dst "Contents\Win64\FacetGlass.ofx")
Write-Host "Installed $($f.FullName) ($($f.Length) bytes, $($f.LastWriteTime))"
