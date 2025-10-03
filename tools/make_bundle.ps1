<#
make_bundle.ps1

Usage examples:
  # 1) If you already have a license file and just want to pack:
  .\make_bundle.ps1 -Client 52431980 -SourceDir "D:\trading-bot new" -EditorPath "D:\bundle_sources\Editor.exe" -PublicPem "D:\bundle_sources\public.pem"

  # 2) If you want script to call your generate_license.py first:
  .\make_bundle.ps1 -Client 52431980 -Expiry "2025-10-09" -RunGenerateLicense

Note: adjust paths below or pass parameters. Script is defensive and prints progress.
#>

param(
    [Parameter(Mandatory=$true)][string]$Client,                     # client id, e.g. 52431980
    [string]$SourceDir = "D:\trading-bot new",                       # root of your project (where tools\generate_license.py sits)
    [string]$EditorPath = "D:\bundle_sources\Editor.exe",            # path to Editor.exe you want to include
    [string]$BotPath = "",                                           # optional Bot.exe path (leave empty if not used)
    [string]$PublicPem = "D:\bundle_sources\public.pem",             # path to public.pem
    [switch]$RunGenerateLicense,                                     # if set, script will try to run python generate_license.py
    [string]$Expiry = "",                                            # e.g. "2025-10-09" (only used if RunGenerateLicense)
    [string]$OutRoot = "D:\bundle_out",                              # where to put final bundle
    [string]$IssFile = "D:\make_installer.iss",                      # optional Inno .iss (if you want the installer built)
    [switch]$BuildInstaller                                           # set to call Inno Setup to compile the .iss if ISCC found
)

function Fail($msg){ Write-Error $msg; exit 1 }

Write-Host "=== make_bundle.ps1 starting for client $Client ===" -ForegroundColor Cyan

# 1. verify sources exist
if (-not (Test-Path $EditorPath)) { Fail "Editor.exe not found at: $EditorPath" }
if (-not (Test-Path $PublicPem)) { Fail "public.pem not found at: $PublicPem" }
if ($BotPath -and (-not (Test-Path $BotPath))) { Write-Warning "Bot.exe path provided but not found: $BotPath; will continue without it." ; $BotPath = "" }

# 2. optionally run generate_license.py
$licenseFile = $null
if ($RunGenerateLicense) {
    $py = "python"
    # find the generate_license script inside SourceDir\tools or SourceDir\tools\generate_license.py
    $genPyCandidates = @(
        Join-Path $SourceDir "tools\generate_license.py",
        Join-Path $SourceDir "tools\generate_license.py"
    )
    $genPy = $genPyCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $genPy) { Fail "generate_license.py not found under $SourceDir\tools. Put it there or run without -RunGenerateLicense." }

    if (-not $Expiry) { Fail "You set -RunGenerateLicense but did not provide -Expiry (YYYY-MM-DD)." }

    Write-Host "Running generate_license.py to create license for client $Client (expiry $Expiry) ..." -ForegroundColor Yellow
    $genArgs = @($genPy, "--client", $Client, "--account", $Client, "--expiry", $Expiry)
    # run python and forward stdout/stderr
    $proc = Start-Process -FilePath $py -ArgumentList $genArgs -NoNewWindow -PassThru -Wait -RedirectStandardOutput ([System.IO.Path]::GetTempFileName()) -RedirectStandardError ([System.IO.Path]::GetTempFileName())
    if ($proc.ExitCode -ne 0) {
        Write-Warning "generate_license.py returned code $($proc.ExitCode). Attempting to locate an existing license file instead."
    } else {
        Write-Host "generate_license.py finished (exit $($proc.ExitCode))."
    }
    # try to locate latest license file in SourceDir or tools directory
    $found = Get-ChildItem -Path (Join-Path $SourceDir "tools") -Filter "license_*.json" -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $found) {
        Write-Warning "No license_*.json found in $SourceDir\tools. Searching project root..."
        $found = Get-ChildItem -Path $SourceDir -Filter "license_*.json" -File -Recurse -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    }
    if ($found) {
        $licenseFile = $found.FullName
        Write-Host "Found generated license file: $licenseFile"
    } else {
        Write-Warning "Could not find a generated license file after running generate_license.py."
    }
}

# 3. if license not yet provided/found - try to find any license_*.json in SourceDir
if (-not $licenseFile) {
    $found2 = Get-ChildItem -Path $SourceDir -Filter "license_*.json" -File -Recurse -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($found2) {
        $licenseFile = $found2.FullName
        Write-Host "Using existing license file: $licenseFile"
    } else {
        Write-Warning "No license_*.json found automatically. If you already have a license, create/copy it into SourceDir or tools folder and re-run the script."
    }
}

if (-not $licenseFile) {
    Write-Warning "No license file found. Script will continue but bundle will be missing license."
}

# 4. prepare output bundle folder
$clientDir = Join-Path $OutRoot $Client
if (Test-Path $clientDir) { Remove-Item -Recurse -Force $clientDir }
New-Item -ItemType Directory -Path $clientDir -Force | Out-Null
Write-Host "Created bundle folder: $clientDir"

# 5. copy required files
Copy-Item -Path $EditorPath -Destination $clientDir -Force
Write-Host "Copied Editor.exe"

if ($BotPath) {
    Copy-Item -Path $BotPath -Destination $clientDir -Force
    Write-Host "Copied Bot.exe"
}

Copy-Item -Path $PublicPem -Destination $clientDir -Force
Write-Host "Copied public.pem"

# 6. copy license if found; rename to .lic (optional)
if ($licenseFile) {
    $licBase = [System.IO.Path]::GetFileNameWithoutExtension($licenseFile)
    $licDestJson = Join-Path $clientDir ([System.IO.Path]::GetFileName($licenseFile))
    Copy-Item -Path $licenseFile -Destination $licDestJson -Force
    # also create a .lic copy if you want the Editor to prefer .lic
    $licDest = Join-Path $clientDir ($licBase + ".lic")
    Copy-Item -Path $licenseFile -Destination $licDest -Force
    Write-Host "Copied license as JSON and .lic: $licDestJson , $licDest"
} else {
    Write-Warning "No license copied (not found)."
}

# 7. create a ZIP archive
$zipName = Join-Path $OutRoot ("bundle_" + $Client + ".zip")
if (Test-Path $zipName) { Remove-Item $zipName -Force }
Compress-Archive -Path (Join-Path $clientDir "*") -DestinationPath $zipName -Force
Write-Host "Created zip bundle: $zipName" -ForegroundColor Green

# 8. optionally build installer with Inno Setup if requested
if ($BuildInstaller) {
    # check for ISCC.exe
    Write-Host "Attempting to find Inno Setup (ISCC.exe)..." -ForegroundColor Yellow
    $iscc = Get-ChildItem -Path "C:\Program Files","C:\Program Files (x86)" -Filter ISCC.exe -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
    if (-not $iscc) {
        Write-Warning "ISCC.exe not found on system. Install Inno Setup or run the .iss in GUI. Skipping installer build."
    } else {
        if (-not (Test-Path $IssFile)) { Write-Warning "Specified .iss file not found: $IssFile. Skipping installer build." }
        else {
            Write-Host "Found ISCC: $iscc. Building installer using $IssFile ..."
            & $iscc $IssFile
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Inno Setup compiled successfully." -ForegroundColor Green
            } else {
                Write-Warning "Inno Setup returned exit code $LASTEXITCODE. Check the .iss or Inno log for details."
            }
        }
    }
}

Write-Host "=== Bundle complete. Output folder: $clientDir and $zipName ===" -ForegroundColor Cyan
