param(
  [Parameter(Mandatory=$true)][string]$Client,     # ex: 52431980 (أو اسم قصير)
  [Parameter(Mandatory=$true)][string]$Account,    # ex: 52431980
  [Parameter(Mandatory=$true)][string]$Expiry,     # ex: 2026-10-02
  [string]$Root = "D:\trading-bot new",
  [string]$EditorProjectRel = "Editor\NashmiConfigEditor",
  [string]$Python = "python"
)

# ===== Paths =====
$EditorProject = Join-Path $Root $EditorProjectRel
$EditorPublish = Join-Path $EditorProject "bin\Release\net8.0-windows\win-x64\publish"
$EditorExeSrc  = Join-Path $EditorPublish "NashmiConfigEditor.exe"
$EditorFolder  = Join-Path $Root "Editor"
$EditorExeDst  = Join-Path $EditorFolder "Editor.exe"

$Tools      = Join-Path $Root "tools"
$Dist       = Join-Path $Root "dist"
$BundleIn   = "D:\bundle_in"
$OutSfx     = Join-Path $Root ("NashmiBot_Installer_{0}.exe" -f $Client)
$ReadmeTxt  = Join-Path $BundleIn "README.txt"

$PublicPemTools = Join-Path $Tools "public.pem"
$PrivatePemTools= Join-Path $Tools "private.pem"

$SevenZipExe = "C:\Program Files\7-Zip\7z.exe"
$SevenZipSfx = "C:\Program Files\7-Zip\7zS.sfx"
$Temp7z      = "D:\bundle.7z"
$SfxConfig   = "D:\config.sfx.txt"

# ===== 0) prereqs =====
Write-Host "== Checking prerequisites ==" -ForegroundColor Cyan
if (!(Test-Path $EditorProject)) { throw "Project path not found: $EditorProject" }
if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) { throw "dotnet SDK not found in PATH" }
if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) { throw "Python not found in PATH (`$Python)" }

# cryptography check (PowerShell-safe)
$pyCheck = & $Python -c "import importlib; import sys; importlib.import_module('cryptography.hazmat.primitives'); print('ok')"
if ($LASTEXITCODE -ne 0 -or ($pyCheck -notlike '*ok*')) {
  throw "Python 'cryptography' package missing. Run: pip install cryptography"
}

# ===== 1) Build Editor =====
Write-Host "== Building Editor (dotnet publish) ==" -ForegroundColor Cyan
Push-Location $EditorProject
dotnet publish -c Release -r win-x64 --self-contained true -p:PublishSingleFile=true | Out-Null
Pop-Location
if (!(Test-Path $EditorExeSrc)) { throw "Build output not found: $EditorExeSrc" }

New-Item -ItemType Directory -Path $EditorFolder -Force | Out-Null
Copy-Item $EditorExeSrc $EditorExeDst -Force

# ===== 2) Generate license (no HWID) =====
Write-Host "== Generating license (no HWID) ==" -ForegroundColor Cyan
New-Item -ItemType Directory -Path $Tools -Force | Out-Null
$GenPy = Join-Path $Tools "generate_license.py"

if (!(Test-Path $GenPy)) {
@"
#!/usr/bin/env python3
import argparse, base64, json
from datetime import datetime
from pathlib import Path
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding

def gen_rsa_keypair(bits=3072):
    priv = rsa.generate_private_key(public_exponent=65537, key_size=bits)
    priv_pem = priv.private_bytes(encoding=serialization.Encoding.PEM, format=serialization.PrivateFormat.PKCS8, encryption_algorithm=serialization.NoEncryption())
    pub_pem = priv.public_key().public_bytes(encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo)
    return priv, priv_pem, pub_pem

def sign_blob(priv_key, blob: bytes) -> bytes:
    return priv_key.sign(blob, padding.PKCS1v15(), hashes.SHA256())

def load_or_create_rsa_keys(keydir: Path):
    priv_path = keydir / "private.pem"; pub_path = keydir / "public.pem"
    if not priv_path.exists() or not pub_path.exists():
        priv, priv_pem, pub_pem = gen_rsa_keypair()
        keydir.mkdir(parents=True, exist_ok=True)
        priv_path.write_bytes(priv_pem); pub_path.write_bytes(pub_pem)
        return priv
    else:
        data = priv_path.read_bytes()
        return serialization.load_pem_private_key(data, password=None)

def make_license(priv, client, account, expiry, outdir: Path, hwid="*"):
    payload = {"client": client, "account": str(account), "hwid": hwid, "expiry": expiry, "issued_at": datetime.utcnow().isoformat()+"Z"}
    payload_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    sig = sign_blob(priv, payload_json)
    lic_obj = {"payload": base64.b64encode(payload_json).decode("ascii"), "sig": base64.b64encode(sig).decode("ascii")}
    outdir.mkdir(parents=True, exist_ok=True)
    lic_filename = outdir / f"license_{client}.json"
    with open(lic_filename, "w", encoding="utf-8") as f: json.dump(lic_obj, f, ensure_ascii=False, indent=2)
    return lic_filename

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--client", required=True); p.add_argument("--account", required=True)
    p.add_argument("--expiry", required=True); p.add_argument("--keydir", default=".")
    p.add_argument("--outdir", default=".")
    args = p.parse_args()
    keydir = Path(args.keydir).resolve(); outdir = Path(args.outdir).resolve()
    priv = load_or_create_rsa_keys(keydir)
    lic = make_license(priv, args.client, args.account, args.expiry, outdir, hwid="*")
    print(str(lic))

if __name__ == "__main__":
    main()
"@ | Set-Content -Path $GenPy -Encoding UTF8
}

$LicOut = & $Python $GenPy --client $Client --account $Account --expiry $Expiry --keydir $Tools --outdir $EditorFolder
$LicOut = (& $Python $GenPy --client $Client --account $Account --expiry $Expiry --keydir $Tools --outdir $EditorFolder 2>&1).Trim()
if ($LASTEXITCODE -ne 0) { throw "License generation failed" } else { Write-Host "== License created at: $LicOut" -ForegroundColor Green }

# ===== 3) bundle_in =====
Write-Host "== Preparing bundle_in ==" -ForegroundColor Cyan
New-Item -ItemType Directory -Path $BundleIn -Force | Out-Null
Copy-Item $EditorExeDst $BundleIn -Force

if (!(Test-Path $PublicPemTools)) { throw "public.pem not found in $Tools" }
Copy-Item $PublicPemTools $BundleIn -Force

$LicFile = Join-Path $EditorFolder ("license_{0}.json" -f $Client)
if (!(Test-Path $LicFile)) { throw "license file not found: $LicFile" }
Copy-Item $LicFile $BundleIn -Force

@"
NashmiBot Editor bundle for client: $Client
------------------------------------------------
This installer extracts to: %ProgramFiles%\NashmiBot
and auto-runs Editor.exe.

1) Click 'Load License' and choose: license_$Client.json
2) Edit settings -> Save encrypted -> config.enc + config.sig
3) Copy outputs to your bot's config folder.
"@ | Set-Content -Path $ReadmeTxt -Encoding UTF8

# ===== 4) SFX =====
Write-Host "== Building one-click SFX installer ==" -ForegroundColor Cyan
if (!(Test-Path $SevenZipExe) -or !(Test-Path $SevenZipSfx)) {
  throw "7-Zip not found. Install it or update paths in the script."
}

if (Test-Path $Temp7z) { Remove-Item $Temp7z -Force }
& "$SevenZipExe" a $Temp7z "$BundleIn\*" | Out-Null

@"
;!@Install@!UTF-8!
Title="NashmiBot Editor"
BeginPrompt=""
RunProgram="Editor.exe"
GUIMode="2"
ExtractTitle="Installing NashmiBot Editor"
InstallPath="%ProgramFiles%\\NashmiBot"
;!@InstallEnd@!
"@ | Set-Content -Path $SfxConfig -Encoding UTF8

cmd /c copy /b "$SevenZipSfx"+"$SfxConfig"+"$Temp7z" "$OutSfx" > nul

Remove-Item $Temp7z -Force
Remove-Item $SfxConfig -Force

New-Item -ItemType Directory -Path $Dist -Force | Out-Null
Copy-Item $OutSfx (Join-Path $Dist ([IO.Path]::GetFileName($OutSfx))) -Force

Write-Host "===================================================" -ForegroundColor Green
Write-Host "✅ Done. One-click installer ready:" -ForegroundColor Green
Write-Host "    $OutSfx" -ForegroundColor Yellow
Write-Host "Also copied to: $Dist" -ForegroundColor Yellow
Write-Host "===================================================" -ForegroundColor Green
Write-Host "Client usage: Double-click the EXE. It auto-extracts to %ProgramFiles%\NashmiBot and runs Editor.exe."
Write-Host "NOTE: Keep PRIVATE key safe: $PrivatePemTools"
