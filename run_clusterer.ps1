<#
.SYNOPSIS
    Kjør video ansiktsgjenkjenning, kryss-videoklynging og tankediagram.

.DESCRIPTION
    PowerShell-skript som forenkler kjøring av video-ansiktsklynging.
    Støtter enkel filvelging, mappebehandling, automatisk aktivering av virtuelt miljø
    og åpning av resultatene i nettleseren.

.PARAMETER Videos
    Én eller flere videofiler (eller filbanemønstre).

.PARAMETER VideoDir
    En mappe som inneholder videoer som skal skannes.

.PARAMETER Output
    Mappen hvor resultatene, tankediagrammet og nettsiden lagres (standard: cross_video_output).

.PARAMETER FrameInterval
    Prøvetakingsfrekvens for frames (standard: 5).

.PARAMETER Method
    Klyngealgoritme: 'dbscan' eller 'agglomerative' (standard: dbscan).

.PARAMETER NoBrowser
    Ikke åpne nettleseren automatisk etter fullført analyse.

.PARAMETER Demo
    Kjør demonstrasjon med de inkluderte testvideoene i sample_videos/.

.EXAMPLE
    .\run_clusterer.ps1
    (Viser interaktiv meny for enkelt valg)

.EXAMPLE
    .\run_clusterer.ps1 -VideoDir "C:\mine_videoer"

.EXAMPLE
    .\run_clusterer.ps1 -Videos "video1.mp4", "video2.mp4" -Output "resultater"
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string[]]$Videos,

    [Parameter()]
    [string]$VideoDir,

    [Parameter()]
    [string]$Output = "cross_video_output",

    [Parameter()]
    [int]$FrameInterval = 5,

    [Parameter()]
    [ValidateSet("dbscan", "agglomerative")]
    [string]$Method = "dbscan",

    [Parameter()]
    [switch]$NoBrowser,

    [Parameter()]
    [switch]$Demo
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "     VIDEO ANSIKTSKLUDERING & TANKEDIAGRAM - POWERSHELL RUNNER" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Finn og sjekk Python
$PythonExe = "python"
if (Test-Path ".\.venv\Scripts\python.exe") {
    $PythonExe = (Resolve-Path ".\.venv\Scripts\python.exe").Path
    Write-Host "[*] Benytter virtuelt miljø: $PythonExe" -ForegroundColor Gray
} elseif (Get-Command "python" -ErrorAction SilentlyContinue) {
    Write-Host "[*] Benytter system-Python" -ForegroundColor Gray
} else {
    Write-Error "Kunne ikke finne Python i PATH eller i .venv. Vennligst installer Python."
    exit 1
}

# 2. Håndter interaktiv modus dersom ingen parametere er oppgitt
if ($Demo) {
    $Videos = @("sample_videos/video_intervju_del1.mp4", "sample_videos/video_intervju_del2.mp4")
} elseif (-not $Videos -and -not $VideoDir) {
    Write-Host "`nVelg hva du onsker a gjore:" -ForegroundColor Yellow
    Write-Host "  [1] Demo: Kjor kryss-video analyse pa sample_videos/ (2 videoer med felles person)"
    Write-Host "  [2] Skann en hel mappe med videoer"
    Write-Host "  [3] Velg videofiler manuelt"
    Write-Host "  [4] Generer nye testvideoer pa nytt (generate_sample_video.py)"
    Write-Host "  [Q] Avslutt"
    
    $Choice = Read-Host "`nSkriv valg [1-4, Q]"
    
    switch ($Choice) {
        "1" {
            if (-not (Test-Path "sample_videos/video_intervju_del1.mp4")) {
                Write-Host "[*] Genererer sample_videos forst..." -ForegroundColor Cyan
                & $PythonExe generate_sample_video.py
            }
            $Videos = @("sample_videos/video_intervju_del1.mp4", "sample_videos/video_intervju_del2.mp4")
        }
        "2" {
            $InputFolder = Read-Host "Skriv inn banen til videomappen"
            if (-not (Test-Path $InputFolder)) {
                Write-Host "Feil: Mappen '$InputFolder' finnes ikke." -ForegroundColor Red
                exit 1
            }
            $VideoDir = $InputFolder
        }
        "3" {
            $InputFiles = Read-Host "Skriv inn stier til videoer (atskilt med komma eller mellomrom)"
            $Videos = $InputFiles -split "[, ]" | Where-Object { $_ -ne "" }
        }
        "4" {
            Write-Host "[*] Genererer testvideoer..." -ForegroundColor Cyan
            & $PythonExe generate_sample_video.py
            Write-Host "[+] Ferdig generert! Kjorer na analysen..." -ForegroundColor Green
            $Videos = @("sample_videos/video_intervju_del1.mp4", "sample_videos/video_intervju_del2.mp4")
        }
        "Q" {
            Write-Host "Avbrutt." -ForegroundColor Gray
            exit 0
        }
        default {
            Write-Host "Ugyldig valg. Avbryter." -ForegroundColor Red
            exit 1
        }
    }
}

# 3. Bygg argumentliste for main.py
$MainArgs = @("main.py", "--output", $Output, "--frame-interval", $FrameInterval, "--method", $Method)

if ($VideoDir) {
    $MainArgs += @("--video-dir", $VideoDir)
}
if ($Videos) {
    $MainArgs += "--videos"
    foreach ($v in $Videos) {
        $MainArgs += $v
    }
}

Write-Host "`n[*] Starter analyse med:" -ForegroundColor Cyan
Write-Host "    $PythonExe $($MainArgs -join ' ')" -ForegroundColor Gray
Write-Host "----------------------------------------------------------------------" -ForegroundColor DarkGray

# 4. Kjor main.py
& $PythonExe $MainArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[!] Analysen stoppet med en feil (kode: $LASTEXITCODE)." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n[+] Fullfort!" -ForegroundColor Green
$IndexHtmlPath = Join-Path (Resolve-Path $Output).Path "index.html"

if (Test-Path $IndexHtmlPath) {
    Write-Host "    - Interaktivt dashbord: $IndexHtmlPath" -ForegroundColor White
    
    if (-not $NoBrowser) {
        Write-Host "[*] Apner resultater i nettleseren..." -ForegroundColor Cyan
        Start-Process $IndexHtmlPath
    }
}
