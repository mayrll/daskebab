$ErrorActionPreference = "Stop"

$obsPath = Join-Path $PSScriptRoot "observations.csv"
$regPath = Join-Path $PSScriptRoot "ring_register.csv"

if (-not (Test-Path $obsPath) -or -not (Test-Path $regPath)) {
    Write-Error "Missing observations.csv or ring_register.csv in $PSScriptRoot"
    exit 1
}

Write-Host "Loading datasets..." -ForegroundColor Cyan
$register = Import-Csv $regPath
$observations = Import-Csv $obsPath

$knownRings = $register | ForEach-Object { $_.ring.Trim().ToUpper() } | Where-Object { $_.Length -eq 6 }

function Test-MatchesKnownRing {
    param([string]$ReadRing, [string[]]$KnownList, [int]$MaxErrors = 1)
    foreach ($known in $KnownList) {
        $errors = 0
        for ($i = 0; $i -lt 6; $i++) {
            $c = $ReadRing[$i]
            if ($c -ne '?' -and $c -ne $known[$i]) {
                $errors++
                if ($errors -gt $MaxErrors) { break }
            }
        }
        if ($errors -le $MaxErrors) { return $true }
    }
    return $false
}

$storkObs = $observations | Where-Object {
    ($_.species -match "Ciconia ciconia" -or $_.common_name -match "White Stork") -and
    -not [string]::IsNullOrWhiteSpace($_.ring_read)
}

Write-Host "Total White Stork ring observations: $($storkObs.Count)"

$unregisteredObs = @()
foreach ($row in $storkObs) {
    $ring = $row.ring_read.Trim().ToUpper()
    if ($ring.Length -ne 6) { continue }
    if (-not (Test-MatchesKnownRing -ReadRing $ring -KnownList $knownRings -MaxErrors 1)) {
        $unregisteredObs += [PSCustomObject]@{
            Ring     = $ring
            District = $row.district
            Lat      = [double]$row.latitude
        }
    }
}

Write-Host "Unregistered observations count: $($unregisteredObs.Count)" -ForegroundColor Yellow

$lowerPrutObs = $unregisteredObs | Where-Object { $_.District -match "Cahul|Cantemir" -or $_.Lat -lt 46.2 }
if ($lowerPrutObs.Count -eq 0) { $lowerPrutObs = $unregisteredObs }

Write-Host "Lower Prut candidate sightings: $($lowerPrutObs.Count)" -ForegroundColor Green

$topSeed = ($lowerPrutObs | Group-Object Ring | Sort-Object Count -Descending | Select-Object -First 1).Name
$cluster = $lowerPrutObs | Where-Object {
    $r = $_.Ring
    $diff = 0
    for ($i = 0; $i -lt 6; $i++) {
        if ($r[$i] -ne '?' -and $topSeed[$i] -ne '?' -and $r[$i] -ne $topSeed[$i]) { $diff++ }
    }
    $diff -le 2
}

Write-Host "Target stork cluster size: $($cluster.Count)" -ForegroundColor Cyan

$finalChars = for ($pos = 0; $pos -lt 6; $pos++) {
    $letterStats = $cluster | ForEach-Object { $_.Ring[$pos] } | Where-Object { $_ -ne '?' } | Group-Object | Sort-Object Count -Descending
    $summary = ($letterStats | ForEach-Object { "$($_.Name)($($_.Count))" }) -join ", "
    Write-Host "Position $($pos + 1): $summary"
    $letterStats[0].Name
}

$recoveredWord = -join $finalChars
Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "Recovered Ring: $recoveredWord" -ForegroundColor Green
Write-Host "Final Flag:     SIGMOID_$recoveredWord" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
