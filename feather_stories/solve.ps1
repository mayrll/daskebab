param(
    [string]$DataDirectory = $(if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path })
)

$ErrorActionPreference = "Stop"

$obsPath = Join-Path $DataDirectory "observations.csv"
$regPath = Join-Path $DataDirectory "ring_register.csv"

if (-not (Test-Path $obsPath) -or -not (Test-Path $regPath)) {
    Write-Error "Missing observations.csv or ring_register.csv in $DataDirectory"
    exit 1
}

$species = "Ciconia ciconia"
$register = @(Import-Csv $regPath)
$knownRings = @($register | Where-Object { $_.species -eq $species } | ForEach-Object { $_.ring.Trim().ToUpper() })

$observations = @(Import-Csv $obsPath | Where-Object {
    $_.species -eq $species -and $_.ring_read -cmatch '^[A-Z?]{6}$'
})

# Filter to Lower Prut Biosphere Reserve area (Cahul district / southern Moldova below 46.0N)
$lowerPrut = @($observations | Where-Object {
    $_.district -eq "Cahul" -or [double]$_.latitude -lt 46.0
})

# Exclude sightings matching any registered White Stork (allowing up to 1 misread)
$unregistered = foreach ($row in $lowerPrut) {
    $read = $row.ring_read.Trim().ToUpper()
    $isKnown = $false
    foreach ($known in $knownRings) {
        $errors = 0
        for ($i = 0; $i -lt 6; $i++) {
            if ($read[$i] -ne '?' -and $read[$i] -ne $known[$i]) { $errors++ }
        }
        if ($errors -le 1) { $isKnown = $true; break }
    }
    if (-not $isKnown) { $row }
}

# Select the primary Lower Prut locality (L5819951 - Lake Beleu)
$targetSite = $unregistered | Group-Object locality_id | Sort-Object Count -Descending | Select-Object -First 1

# Reconstruct each of the 6 ring letters via majority voting (ignoring '?')
$chars = for ($pos = 0; $pos -lt 6; $pos++) {
    $targetSite.Group |
        ForEach-Object { [string]$_.ring_read[$pos] } |
        Where-Object { $_ -ne '?' } |
        Group-Object |
        Sort-Object Count -Descending |
        Select-Object -First 1 -ExpandProperty Name
}

$word = -join $chars
Write-Output ("SIGMOID_" + $word)
