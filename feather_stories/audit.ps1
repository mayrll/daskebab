param([string]$DataDirectory = $PSScriptRoot)

$species = 'Ciconia ciconia'
$register = @(Import-Csv (Join-Path $DataDirectory 'ring_register.csv'))
$storkRings = @($register | Where-Object species -eq $species)
$reads = @(Import-Csv (Join-Path $DataDirectory 'observations.csv') |
    Where-Object { $_.species -eq $species -and $_.ring_read -cmatch '^[A-Z?]{6}$' })

$results = foreach ($site in ($reads | Group-Object locality_id | Where-Object Count -ge 5)) {
    $votes = foreach ($position in 0..5) {
        $counts = @($site.Group | ForEach-Object { [string]$_.ring_read[$position] } |
            Where-Object { $_ -ne '?' } | Group-Object | Sort-Object Count -Descending)
        if (!$counts -or ($counts.Count -gt 1 -and $counts[0].Count -eq $counts[1].Count)) {
            [pscustomobject]@{ Letter = '?'; Count = 0 }
        } else {
            [pscustomobject]@{ Letter = $counts[0].Name; Count = $counts[0].Count }
        }
    }

    $ring = -join $votes.Letter
    if ($ring.Contains('?') -or $ring -in $register.ring) { continue }

    $exact = @($site.Group | Where-Object ring_read -ceq $ring)
    if (!$exact) { continue }

    $nearest = $storkRings | ForEach-Object {
        $candidate = $_.ring
        $distance = @(0..5 | Where-Object { $candidate[$_] -cne $ring[$_] }).Count
        [pscustomobject]@{ Ring = $candidate; Distance = $distance }
    } | Sort-Object Distance | Select-Object -First 1

    [pscustomobject]@{
        Ring = $ring
        Locality = $site.Name
        District = $site.Group[0].district
        Readings = $site.Count
        Exact = $exact.Count
        ObserverGroups = @($exact.observers | Sort-Object -Unique).Count
        NearestRegistered = $nearest.Ring
        Differences = $nearest.Distance
        Votes = ($votes | ForEach-Object { $_.Count }) -join ','
    }
}

$results | Sort-Object Exact, ObserverGroups -Descending | Format-Table -AutoSize

