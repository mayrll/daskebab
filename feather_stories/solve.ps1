param([string]$DataDirectory = $PSScriptRoot)

$reads = @(Import-Csv (Join-Path $DataDirectory 'observations.csv') |
    Where-Object {
        $_.species -eq 'Ciconia ciconia' -and $_.district -eq 'Cahul' -and
        $_.ring_read -cmatch '^[A-Z?]{6}$'
    })
$site = $reads | Group-Object locality_id | Sort-Object Count -Descending | Select-Object -First 1

$ring = -join $(foreach ($position in 0..5) {
    $vote = $site.Group | ForEach-Object { [string]$_.ring_read[$position] } |
        Where-Object { $_ -ne '?' } | Group-Object | Sort-Object Count -Descending
    if (!$vote -or ($vote.Count -gt 1 -and $vote[0].Count -eq $vote[1].Count)) {
        throw "No unique consensus at position $($position + 1)"
    }
    $vote[0].Name
})

$register = @(Import-Csv (Join-Path $DataDirectory 'ring_register.csv'))
if ($ring -in $register.ring) {
    throw "Consensus ring $ring is already in the register"
}
$exact = @($site.Group | Where-Object ring_read -eq $ring).Count
$nearest = $register | Where-Object species -eq 'Ciconia ciconia' | ForEach-Object {
    $candidate = $_.ring
    $distance = @(0..5 | Where-Object { $candidate[$_] -cne $ring[$_] }).Count
    [pscustomobject]@{ Ring = $candidate; Distance = $distance }
} | Sort-Object Distance | Select-Object -First 1

"Site $($site.Name): $($site.Count) readings, $exact exact matches"
"Closest registered White Stork: $($nearest.Ring), $($nearest.Distance) differing letters"
"SIGMOID_$ring"

