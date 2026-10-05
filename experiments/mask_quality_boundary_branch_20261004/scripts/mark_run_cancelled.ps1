param(
    [Parameter(Mandatory=$true)][string]$RunJson,
    [Parameter(Mandatory=$true)][string]$MissingOutput
)
$j = Get-Content -LiteralPath $RunJson -Raw | ConvertFrom-Json
$j.status = "cancelled"
$values = @{
    return_code = 130
    finished_at = [DateTime]::UtcNow.ToString("o")
    error = "stopped by user after epoch 4; epoch 5 incomplete"
    artifact_completeness = "partial"
    missing_outputs = @($MissingOutput)
}
foreach ($entry in $values.GetEnumerator()) {
    if ($j.PSObject.Properties[$entry.Key]) { $j.PSObject.Properties.Remove($entry.Key) }
    $j | Add-Member -MemberType NoteProperty -Name $entry.Key -Value $entry.Value
}
$j | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $RunJson -Encoding UTF8
