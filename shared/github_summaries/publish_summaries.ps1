param(
  [string]$ProjectRoot = 'C:\Dpan\codexproject\paper-disc-organized',
  [string]$PublicationRepo = 'C:\Dpan\codexproject\paper-disc-summary',
  [switch]$Push
)
$ErrorActionPreference = 'Stop'
$Python = 'C:\Dpan\envsfiles\miniconda3\python.exe'
$Exporter = Join-Path $ProjectRoot 'shared\github_summaries\export_summaries.py'
$Context = Join-Path $ProjectRoot 'shared\github_summaries\publication_context.json'
if (-not (Test-Path -LiteralPath $Exporter)) { throw "Exporter not found: $Exporter" }
if (-not (Test-Path -LiteralPath $Context)) { throw "Publication context not found: $Context" }
& $Python $Exporter --project-root $ProjectRoot --output $PublicationRepo --context $Context
if ($LASTEXITCODE -ne 0) { throw "Export failed with exit code $LASTEXITCODE" }
$status = git -C $PublicationRepo status --short
# The exporter records an export timestamp. If that is the only changed file,
# restore it so a no-op refresh does not create an empty documentation commit.
$changed = @(git -C $PublicationRepo diff --name-only)
if ($changed.Count -eq 1 -and $changed[0] -eq 'PUBLICATION.json' -and -not ($status | Where-Object { $_ -match '^\?\?' })) {
  git -C $PublicationRepo restore --source=HEAD -- PUBLICATION.json
  $status = git -C $PublicationRepo status --short
}
if ($status) {
  git -C $PublicationRepo add -A
  git -C $PublicationRepo commit -m 'Refresh research experiment summaries'
  if ($Push) { git -C $PublicationRepo push origin main }
} else {
  Write-Host 'No publication changes.'
}
git -C $PublicationRepo status --short
