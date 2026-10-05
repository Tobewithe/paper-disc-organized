param([string]$TexBin = 'C:/Users/Administrator/AppData/Local/Programs/MiKTeX/miktex/bin/x64')
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    New-Item -ItemType Directory -Force build | Out-Null
    foreach ($paperDoc in @('main','supplement','title_page')) {
        & "$TexBin/pdflatex.exe" --enable-installer -interaction=nonstopmode -halt-on-error -output-directory=build "$paperDoc.tex" *> "build/compile_$paperDoc.txt"
        if ($LASTEXITCODE -ne 0) { throw "LaTeX failed: build/compile_$paperDoc.txt" }
        if ($paperDoc -eq 'main') {
            & "$TexBin/bibtex.exe" "build/$paperDoc" *> 'build/bibtex_main.txt'
            if ($LASTEXITCODE -ne 0) { throw 'BibTeX failed: build/bibtex_main.txt' }
        }
        1..3 | ForEach-Object {
            & "$TexBin/pdflatex.exe" --disable-installer -interaction=nonstopmode -halt-on-error -output-directory=build "$paperDoc.tex" *> "build/compile_$paperDoc.txt"
            if ($LASTEXITCODE -ne 0) { throw "LaTeX failed: build/compile_$paperDoc.txt" }
        }
        Copy-Item -LiteralPath "build/$paperDoc.pdf" -Destination "$paperDoc.pdf" -Force
        Write-Output "Built $paperDoc.pdf"
    }
} finally { Pop-Location }
