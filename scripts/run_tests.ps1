Write-Host "Running unit tests..."
pytest tests/unit -m unit
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Running metamorphic tests..."
pytest tests/metamorphic -m metamorphic
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "All tests passed."
