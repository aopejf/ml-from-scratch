# Opens the workbook in Excel, lets Excel work out every formula, and compares the
# answers with the Python analysis in results.json. Needs Microsoft Excel on this computer.
#
#     powershell -File check_excel.ps1
#
# Nothing is saved: the workbook is closed without changes.

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$results = Get-Content (Join-Path $here "results.json") -Raw | ConvertFrom-Json
$excel = New-Object -ComObject Excel.Application
$excel.Visible = $false
$excel.DisplayAlerts = $false
$failed = $false

function Compare-Value($label, $excelValue, $pythonValue, $tolerance) {
    $gap = [math]::Abs([double]$excelValue - [double]$pythonValue)
    $ok = $gap -le $tolerance
    if (-not $ok) { $script:failed = $true }
    "{0} {1,-46} Excel {2,16:N4}   Python {3,16:N4}" -f $(if ($ok) { "ok  " } else { "FAIL" }), $label, [double]$excelValue, [double]$pythonValue
}

try {
    $book = $excel.Workbooks.Open((Join-Path $here "car_wash_finance_model.xlsx"), 0, $true)
    $s = $book.Worksheets.Item("Site 24")
    foreach ($c in $results.site24.candidates) {
        foreach ($case in "Low", "Expected", "High") {
            $s.Range("B4").Value2 = $c.candidate
            $s.Range("B5").Value2 = $case
            $excel.Calculate()
            $p = $c.cases.$case
            Compare-Value "$($c.candidate) $case NPV" $s.Range("E29").Value2 $p.npv 1
            Compare-Value "$($c.candidate) $case IRR" $s.Range("E30").Value2 $p.irr 0.00001
            if ($null -ne $p.payback_years) { Compare-Value "$($c.candidate) $case payback" $s.Range("E31").Value2 $p.payback_years 0.001 }
            elseif ($s.Range("E31").Value2 -ne "over 10 years") { $failed = $true; "FAIL $($c.candidate) $case payback should be over 10 years" }
        }
    }
    $sc = $book.Worksheets.Item("Scorecard")
    $y = $results.chain.by_year."2025"
    Compare-Value "Scorecard: chain revenue 2025" $sc.Range("B28").Value2 $y.revenue 1
    Compare-Value "Scorecard: chain profit 2025" $sc.Range("D28").Value2 $y.site_profit 1
    foreach ($row in $results.scorecard | Where-Object { $_.site_id -in 3, 7, 14, 18 }) {
        $r = 4 + $row.site_id
        Compare-Value "Scorecard: $($row.name) margin" $sc.Range("E$r").Value2 $row.margin 0.0001
        Compare-Value "Scorecard: $($row.name) labour share" $sc.Range("H$r").Value2 $row.labour_share 0.0001
        Compare-Value "Scorecard: $($row.name) cancel rate" $sc.Range("K$r").Value2 $row.churn 0.0001
    }
    $b = $book.Worksheets.Item("Budget vs actual")
    Compare-Value "Budget vs actual: budget profit" $b.Range("B15").Value2 $results.variance.budget_profit 1
    Compare-Value "Budget vs actual: actual profit" $b.Range("C15").Value2 $results.variance.actual_profit 1
    Compare-Value "Revenue split adds up to the difference" $b.Range("D28").Value2 ($b.Range("D5").Value2 + $b.Range("D6").Value2) 1
    $b.Range("B2").Value2 = 11
    $excel.Calculate()
    $v = $results.variance_by_site."11"
    Compare-Value "Budget vs actual: site 11 actual profit" $b.Range("C15").Value2 $v.actual_profit 1
    Compare-Value "Budget vs actual: site 11 budget profit" $b.Range("B15").Value2 $v.budget_profit 1
    $errors = 0
    foreach ($sheet in $book.Worksheets) {
        try { $errors += $sheet.UsedRange.SpecialCells(-4123, 16).Count } catch { }
    }
    if ($errors -gt 0) { $failed = $true }
    "{0} cells showing an Excel error: {1}" -f $(if ($errors -eq 0) { "ok  " } else { "FAIL" }), $errors
}
finally {
    if ($book) { $book.Close($false) }
    $excel.Quit()
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($excel)
}
if ($failed) { "Some checks failed."; exit 1 } else { "Excel and Python agree." }
