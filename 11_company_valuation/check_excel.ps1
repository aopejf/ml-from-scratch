# Opens the workbook in Excel, lets Excel work out every formula, and compares the
# answers with the Python valuation in results.json. Needs Microsoft Excel on this computer.
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
    "{0} {1,-52} Excel {2,16:N4}   Python {3,16:N4}" -f $(if ($ok) { "ok  " } else { "FAIL" }), $label, [double]$excelValue, [double]$pythonValue
}

try {
    $book = $excel.Workbooks.Open((Join-Path $here "company_valuation_model.xlsx"), 0, $true)
    $excel.CalculateFull()
    $v = $book.Worksheets.Item("Valuation")
    $w = $book.Worksheets.Item("WACC")
    $a = $results.assumptions
    $b = $results.base

    # The assumptions Excel derives from the History sheet, against Python's.
    Compare-Value "Sales growth next year" $v.Range("B5").Value2 $a.growth_start 0.0000001
    Compare-Value "Operating margin" $v.Range("B7").Value2 $a.margin 0.0000001
    Compare-Value "Tax rate" $v.Range("B8").Value2 $a.tax_rate 0.0000001
    Compare-Value "Depreciation share" $v.Range("B9").Value2 $a.depreciation_share 0.0000001
    Compare-Value "Buildings and equipment share" $v.Range("B10").Value2 $a.capex_share 0.0000001
    Compare-Value "Working capital share" $v.Range("B11").Value2 $a.working_capital_share 0.0000001
    Compare-Value "Cost of equity" $w.Range("B8").Value2 $results.capital.cost_of_equity 0.0000001
    Compare-Value "Cost of debt" $w.Range("B13").Value2 $results.capital.cost_of_debt 0.0000001
    Compare-Value "WACC" $w.Range("B22").Value2 $a.wacc 0.0000001

    # The base case, year by year and in total. The workbook is in $ millions.
    for ($y = 1; $y -le 10; $y++) {
        Compare-Value "Free cash flow, $($b.rows[$y - 1].year)" $v.Cells.Item(31, 1 + $y).Value2 ($b.rows[$y - 1].free_cash_flow / 1e6) 0.01
    }
    Compare-Value "Ten years of cash, worth today" $v.Range("B36").Value2 ($b.forecast_present / 1e6) 0.01
    Compare-Value "Everything after, worth today" $v.Range("B38").Value2 ($b.terminal_present / 1e6) 0.01
    Compare-Value "Enterprise value" $v.Range("B39").Value2 ($b.enterprise_value / 1e6) 0.01
    Compare-Value "Value per share" $v.Range("B43").Value2 $b.value_per_share 0.0001
    Compare-Value "Model against price" $v.Range("B45").Value2 $b.gap 0.000001
    Compare-Value "Year-ten multiple" $v.Range("B48").Value2 $b.terminal_multiple 0.0001

    # Change the two touchiest assumptions and compare with every cell of the Python sensitivity table.
    $s = $results.sensitivity
    for ($i = 0; $i -lt $s.waccs.Count; $i++) {
        for ($j = 0; $j -lt $s.terminal_growths.Count; $j++) {
            $v.Range("B12").Value2 = $s.waccs[$i]
            $v.Range("B6").Value2 = $s.terminal_growths[$j]
            $excel.Calculate()
            Compare-Value ("Per share at WACC {0:P1}, growth {1:P1}" -f $s.waccs[$i], $s.terminal_growths[$j]) $v.Range("B43").Value2 $s.value_per_share[$i][$j] 0.0001
        }
    }

    # The three stories: growth and margin changed together.
    $v.Range("B12").Value2 = $a.wacc
    $v.Range("B6").Value2 = $a.terminal_growth
    foreach ($name in "Cautious", "Base", "Optimistic") {
        $c = $results.scenarios.$name
        $v.Range("B5").Value2 = $c.growth_start
        $v.Range("B7").Value2 = $c.margin
        $excel.Calculate()
        Compare-Value "Per share, $name story" $v.Range("B43").Value2 $c.value_per_share 0.0001
    }

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
